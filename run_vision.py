"""Baseline cost-sensitive de vision (estilo Desafio 2). Correr en Colab con GPU.

    !pip -q install timm
    !python run_vision.py --train data/train --submit data/only_submit \
        --cost d2 --model convnext_tiny.fb_in22k --size 320 --epochs 8 --folds 3

Genera work/submit.csv, work/oof_vision.npy y un informe de errores por reino.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ic_kit import costs, submit as sub_mod, vision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="carpeta train/ con subdirs por clase")
    ap.add_argument("--submit", required=True, help="carpeta only_submit/ con las imagenes")
    ap.add_argument("--cost", default="d2")
    ap.add_argument("--model", default="convnext_tiny.fb_in22k")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="work/submit.csv")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--label-map", default=None,
                    help="JSON {nombre_carpeta: etiqueta_entera}. Sin esto se usa "
                         "el orden alfabetico de las carpetas.")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. datos
    paths, y, classes, groups = vision.build_index(args.train)
    bad = vision.corrupt_images(paths)
    if bad:
        print("[AVISO] %d imagenes corruptas, se descartan" % len(bad))
        keep = [i for i, p in enumerate(paths) if p not in set(bad)]
        paths = [paths[i] for i in keep]
        y, groups = y[keep], groups[keep]

    label_map = json.loads(Path(args.label_map).read_text()) if args.label_map else \
        {c: i for i, c in enumerate(classes)}
    print("mapeo carpeta -> etiqueta del submit:", label_map)
    print("!! VERIFICA ESTE MAPEO CONTRA LA TABLA DE LA CONSIGNA !!")

    if args.cost == "d2":
        C = costs.COST_D2
        kingdoms = costs.KINGDOM_D2
    else:
        C = pd.read_csv(args.cost, header=None).to_numpy(dtype=float)
        kingdoms = np.zeros(len(classes), dtype=int)
    assert C.shape[0] == len(classes), \
        "la matriz de costos es %s pero hay %d carpetas" % (C.shape, len(classes))

    sub_paths = vision.list_images(args.submit)
    print("%d imagenes a predecir" % len(sub_paths))

    # ----------------------------------------------- 2. CV agrupado por hash
    n = len(paths)
    oof = np.zeros((n, len(classes)))
    pte = np.zeros((len(sub_paths), len(classes)))
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    for f, (tr, va) in enumerate(skf.split(np.arange(n), y, groups)):
        print("\n===== fold %d/%d  (train %d / val %d) =====" % (f + 1, args.folds, len(tr), len(va)))
        dl_tr, dl_va = vision.make_loaders(paths, y, tr, va, img_size=args.size,
                                           batch=args.batch)
        model = vision.train_fold(dl_tr, dl_va, len(classes), args.model,
                                  epochs=args.epochs, lr=args.lr)
        oof[va] = vision.predict_tta(model, [paths[i] for i in va], args.size, args.batch)
        pte += vision.predict_tta(model, sub_paths, args.size, args.batch) / args.folds
        c = costs.mean_cost(y[va], costs.bayes_decision(oof[va], C), C)
        print("fold %d  costo (bayes) = %.4f" % (f + 1, c))

    # ------------------------------------------------- 3. decision + reporte
    from ic_kit.tabular import fit_prior
    w = fit_prior(oof, y, C)
    pred_oof = costs.bayes_decision(oof * w, C)
    print("\ncosto OOF  argmax = %.4f  ->  bayes+pesos = %.4f"
          % (costs.mean_cost(y, oof.argmax(1), C), costs.mean_cost(y, pred_oof, C)))
    print("accuracy OOF = %.4f" % float((oof.argmax(1) == y).mean()))
    kk = np.asarray(kingdoms)
    print("errores DE REINO (los que cuestan 5): %.2f%%"
          % (100 * float((kk[pred_oof] != kk[y]).mean())))
    print("\n" + costs.confusion_cost_report(y, pred_oof, C, labels=classes))

    print("\n=== ETIQUETAS SOSPECHOSAS (revisalas a ojo, rinde mucho) ===")
    order, info = vision.suspect_labels(oof, y, C, top=40)
    for i in order[:20]:
        print("  %-60s  asignada=%-28s modelo=%-28s" %
              (paths[i].name, classes[y[i]], classes[costs.bayes_decision(oof, C)[i]]))
    pd.DataFrame({"archivo": [str(paths[i]) for i in order],
                  "clase_asignada": [classes[y[i]] for i in order],
                  "clase_modelo": [classes[j] for j in costs.bayes_decision(oof, C)[order]]}
                 ).to_csv(work / "etiquetas_sospechosas.csv", index=False)

    # ------------------------------------------------------------ 4. submit
    pred = costs.bayes_decision(pte * w, C)
    etiquetas = [label_map[classes[i]] for i in pred]
    s = sub_mod.make_submission([p.name for p in sub_paths], etiquetas,
                                path=args.out, id_col="Image", target_col="Predict")
    sub_mod.validate(s, allowed_labels=sorted(set(label_map.values())))
    np.save(work / "oof_vision.npy", oof)
    np.save(work / "pte_vision.npy", pte)
    print("\nListo. Registra el envio con ic_kit.submit.SubmitLog(lower_is_better=True).")


if __name__ == "__main__":
    main()
