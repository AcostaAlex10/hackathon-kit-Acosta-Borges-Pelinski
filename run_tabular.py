"""Baseline cost-sensitive de punta a punta para un desafio tabular.

Uso tipico (desafio 1, triaje):

  python run_tabular.py --train data/train_labeled.csv --test data/test_features.csv \
      --target nivel_urgencia --id id_paciente --cost d1 --out work/submit.csv

Otros costos:
  --cost d2                     matriz por reino (10 clases del desafio 2)
  --cost accuracy               0/1 (equivale a argmax, para comparar)
  --cost path/a/matriz.csv      matriz propia C[real, predicho]

Salidas: work/audit.csv, work/oof.npy, work/report.txt y el submit.csv.
El objetivo de este script es tener un numero honesto en la primera hora.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ic_kit import cleaning, costs, submit as sub_mod, tabular


def load_cost(spec: str, k: int) -> np.ndarray:
    if spec == "d1":
        return costs.COST_D1
    if spec == "d2":
        return costs.COST_D2
    if spec == "accuracy":
        return 1.0 - np.eye(k)
    return pd.read_csv(spec, header=None).to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--unlabeled", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", dest="id_col", default=None)
    ap.add_argument("--cost", default="d1")
    ap.add_argument("--out", default="work/submit.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--classes", default=None,
                    help="orden de clases separado por ; DEBE coincidir con el orden "
                         "de filas/columnas de la matriz de costos. Con --cost d1 se "
                         "usa el orden clinico por defecto.")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)
    print("train %s  test %s" % (tr.shape, te.shape))

    # ---------------------------------------------------------- 1. auditoria
    aud = cleaning.audit(tr, target=args.target, df_test=te)
    aud.to_csv(work / "audit.csv", index=False)
    flagged = aud[aud.alertas != ""]
    print("\n=== AUDITORIA (columnas con alertas) ===")
    print(flagged.to_string(index=False) if len(flagged) else "  sin alertas")
    print("\n=== DUPLICADOS ===")
    print(cleaning.duplicate_report(tr, args.target, ignore=(args.id_col,) if args.id_col else ()))

    # ------------------------------------------------------------ 2. limpieza
    if not args.no_clean:
        print("\n=== LIMPIEZA ===")
        tr, _ = cleaning.auto_clean(tr, target=args.target,
                                    drop_cols=[c for c in aud[aud.alertas.str.contains("CONSTANTE")].col])
        te, _ = cleaning.auto_clean(te, target=None,
                                    drop_cols=[c for c in aud[aud.alertas.str.contains("CONSTANTE")].col],
                                    verbose=False)

    class_order = args.classes.split(";") if args.classes else (
        costs.LABELS_D1 if args.cost == "d1" else
        list(range(10)) if args.cost == "d2" else None)
    if class_order is not None:
        vistas = set(map(str, tr[args.target].dropna().unique()))
        if not vistas <= set(map(str, class_order)):
            print("[AVISO] las etiquetas del train no coinciden con el orden por defecto")
            print("        train:", sorted(vistas))
            print("        esperado:", class_order)
            print("        -> pasa --classes \"a;b;c\" en el orden de la matriz de costos")
            class_order = None
        else:
            class_order = [c for c in class_order
                           if str(c) in vistas or len(vistas) == len(class_order)]
            tipo = type(next(iter(tr[args.target].dropna())))
            class_order = [tipo(c) if tipo is not str else c for c in class_order]
    X, y, Xte, ids, classes = tabular.prepare(tr, te, args.target, args.id_col,
                                              class_order=class_order)
    k = len(classes)
    C = load_cost(args.cost, k)
    assert C.shape == (k, k), "la matriz de costos es %s pero hay %d clases" % (C.shape, k)
    print("\nclases (orden interno):", classes)
    print("features:", X.shape[1], "| distribucion train:",
          np.round(np.bincount(y, minlength=k) / len(y), 3))

    # ------------------------------------------------- 3. modelos + decision
    seeds = tuple(42 + i for i in range(args.seeds))
    print("\n=== MODELO A: sin pesos (probabilidades calibradas) ===")
    oof_a, pte_a, _ = tabular.oof_lgb(X, y, Xte, n_splits=args.folds, seeds=seeds)
    print("\n=== MODELO B: con sample_weight derivado del costo ===")
    w_cls = costs.class_weights_from_cost(C)
    oof_b, pte_b, _ = tabular.oof_lgb(X, y, Xte, n_splits=args.folds, seeds=seeds,
                                      sample_weight=w_cls[y])

    results = {}
    for name, oof, pte in (("A_sin_pesos", oof_a, pte_a), ("B_con_pesos", oof_b, pte_b)):
        results[name] = dict(oof=oof, pte=pte,
                             cost=costs.mean_cost(y, costs.bayes_decision(oof, C), C))
        g = costs.decision_gain(y, oof, C)
        print("%-12s  argmax=%.5f  bayes=%.5f  (cambia %.1f%% de las filas)"
              % (name, g["cost_argmax"], g["cost_bayes"], 100 * g["pct_changed"]))

    print("\n=== BLEND + AJUSTE DE PESOS DE DECISION ===")
    bw, _ = tabular.blend_weights([oof_a, oof_b], y, C)
    oof_bl = bw[0] * oof_a + bw[1] * oof_b
    pte_bl = bw[0] * pte_a + bw[1] * pte_b
    w = tabular.fit_prior(oof_bl, y, C)

    final_oof_cost = costs.mean_cost(y, costs.bayes_decision(oof_bl * w, C), C)
    baseline = costs.mean_cost(y, oof_bl.argmax(1), C)
    print("\ncosto OOF  argmax simple = %.5f   ->   pipeline completo = %.5f  (%.1f%% mejor)"
          % (baseline, final_oof_cost, 100 * (1 - final_oof_cost / max(baseline, 1e-9))))
    if args.cost == "d1":
        print("SCORE OOF estimado (desafio 1) = %.4f" % (1 / (1 + final_oof_cost)))

    print("\n=== DE DONDE VIENE EL COSTO (atacar de arriba hacia abajo) ===")
    print(costs.confusion_cost_report(y, costs.bayes_decision(oof_bl * w, C), C,
                                      labels=[str(c) for c in classes]))

    # ------------------------------------------------------------- 4. submit
    pred = costs.bayes_decision(pte_bl * w, C)
    s = sub_mod.make_submission(ids, pred, classes, args.out,
                                id_col=args.id_col or "id",
                                target_col=args.target)
    print("\n=== VALIDACION DEL ENVIO ===")
    train_dist = pd.Series(np.bincount(y, minlength=k) / len(y),
                           index=[str(c) for c in classes])
    sub_mod.validate(s, expected_ids=ids, allowed_labels=classes,
                     train_dist=train_dist)

    np.save(work / "oof.npy", oof_bl * w)
    np.save(work / "pte.npy", pte_bl * w)
    (work / "report.json").write_text(json.dumps(
        {"oof_cost": final_oof_cost, "oof_cost_argmax": baseline,
         "blend_w": bw.tolist(), "decision_w": w.tolist(),
         "classes": [str(c) for c in classes]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\nArtefactos en %s/ . Registra el envio con ic_kit.submit.SubmitLog." % work)


if __name__ == "__main__":
    main()
