"""Arranque en una maquina desconocida. Correlo APENAS te sientas.

La hackathon es presencial en las aulas de informatica: puede que no sea tu
computadora, que no tengas permisos de administrador, o que la version de
Python sea otra. Este script:

  1. reporta que Python y que paquetes hay
  2. instala lo que falte en un venv local (o con --user si no puede)
  3. degrada con elegancia: si LightGBM no entra, el kit usa sklearn
  4. corre un autotest de 20 segundos que valida el kit de punta a punta

Uso:
    python bootstrap.py            # diagnostico + autotest
    python bootstrap.py --install  # ademas instala lo que falte
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ESENCIALES = ["numpy", "pandas", "sklearn"]
DESEABLES = ["lightgbm", "scipy", "matplotlib", "joblib"]
PIP_NAME = {"sklearn": "scikit-learn", "PIL": "pillow"}

RAIZ = Path(__file__).resolve().parent


def estado():
    print("Python  :", sys.version.split()[0], "|", sys.executable)
    falta = []
    for m in ESENCIALES + DESEABLES:
        try:
            mod = importlib.import_module(m)
            print("  [ok]   %-12s %s" % (m, getattr(mod, "__version__", "?")))
        except Exception:
            print("  [FALTA] %-12s" % m)
            falta.append(m)
    return falta


def instalar(paquetes):
    if not paquetes:
        print("nada que instalar")
        return
    nombres = [PIP_NAME.get(p, p) for p in paquetes]
    for flags in ([], ["--user"]):
        cmd = [sys.executable, "-m", "pip", "install", "-q", *flags, *nombres]
        print("$", " ".join(cmd))
        if subprocess.call(cmd) == 0:
            print("instalacion ok")
            return
        print("fallo; reintentando con --user" if not flags else "fallo tambien con --user")
    print("\nNO se pudo instalar. Plan B:")
    print("  - probar en Google Colab (el kit corre igual, subi la carpeta ic_kit/)")
    print("  - o seguir sin lightgbm: el kit cae solo a HistGradientBoosting de sklearn")


def autotest():
    """Valida el kit entero con datos sinteticos. Si esto pasa, estas listo."""
    sys.path.insert(0, str(RAIZ))
    import numpy as np
    import pandas as pd
    from ic_kit import costs, cleaning, tabular, traps
    from ic_kit import submit as sub_mod

    print("\n--- autotest ---")
    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({
        "id": np.arange(n),
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.choice(["Alto", "bajo ", "ALTO", "medio"], n),
    })
    riesgo = df.a * 1.4 + (df.b - 5) * 0.5 + rng.normal(0, 1, n)
    clases = ["baja", "media", "alta", "critica"]
    df["y"] = [clases[i] for i in np.digitize(riesgo, np.quantile(riesgo, [.5, .8, .93]))]
    tr, te = df.iloc[:600].copy(), df.iloc[600:].copy()
    te_y = te.pop("y")

    C = costs.COST_D1
    trc, _ = cleaning.auto_clean(tr, target="y", verbose=False)
    tec, _ = cleaning.auto_clean(te, verbose=False)
    X, y, Xte, ids, cls = tabular.prepare(trc, tec, "y", "id", class_order=clases)
    assert cls == clases, "class_order roto"
    oof, pte, _ = tabular.oof_lgb(X, y, Xte, n_splits=3, seeds=(0,), verbose=False)

    c_argmax = costs.mean_cost(y, oof.argmax(1), C)
    c_bayes = costs.mean_cost(y, costs.bayes_decision(oof, C), C)
    print("  costo OOF  argmax=%.4f  bayes=%.4f" % (c_argmax, c_bayes))
    assert c_bayes <= c_argmax + 1e-9, "bayes_decision no deberia ser peor que argmax"

    auc, _, _ = traps.adversarial_validation(X, Xte, verbose=False)
    print("  adversarial AUC = %.3f" % auc)
    traps.fit_prior_honest(oof, y, C, verbose=False)
    traps.validate_prior_em(oof / oof.sum(1, keepdims=True), y, verbose=False)

    s = sub_mod.make_submission(ids, costs.bayes_decision(pte, C), clases,
                                RAIZ / "work" / "_autotest.csv", id_col="id",
                                target_col="y")
    assert sub_mod.validate(s, expected_ids=ids, allowed_labels=clases), "submit invalido"
    (RAIZ / "work" / "_autotest.csv").unlink(missing_ok=True)

    backend = "lightgbm" if tabular.HAS_LGB else "sklearn (fallback)"
    print("\nAUTOTEST OK  |  backend: %s" % backend)
    print("El kit funciona en esta maquina. A trabajar.")


if __name__ == "__main__":
    falta = estado()
    if "--install" in sys.argv:
        instalar(falta)
        falta = estado()
    duros = [m for m in ESENCIALES if m in falta]
    if duros:
        print("\nFaltan paquetes esenciales: %s" % duros)
        print("Corre:  python bootstrap.py --install")
        sys.exit(1)
    autotest()
