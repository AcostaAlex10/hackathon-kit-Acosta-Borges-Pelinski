# -*- coding: utf-8 -*-
"""Regresión: las cabezas jerárquicas tienen que funcionar con XGBoost.

XGBoost >= 2 exige etiquetas 0..k-1 y rechaza clases no contiguas. La cabeza
interna de cada familia se entrenaba con los estados crudos, que para la familia
eléctrica son [8, 9, 10, 11]: con xgboost 3.x eso levanta
"Invalid classes inferred from unique values of `y`".
"""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from celdas_seccion10 import S
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(3)
FAULT_IDS = [f"F{i:02d}" for i in range(1, 14)]
FAMILIES = {f: ("mechanical" if i <= 6 else "structural" if i == 7 else
                "electrical" if i <= 11 else "hydraulic") for i, f in enumerate(FAULT_IDS, 1)}
FAMS_ORD = sorted(set(FAMILIES.values()))
FAM_IDX = {g: i for i, g in enumerate(FAMS_ORD)}
FAM_DE_ESTADO = np.array(["sano"] + [FAMILIES[f] for f in FAULT_IDS])
ESTADOS_DE_FAM = {g: np.array([k for k in range(1, 14)
                               if FAMILIES[FAULT_IDS[k - 1]] == g]) for g in FAMS_ORD}

n = 900
estado = rng.integers(0, 14, n)                       # incluye la familia eléctrica: 8..11
X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
X_te = pd.DataFrame(rng.normal(size=(200, 8)), columns=X.columns)
maq = rng.choice([f"M{i}" for i in range(10)], n)
FOLDS = [(np.setdiff1d(np.arange(n), np.where(np.isin(maq, g))[0]),
          np.where(np.isin(maq, g))[0])
         for g in np.array_split(np.array(sorted(set(maq))), 4)]

XGB = lambda: xgb.XGBClassifier(n_estimators=20, max_depth=3, tree_method="hist",
                                verbosity=0, random_state=0)
RF = lambda: make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestClassifier(n_estimators=20, random_state=0))

G = dict(np=np, pd=pd, estado=estado, FOLDS=FOLDS, FAM_IDX=FAM_IDX,
         FAM_DE_ESTADO=FAM_DE_ESTADO, FAMS_ORD=FAMS_ORD, ESTADOS_DE_FAM=ESTADOS_DE_FAM)
exec(S["fn_jerarquico"].split("def barrer_mezcla")[0], G)
exec(S["fn_predecir_jer"].split("# Entrega final")[0], G)
ok = lambda t: print("  OK ", t)

print("estados de la familia electrica:", ESTADOS_DE_FAM["electrical"],
      " <- no contiguos desde 0")

for nom, ctor in (("XGBoost", XGB), ("RandomForest", RF)):
    oof = G["correr_jerarquico"](ctor, X)
    assert oof.shape == (n, 14) and np.allclose(oof.sum(1), 1) and np.isfinite(oof).all()
    ok(f"correr_jerarquico con {nom}: (n,14), normalizado, sin NaN")
    P = G["predecir_jerarquico"](ctor, X, X_te)
    assert P.shape == (200, 14) and np.allclose(P.sum(1), 1) and np.isfinite(P).all()
    ok(f"predecir_jerarquico con {nom}: (n_te,14), normalizado, sin NaN")

# La masa de cada familia tiene que caer en SUS columnas, no corridas.
P = G["predecir_jerarquico"](XGB, X, X_te)
for g in FAMS_ORD:
    assert P[:, ESTADOS_DE_FAM[g]].sum() > 0, f"la familia {g} quedo sin masa"
ok("las cuatro familias reciben masa en sus columnas")
print("\nTODAS LAS PRUEBAS DEL JERARQUICO PASARON")
