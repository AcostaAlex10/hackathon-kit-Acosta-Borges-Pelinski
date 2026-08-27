"""Calibracion (temperatura e isotonica) medida con el protocolo de 5 cortes.

Pipeline evaluado: ensamble lgb+et+rf softmax 14 clases sobre 174 columnas
(fisicas + z-score, la config de base4). La calibracion se ajusta con OOF
anidado DENTRO de las maquinas de ajuste (3-fold agrupado) y se aplica a las
maquinas de validacion: nunca ve el lado de validacion.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS, FAMILIES
from fastscore import prep_truth, fast_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)
TP = prep_truth(train)
src = open("generar_submit.py").read()
exec(src.split("Xtr, Xte = construir")[0].split("from scoring import")[1].split("\n", 1)[1])
X = construir(train)
G = train.machine_id.to_numpy()
print("X:", X.shape)

def mk_models():
    return [
        lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=500,
            learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1,
            random_state=RS, deterministic=True, force_col_wise=True, n_jobs=1),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
            max_features="sqrt", n_jobs=1, random_state=RS),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
            max_features="sqrt", n_jobs=1, random_state=RS),
    ]

def ens_fit_predict(idx_fit, idx_pred):
    """Alinea predict_proba a las 14 clases aunque falte alguna en el fold."""
    P = np.zeros((len(idx_pred), 14))
    for m in mk_models():
        m.fit(X.iloc[idx_fit], estado[idx_fit])
        pp = m.predict_proba(X.iloc[idx_pred])
        cls = np.asarray(m.classes_, int)
        Q = np.zeros((len(idx_pred), 14)); Q[:, cls] = pp
        P += Q
    P /= 3
    return P

def score13(P14, idx):
    r = fast_score(P14[:, 1:], tuple(t[idx] for t in TP))
    return r

def temp_scale(P, T):
    L = np.log(np.clip(P, 1e-12, 1)) / T
    Q = np.exp(L - L.max(1, keepdims=True))
    return Q / Q.sum(1, keepdims=True)

def fit_isotonic(P_oof, y14):
    cals = []
    for k in range(14):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(P_oof[:, k], (y14 == k).astype(float))
        cals.append(ir)
    return cals

def apply_isotonic(P, cals):
    Q = np.column_stack([c.predict(P[:, k]) for k, c in enumerate(cals)])
    Q = np.clip(Q, 1e-9, 1)
    return Q / Q.sum(1, keepdims=True)

rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]
res = {m: [] for m in ["sin_calibrar", "temperatura", "isotonica"]}
Tgrid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.7]

for k in CORTES:
    rapidas = set(rpm_maq.index[k:])
    es_rap = train.machine_id.isin(rapidas).to_numpy()
    fit_idx, val_idx = np.where(~es_rap)[0], np.where(es_rap)[0]
    # OOF anidado dentro del fit (3-fold por maquina)
    Poof = np.zeros((len(fit_idx), 14))
    gkf = GroupKFold(n_splits=3)
    for a, b in gkf.split(fit_idx, estado[fit_idx], G[fit_idx]):
        Poof[b] = ens_fit_predict(fit_idx[a], fit_idx[b])
    y_fit = estado[fit_idx]
    # calibradores ajustados SOLO con el lado fit
    best_T, best_s = 1.0, -1
    for T in Tgrid:
        s = fast_score(temp_scale(Poof, T)[:, 1:], tuple(t[fit_idx] for t in TP))["final_score"]
        if s > best_s: best_s, best_T = s, T
    cals = fit_isotonic(Poof, y_fit)
    # modelo completo sobre fit, aplicado a validacion
    Pval = ens_fit_predict(fit_idx, val_idx)
    r0 = score13(Pval, val_idx)
    rT = score13(temp_scale(Pval, best_T), val_idx)
    rI = score13(apply_isotonic(Pval, cals), val_idx)
    res["sin_calibrar"].append(r0); res["temperatura"].append(rT); res["isotonica"].append(rI)
    print("k=%d  sin=%.2f  temp(T=%.2f)=%.2f  iso=%.2f   | prob: %.1f -> T %.1f / iso %.1f"
          % (k, r0["final_score"], best_T, rT["final_score"], rI["final_score"],
             r0["prob_score"], rT["prob_score"], rI["prob_score"]), flush=True)

print("\n=== RESUMEN (media +- sd sobre 5 cortes) ===")
for m, L in res.items():
    fs = [r["final_score"] for r in L]
    pr = [r["prob_score"] for r in L]
    co = [r["cost_score"] for r in L]
    dg = [r["diag_score"] for r in L]
    print("%-14s final %.2f +- %.2f | cost %.2f | prob %.2f | diag %.2f"
          % (m, np.mean(fs), np.std(fs), np.mean(co), np.mean(pr), np.mean(dg)))
