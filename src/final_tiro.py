"""Mejor config medida: base4 (174 col) + cobertura OvR 0.3. Valida y entrega."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS, validate_prediction_package
from fastscore import prep_truth, fast_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
test = pd.read_parquet(D + "test.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)
TP = prep_truth(train)
src = open("generar_submit.py").read()
exec(src.split("Xtr, Xte = construir")[0].split("from scoring import")[1].split("\n", 1)[1])
Xtr, Xte = construir(train), construir(test)
assert list(Xtr.columns) == list(Xte.columns)
print("features:", Xtr.shape[1])

# params de base4 (el que dio 48,115), con determinismo forzado
def mk():
    return [
        lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=600,
            learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
            random_state=RS, deterministic=True, force_col_wise=True, n_jobs=1),
        ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
            max_features="sqrt", n_jobs=1, random_state=RS),
        RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
            max_features="sqrt", n_jobs=1, random_state=RS),
    ]
def mk_ovr():
    return lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=15, min_child_samples=25, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=RS,
        deterministic=True, force_col_wise=True, n_jobs=1)

def softmax14(Xa, ya, Xb):
    P = np.zeros((len(Xb), 14))
    for m in mk():
        m.fit(Xa, ya)
        pp = m.predict_proba(Xb); cls = np.asarray(m.classes_, int)
        Q = np.zeros((len(Xb), 14)); Q[:, cls] = pp; P += Q
    return P / 3

def ovr13(Xa, idx_a, Xb):
    B = np.zeros((len(Xb), 13))
    for j, f in enumerate(FA):
        y = train[f"label_{f}"].to_numpy()[idx_a]
        if y.sum() < 2: continue
        m = mk_ovr(); m.fit(Xa, y); B[:, j] = m.predict_proba(Xb)[:, 1]
    return B

W_OVR = 0.3
rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]
res = {"base4": [], "base4+OvR": []}
for k in CORTES:
    rap = set(rpm_maq.index[k:])
    er = train.machine_id.isin(rap).to_numpy()
    a, b = np.where(~er)[0], np.where(er)[0]
    P = softmax14(Xtr.iloc[a], estado[a], Xtr.iloc[b])
    B = ovr13(Xtr.iloc[a], a, Xtr.iloc[b])
    tp = tuple(t[b] for t in TP)
    r0 = fast_score(P[:, 1:], tp)
    r1 = fast_score((1 - W_OVR) * P[:, 1:] + W_OVR * B, tp)
    res["base4"].append(r0["final_score"]); res["base4+OvR"].append(r1["final_score"])
    print("k=%2d  base4=%.2f   +OvR=%.2f" % (k, r0["final_score"], r1["final_score"]), flush=True)
print("\n=== 5 cortes ===")
for n, v in res.items():
    print("%-12s %.2f +- %.2f" % (n, np.mean(v), np.std(v)))

# ---- entrega ----
GAN = "base4+OvR" if np.mean(res["base4+OvR"]) >= np.mean(res["base4"]) else "base4"
print("\nconfig elegida:", GAN)
P = softmax14(Xtr, estado, Xte)
Pf = P[:, 1:].copy()
if GAN == "base4+OvR":
    B = ovr13(Xtr, np.arange(len(train)), Xte)
    Pf = (1 - W_OVR) * Pf + W_OVR * B
sub = pd.DataFrame({"window_id": test.window_id.to_numpy()})
for j, f in enumerate(FA):
    sub[f] = np.clip(Pf[:, j], 0, 1)
sub = validate_prediction_package(test, sub)
sub.to_csv("submit_final.csv", index=False)
print("submit_final.csv", sub.shape, "validado | suma media %.3f" % sub[FA].sum(1).mean())
