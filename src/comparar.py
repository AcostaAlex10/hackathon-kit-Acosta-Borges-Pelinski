"""Enfoque de Ana (z-score) vs fisicas vs combinacion. Mismo protocolo."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS, FAMILIES
from fastscore import prep_truth, fast_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
test = pd.read_parquet(D + "test.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)
TP = prep_truth(train)
FAM_DE = np.array(["sano"] + [FAMILIES[f] for f in FA])
exec(open("generar_submit.py").read().split("Xtr, Xte = construir")[0]
     .split('from scoring import')[1].split('\n', 1)[1])   # trae fisicas() y BASE

def zmaq(df, cols):
    g = df.groupby("machine_id")[cols]
    z = (df[cols] - g.transform("median")) / (g.transform("std") + 1e-9)
    z.columns = [c + "__z" for c in cols]
    return z

FEATS_ANA = [c for c in train.columns
             if c not in set(["window_id", "machine_id", "session_id", "is_normal", "is_combo"]
                             + LABEL_COLUMNS + [f"severity_{f}" for f in FA])]

def build(df, modo):
    if modo == "ana_z":          # exactamente lo de Ana
        return pd.concat([df[FEATS_ANA], zmaq(df, FEATS_ANA)], axis=1)
    fis = pd.concat([df[BASE], fisicas(df)], axis=1)
    if modo == "fis":
        return fis
    if modo == "fis_z":          # combinacion
        tmp = fis.copy(); tmp["machine_id"] = df["machine_id"].to_numpy()
        return pd.concat([fis, zmaq(tmp, list(fis.columns))], axis=1)
    raise ValueError(modo)

rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
es_rap = train.machine_id.isin(set(rpm_maq.index[13:])).to_numpy()
IDX_SHIFT = (np.where(~es_rap)[0], np.where(es_rap)[0])

def mk_lgb():
    return lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=500,
        learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1,
        random_state=RS, deterministic=True, force_col_wise=True, n_jobs=1)
def mk_et():
    return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
        max_features="sqrt", n_jobs=1, random_state=RS)

def oof_gkf(X, ctors, seeds=(42, 2024, 7)):
    acc = np.zeros((len(train), 14))
    for s in seeds:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=s)
        for a, b in cv.split(X, estado, train.machine_id.to_numpy()):
            for ct in ctors:
                m = ct(); m.fit(X.iloc[a], estado[a])
                acc[b] += m.predict_proba(X.iloc[b]) / len(ctors)
    return acc / len(seeds)

def oof_shift(X, ctors):
    a, b = IDX_SHIFT
    o = np.zeros((len(train), 14))
    for ct in ctors:
        m = ct(); m.fit(X.iloc[a], estado[a])
        o[b] += m.predict_proba(X.iloc[b]) / len(ctors)
    return o

def sesion(P, df):
    L = pd.DataFrame(np.log(np.clip(P, 1e-9, 1))); L["s"] = df.session_id.to_numpy()
    M = np.exp(L.groupby("s").transform("mean").to_numpy()); return M / M.sum(1, keepdims=True)

def sc(P, idx):
    r = fast_score(P[idx][:, 1:], tuple(x[idx] for x in TP))
    return r["final_score"], float((FAM_DE[estado[idx]] == FAM_DE[P[idx].argmax(1)]).mean())

ALL = np.arange(len(train))
CT = [mk_lgb, mk_et]
print("%-14s %-22s %-22s" % ("features", "GroupKFold(3 sem)", "shift de regimen"))
print("-" * 60)
res = {}
for modo in ("ana_z", "fis", "fis_z"):
    X = build(train, modo)
    g, s = oof_gkf(X, CT), oof_shift(X, CT)
    gs = sesion(g, train); ss = sesion(s, train)
    fg, ag = sc(g, ALL); fs, as_ = sc(s, IDX_SHIFT[1])
    fg2, _ = sc(gs, ALL); fs2, _ = sc(ss, IDX_SHIFT[1])
    res[modo] = (fg, fs, fg2, fs2)
    print("%-14s %6.2f (fam %.3f)      %6.2f (fam %.3f)   [%d cols]"
          % (modo, fg, ag, fs, as_, X.shape[1]))
    print("%-14s %6.2f  + sesion        %6.2f  + sesion" % ("", fg2, fs2))
print()
print("Referencia real: el submit de Ana (ana_z, ensamble) obtuvo 43.42 en el servidor.")
for m, (fg, fs, fg2, fs2) in res.items():
    print("  %-8s GKF=%.2f -> si el desfasaje GKF->servidor fuese el de Ana (%.2f), daria %.2f"
          % (m, fg2, res["ana_z"][2] - 43.42, fg2 - (res["ana_z"][2] - 43.42)))
