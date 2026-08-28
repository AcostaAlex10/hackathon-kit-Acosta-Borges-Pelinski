"""lgb+et+rf vs lgb+rf+lr sobre las 174 columnas, con los dos jueces."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS
from fastscore import prep_truth, fast_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)
TP = prep_truth(train)
src = open("generar_submit.py").read()
exec(src.split("Xtr, Xte = construir")[0].split("from scoring import")[1].split("\n", 1)[1])
X = construir(train).replace([np.inf, -np.inf], np.nan).fillna(0.0)
print("X:", X.shape)

def LGB(): return lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=600,
    learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=RS,
    deterministic=True, force_col_wise=True, n_jobs=1)
def ET(): return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=RS)
def RF(): return RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=RS)
def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3, random_state=RS))

COMBOS = {"lgb+et+rf": [LGB, ET, RF], "lgb+rf+lr": [LGB, RF, LR], "lr sola": [LR]}

def proba(ctors, Xa, ya, Xb):
    P = np.zeros((len(Xb), 14))
    for ct in ctors:
        m = ct(); m.fit(Xa, ya); pp = m.predict_proba(Xb)
        cls = np.asarray(m.classes_, int); Q = np.zeros((len(Xb), 14)); Q[:, cls] = pp; P += Q
    return P / len(ctors)

rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
PART = []
for k in (11, 12, 13, 14, 15):
    er = train.machine_id.isin(set(rpm_maq.index[k:])).to_numpy()
    PART.append((np.where(~er)[0], np.where(er)[0]))

print("\n%-12s %-22s %s" % ("ensamble", "GroupKFold (3 sem)", "5 cortes de regimen"))
print("-" * 62)
for nom, ct in COMBOS.items():
    # GroupKFold, 3 semillas: el juez que engaña
    oof = np.zeros((len(train), 14))
    for s in (42, 2024, 7):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=s)
        o = np.zeros((len(train), 14))
        for a, b in cv.split(X, estado, train.machine_id.to_numpy()):
            o[b] = proba(ct, X.iloc[a], estado[a], X.iloc[b])
        oof += o
    oof /= 3
    g = fast_score(oof[:, 1:], TP)["final_score"]
    # 5 cortes por regimen: el juez valido
    v = [fast_score(proba(ct, X.iloc[a], estado[a], X.iloc[b])[:, 1:],
                    tuple(t[b] for t in TP))["final_score"] for a, b in PART]
    print("%-12s %6.2f                 %6.2f +- %.2f   [%s]"
          % (nom, g, np.mean(v), np.std(v), " ".join("%.1f" % x for x in v)), flush=True)
