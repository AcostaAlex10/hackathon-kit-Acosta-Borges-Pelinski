"""Sobre base4+OvR: reponderacion por familia, mas semillas, mas diversidad.

La familia vale el 70% de la metrica y su accuracy esta en ~0.68, con un techo
de +12.8 puntos. El modelo de 14 estados no esta entrenado para acertar familia;
un modelo dedicado de 5 clases si. La reponderacion inyecta esa estimacion mejor
conservando la forma dentro de la familia.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS, FAMILIES
from fastscore import prep_truth, fast_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

FAMS = ["sano"] + sorted(set(FAMILIES.values()))          # 5 clases
FAM_DE = np.array(["sano"] + [FAMILIES[f] for f in FA])   # estado -> familia
fam_id = np.array([FAMS.index(g) for g in FAM_DE])        # estado(14) -> familia(5)
y_fam = fam_id[estado]
print("X:", X.shape, "| familias:", FAMS)

def lgbm(nc, seed, ne=600, nl=31):
    return lgb.LGBMClassifier(objective="multiclass", num_class=nc, n_estimators=ne,
        learning_rate=0.05, num_leaves=nl, min_child_samples=20, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
        random_state=seed, deterministic=True, force_col_wise=True, n_jobs=1)
def et(seed): return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
    max_features="sqrt", n_jobs=1, random_state=seed)
def rf(seed): return RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
    max_features="sqrt", n_jobs=1, random_state=seed)

def proba_k(models, Xa, ya, Xb, k):
    P = np.zeros((len(Xb), k))
    for m in models:
        m.fit(Xa, ya)
        pp = m.predict_proba(Xb); cls = np.asarray(m.classes_, int)
        Q = np.zeros((len(Xb), k)); Q[:, cls] = pp; P += Q
    return P / len(models)

def ovr13(Xa, idx_a, Xb):
    B = np.zeros((len(Xb), 13))
    for j, f in enumerate(FA):
        y = train[f"label_{f}"].to_numpy()[idx_a]
        if y.sum() < 2: continue
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
            num_leaves=15, min_child_samples=25, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=RS,
            deterministic=True, force_col_wise=True, n_jobs=1)
        m.fit(Xa, y); B[:, j] = m.predict_proba(Xb)[:, 1]
    return B

def reponderar(P14, Pfam, alpha):
    """Ajusta la masa de cada familia hacia la que estima el modelo dedicado,
    conservando la forma relativa dentro de cada familia."""
    if alpha == 0: return P14
    M = np.zeros((len(P14), 5))
    for g in range(5):
        M[:, g] = P14[:, fam_id == g].sum(1)
    r = (np.clip(Pfam, 1e-9, 1) / np.clip(M, 1e-9, 1)) ** alpha
    Q = P14 * r[:, fam_id]
    return Q / Q.sum(1, keepdims=True)

def sc(P13, idx):
    return fast_score(P13, tuple(t[idx] for t in TP))

rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]
W = 0.3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
res = {}
def add(n, v): res.setdefault(n, []).append(v)

for k in CORTES:
    rap = set(rpm_maq.index[k:])
    er = train.machine_id.isin(rap).to_numpy()
    a, b = np.where(~er)[0], np.where(er)[0]
    Xa, Xb = X.iloc[a], X.iloc[b]
    # base: ensamble 3 modelos, semilla 42
    P1 = proba_k([lgbm(14, RS), et(RS), rf(RS)], Xa, estado[a], Xb, 14)
    B = ovr13(Xa, a, Xb)
    base = (1 - W) * P1[:, 1:] + W * B
    add("base4+OvR", sc(base, b)["final_score"])
    # A: multi-semilla (3 semillas x 3 modelos)
    P3 = np.mean([proba_k([lgbm(14, s), et(s), rf(s)], Xa, estado[a], Xb, 14)
                  for s in (42, 2024, 7)], 0)
    ms = (1 - W) * P3[:, 1:] + W * B
    add("+ 3 semillas", sc(ms, b)["final_score"])
    # B: reponderacion por familia (modelo dedicado de 5 clases)
    Pf = proba_k([lgbm(5, RS), et(RS), rf(RS)], Xa, y_fam[a], Xb, 5)
    accF_dir = (y_fam[b] == Pf.argmax(1)).mean()
    accF_14 = (y_fam[b] == fam_id[P3.argmax(1)]).mean()
    for al in ALPHAS[1:]:
        Q = reponderar(P3, Pf, al)
        add("+ familia a=%.2f" % al, sc((1 - W) * Q[:, 1:] + W * B, b)["final_score"])
    print("k=%2d  base=%.2f  3sem=%.2f | accFam: 14clases=%.3f  dedicado=%.3f"
          % (k, res["base4+OvR"][-1], res["+ 3 semillas"][-1], accF_14, accF_dir), flush=True)

print("\n=== promedio de los cinco cortes ===")
orden = sorted(res, key=lambda n: -np.mean(res[n]))
for n in orden:
    v = res[n]
    print("%-20s %.2f +- %.2f" % (n, np.mean(v), np.std(v)))
