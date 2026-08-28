"""Ablation BASE / Z / FIS / FIS+Z / Top-K FIS, con el protocolo de 5 cortes.

Aviso: quitar features por redundancia ya se midio que ROMPE este ensamble
(45.44 -> 44.45 en el intento 4). Con max_features="sqrt", las copias
correlacionadas aumentan la probabilidad de que una variable util entre en cada
split: para bagging la redundancia no es ruido, es muestreo. Por eso la ablation
se mide con el ENSAMBLE real, no con un modelo suelto, que fue el error de v4.
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

BASEc = list(BASE)
FIS = fisicas(train)
FISc = list(FIS.columns)
tab = pd.concat([train[BASEc], FIS], axis=1)
tmp = tab.copy(); tmp["machine_id"] = train.machine_id.to_numpy()
Zall = zmaq(tmp, list(tab.columns))
print("BASE %d | FIS %d | Z %d" % (len(BASEc), len(FISc), Zall.shape[1]))

# ranking por MI_14 del EDA del equipo
TOP = ["S_ap","I_por_rpm","I_por_flow","p_in_n","flow_n",
       "acc_radial_b__rms_n","vib_tot","acc_radial_a__rms_n","acc_axial__rms_n","rad_b_a",
       "ax_rad","acc_radial_a__crest_n","hidr_pot","acc_radial_a__1x_r","acc_radial_b__1x_r",
       "acc_radial_b__crest_n","p_ratio","head_n","acc_radial_a__2x1x","flow_head"]
TOP = [c for c in TOP if c in FISc]
Zb = Zall[[c + "__z" for c in BASEc]]

def mat(nombre):
    if nombre == "BASE":            return train[BASEc]
    if nombre == "BASE+Z":          return pd.concat([train[BASEc], Zb], axis=1)
    if nombre == "BASE+FIS":        return tab
    if nombre == "BASE+FIS+Z":      return pd.concat([tab, Zall], axis=1)
    if nombre.startswith("BASE+Z+top"):
        k = int(nombre.split("top")[1])
        return pd.concat([train[BASEc], Zb, FIS[TOP[:k]]], axis=1)
    raise ValueError(nombre)

CONFIGS = ["BASE","BASE+Z","BASE+FIS","BASE+FIS+Z",
           "BASE+Z+top5","BASE+Z+top10","BASE+Z+top15","BASE+Z+top20"]

def mk():
    return [lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=600,
              learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
              random_state=RS, deterministic=True, force_col_wise=True, n_jobs=1),
            ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt",
              n_jobs=1, random_state=RS),
            RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt",
              n_jobs=1, random_state=RS)]

def proba(Xa, ya, Xb):
    P = np.zeros((len(Xb), 14))
    for m in mk():
        m.fit(Xa, ya); pp = m.predict_proba(Xb); cls = np.asarray(m.classes_, int)
        Q = np.zeros((len(Xb), 14)); Q[:, cls] = pp; P += Q
    return P / 3

rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]
PART = []
for k in CORTES:
    er = train.machine_id.isin(set(rpm_maq.index[k:])).to_numpy()
    PART.append((np.where(~er)[0], np.where(er)[0]))

res = {}
for nom in CONFIGS:
    X = mat(nom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    v = []
    for (a, b) in PART:
        P = proba(X.iloc[a], estado[a], X.iloc[b])
        v.append(fast_score(P[:, 1:], tuple(t[b] for t in TP))["final_score"])
    res[nom] = v
    print("%-14s %5d col   %.2f +- %.2f   [%s]"
          % (nom, X.shape[1], np.mean(v), np.std(v), " ".join("%.1f" % x for x in v)), flush=True)

print("\n=== ordenado ===")
for n in sorted(res, key=lambda n: -np.mean(res[n])):
    print("%-14s %.2f +- %.2f" % (n, np.mean(res[n]), np.std(res[n])))
ref = np.mean(res["BASE+FIS+Z"])
print("\nreferencia BASE+FIS+Z (lo que se envia) = %.2f" % ref)
for n in CONFIGS:
    d = np.mean(res[n]) - ref
    gana = sum(1 for i in range(5) if res[n][i] > res["BASE+FIS+Z"][i])
    print("  %-14s %+.2f   gana en %d de 5 cortes" % (n, d, gana))
