"""Responde con numeros las preguntas del EDA dirigido de Ana."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, LABEL_COLUMNS, FAMILIES
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
test = pd.read_parquet(D + "test.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)
FAM_DE = np.array(["sano"] + [FAMILIES[f] for f in FA])
fam_real = FAM_DE[estado]
src = open("generar_submit.py").read()
exec(src.split("Xtr, Xte = construir")[0].split("from scoring import")[1].split("\n", 1)[1])
X = construir(train)                      # fisicas + z
FIS = pd.concat([train[BASE], fisicas(train)], axis=1)   # sin z
print("features: base %d | fisicas %d | con z %d" % (len(BASE), FIS.shape[1], X.shape[1]))

def auc_tab(cols_df, y):
    out = {}
    for c in cols_df.columns:
        v = pd.to_numeric(cols_df[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 50 or len(np.unique(v[ok])) < 2: continue
        try: out[c] = abs(roc_auc_score(y[ok], v[ok]) - 0.5)
        except ValueError: pass
    return pd.Series(out).sort_values(ascending=False)

print("\n" + "=" * 70)
print("P1. Que separa SANO de FALLA")
print("=" * 70)
s = auc_tab(FIS, (estado > 0).astype(int))
print((0.5+s.head(10)).round(3).rename("AUC").to_string())

print("\n" + "=" * 70)
print("P2. Que separa cada FAMILIA del resto (aqui se juega el 70%)")
print("=" * 70)
# SOLO entre ventanas con falla: aisla "que familia" de "hay falla"
enf = estado > 0
for g in ["mechanical", "electrical", "hydraulic", "structural"]:
    s = auc_tab(FIS[enf], (fam_real[enf] == g).astype(int))
    print("\n%s  (n=%d de %d con falla)" % (g.upper(), (fam_real[enf]==g).sum(), enf.sum()))
    print("  " + " | ".join("%s AUC=%.3f" % (k, 0.5+v) for k, v in s.head(5).items()))

print("\n" + "=" * 70)
print("P3. Que separa fallas DENTRO de la misma familia")
print("=" * 70)
PARES = [("F01", "F02"), ("F03", "F04"), ("F03", "F05"), ("F08", "F09"),
         ("F10", "F11"), ("F12", "F13")]
for a, b in PARES:
    ia, ib = FA.index(a) + 1, FA.index(b) + 1
    m = np.isin(estado, [ia, ib])
    s = auc_tab(FIS[m], (estado[m] == ia).astype(int))
    top = s.head(3)
    print("%s vs %s (n=%3d):  %s" % (a, b, m.sum(),
          " | ".join("%s AUC=%.3f" % (k, 0.5+v) for k, v in top.items())))

print("\n" + "=" * 70)
print("P4. Features que NO aportan y P5. redundancia")
print("=" * 70)
todas = auc_tab(FIS, (estado > 0).astype(int))
inutil = [c for c in FIS.columns if todas.get(c, 0) < 0.02]
print("features con |AUC-0.5| < 0.02 contra sano/falla: %d de %d" % (len(inutil), FIS.shape[1]))
print("  ", inutil[:12])
cor = FIS.corr().abs()
corv = cor.to_numpy(copy=True)
np.fill_diagonal(corv, 0)
cor = pd.DataFrame(corv, index=cor.index, columns=cor.columns)
pares = [(cor.columns[i], cor.columns[j], cor.iloc[i, j])
         for i in range(len(cor)) for j in range(i + 1, len(cor)) if cor.iloc[i, j] > 0.95]
print("pares con |corr| > 0.95: %d" % len(pares))
for a, b, v in sorted(pares, key=lambda t: -t[2])[:8]:
    print("   %.3f  %s ~ %s" % (v, a, b))

print("\n" + "=" * 70)
print("P6/P7. Huella de maquina: se puede predecir machine_id desde las features?")
print("=" * 70)
maq = train.machine_id.astype("category").cat.codes.to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
et = lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                  max_features="sqrt", n_jobs=1, random_state=RS)
for nom, M in (("crudas (BASE)", train[BASE]), ("fisicas", FIS),
               ("fisicas + z-score", X)):
    a = cross_val_score(et(), M, maq, cv=cv, scoring="accuracy", n_jobs=1).mean()
    print("  %-20s accuracy prediciendo machine_id (26 clases): %.3f  (azar=%.3f)"
          % (nom, a, 1 / 26))
print("\nSi la accuracy es alta, las features llevan huella de la maquina:")
print("el modelo puede memorizar maquinas del train en vez de aprender la falla.")

m = et(); m.fit(train[BASE], maq)
imp = pd.Series(m.feature_importances_, index=BASE).sort_values(ascending=False)
print("\nFeatures que mas identifican a la maquina (candidatas a normalizar):")
print("  " + " | ".join("%s %.3f" % (k, v) for k, v in imp.head(8).items()))
