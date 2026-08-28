# -*- coding: utf-8 -*-
"""Prueba las funciones de la sección 10 con datos y señales sintéticas."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from celdas_seccion10 import S

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- escenario
N_MAQ, SES_POR_MAQ, VENT = 8, 5, 7
filas = []
for m in range(N_MAQ):
    for k in range(SES_POR_MAQ):
        st = int(rng.integers(0, 14))
        for w in range(VENT):
            filas.append((f"w{len(filas)}", f"M{m}", f"S{m}_{k}", st))
train = pd.DataFrame(filas, columns=["window_id", "machine_id", "session_id", "st"])
estado = train.pop("st").to_numpy()
n = len(train)

FAULT_IDS = [f"F{i:02d}" for i in range(1, 14)]
FAMILIES = {}
for i, f in enumerate(FAULT_IDS, start=1):
    FAMILIES[f] = ("mechanical" if i <= 6 else "structural" if i == 7 else
                   "electrical" if i <= 11 else "hydraulic")
SEVERITY_COLS = [f"severity_{f}" for f in FAULT_IDS]
for c in SEVERITY_COLS:
    train[c] = 0.0
for i, e in enumerate(estado):
    if e > 0:
        train.loc[i, SEVERITY_COLS[e - 1]] = float(rng.random())

X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"f{i}" for i in range(12)])
test = train[["window_id", "machine_id", "session_id"]].copy()

FOLDS = []
maq = train.machine_id.to_numpy()
for grupo in np.array_split(np.array(sorted(set(maq))), 4):
    b = np.where(np.isin(maq, grupo))[0]
    FOLDS.append((np.setdiff1d(np.arange(n), b), b))

NAIVE = 9.42
def compute_score(df, entrega):
    P = entrega[FAULT_IDS].to_numpy()
    y = np.zeros_like(P)
    idx = np.arange(len(df))
    e = estado[df.index.to_numpy()] if len(df) < n else estado
    y[idx[e > 0], e[e > 0] - 1] = 1
    brier = float(((P - y) ** 2).mean())
    cost = 100 * max(0.0, 1 - (0.5 + P.std()) * 6 / NAIVE)
    prob = 100 * max(0.0, 1 - brier / 0.25)
    diag = 100 * float((P.argmax(1) == y.argmax(1)).mean())
    return {"overall": {"final_score": .7*cost + .2*prob + .1*diag, "cost_score": cost,
                        "prob_score": prob, "diag_score": diag, "brier": brier,
                        "macro_f1": diag/100, "naive_cost": NAIVE}}

def evaluar(p13, idx=None):
    idx = np.arange(n) if idx is None else idx
    e = pd.DataFrame({"window_id": train.window_id.iloc[idx].to_numpy()})
    for j, f in enumerate(FAULT_IDS):
        e[f] = np.clip(p13[idx][:, j], 0, 1)
    return compute_score(train.iloc[idx], e)["overall"]

FAMS = sorted(set(FAMILIES.values()))
FIDX = {g: np.array([i for i, f in enumerate(FAULT_IDS) if FAMILIES[f] == g]) for g in FAMS}
def acciones(P13):
    Sm = P13.sum(1); c = np.empty((len(P13), 3 + len(FAMS)))
    c[:, 0] = 10*Sm; c[:, 1] = 7 + 3.5*Sm; c[:, 2] = 12
    for k, g in enumerate(FAMS):
        c[:, 3+k] = 4 + 8*(Sm - P13[:, FIDX[g]].sum(1))
    nom = np.array(["MONITOR", "DERATE", "STOP"] + ["INSPECT_"+g for g in FAMS])
    return pd.Series(nom[c.argmin(1)])

def ensamblar(oofs, nombres):
    P = np.mean([oofs[x] for x in nombres], axis=0)
    return P / P.sum(1, keepdims=True)

def mostrar(t, index=False):
    return t

class Dummy:
    """Modelo con algo de señal real, para que las cabezas jerárquicas no sean ruido."""
    def fit(self, Xf, y):
        self.classes_ = np.unique(y)
        self.cen = np.array([Xf.to_numpy()[y == c].mean(0) for c in self.classes_])
        return self
    def predict_proba(self, Xf):
        d = ((Xf.to_numpy()[:, None, :] - self.cen[None]) ** 2).sum(-1)
        L = -d / (d.std() + 1e-9)
        L -= L.max(1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(1, keepdims=True)

MODELOS = {"LightGBM": Dummy, "XGBoost": Dummy}
MODELOS_RAPIDOS = MODELOS
CANDIDATOS = {"LGB + RF + RegLogistica": ["LightGBM", "XGBoost"]}

G = dict(np=np, pd=pd, train=train, test=test, estado=estado, X=X, FOLDS=FOLDS,
         FAULT_IDS=FAULT_IDS, FAMILIES=FAMILIES, SEVERITY_COLS=SEVERITY_COLS,
         evaluar=evaluar, acciones=acciones, ensamblar=ensamblar, mostrar=mostrar,
         MODELOS=MODELOS, MODELOS_RAPIDOS=MODELOS_RAPIDOS, CANDIDATOS=CANDIDATOS,
         compute_score=compute_score, display=lambda *a, **k: None)

for k in ("fn_sesion", "fn_costo", "fn_jerarquico", "fn_temp", "fn_pool_z"):
    exec(S[k], G)

ok = lambda t: print(f"  OK  {t}")
print("== 10.2 sesión ==")
Xs = G["agregar_sesion"](X, train)
assert Xs.shape == (n, 36) and np.isfinite(Xs.to_numpy()).all()
assert list(Xs.columns[:12]) == list(X.columns)
sm = Xs["f0__smean"].to_numpy()
for s in train.session_id.unique():
    m = (train.session_id == s).to_numpy()
    assert np.allclose(sm[m], X.f0.to_numpy()[m].mean())
ok("agregar_sesion: media y desvío correctos por sesión")

P = rng.random((n, 14)); P /= P.sum(1, keepdims=True)
for modo in ("aritmetica", "geometrica", "producto"):
    Q = G["poolear_sesion"](P, train, modo)
    assert np.allclose(Q.sum(1), 1) and np.isfinite(Q).all()
    for s in train.session_id.unique()[:3]:
        m = (train.session_id == s).to_numpy()
        assert np.allclose(Q[m], Q[m][0]), f"{modo}: la sesión no quedó uniforme"
    ok(f"poolear_sesion({modo}): normaliza y uniformiza la sesión")
Qa = G["poolear_sesion"](P, train, "aritmetica")
for s in train.session_id.unique()[:3]:
    m = (train.session_id == s).to_numpy()
    assert np.allclose(Qa[m][0], P[m].mean(0))
ok("poolear_sesion(aritmetica) == promedio exacto")
Qp = G["poolear_sesion"](P, train, "producto")
assert Qp.max(1).mean() > Qa.max(1).mean(), "el producto debería afilar más"
ok("el producto afila más que la aritmética")
tab = G["comparar_pooling"](P)
assert list(tab.columns) == ["pooling", "vs_crudo", "final", "cost", "prob", "diag"]
assert len(tab) == 4 and tab.vs_crudo.iloc[0] == 0
ok("comparar_pooling devuelve las 4 filas")

print("== 10.3 costo ==")
Pg = np.zeros((n, 14)); Pg[np.arange(n), estado] = 1.0
costo, acc, fam = G["costo_realizado"](Pg)
assert np.isfinite(costo).all()
sanas, conf = estado == 0, estado > 0
assert (costo[sanas] == 0).all(), "una ventana sana no debería costar nada con el oráculo"
assert np.allclose(costo[conf], 4.0), "el oráculo paga sólo la inspección de $4"
ok("costo_realizado sobre el oráculo: 0 en sanas y $4 en fallas")
assert abs(costo.mean() - conf.mean()*4) < 1e-9
ok(f"costo medio del oráculo = {costo.mean():.4f} = P(falla)x$4")
c2, a2, _ = G["costo_realizado"](np.full((n, 14), 1/14))
assert np.isfinite(c2).all() and set(np.unique(a2)) <= set(
    ["MONITOR", "DERATE", "STOP"] + ["INSPECT_"+g for g in FAMS])
ok("costo_realizado sobre probabilidades planas: acciones válidas")

print("== 10.4 jerárquico ==")
oj = G["correr_jerarquico"](Dummy, X)
assert oj.shape == (n, 14) and np.allclose(oj.sum(1), 1) and np.isfinite(oj).all()
assert (oj >= 0).all()
ok("correr_jerarquico: (n,14), filas normalizadas, sin NaN")
est_idx = np.array([7])
assert oj[:, est_idx].min() > 0, "F07 (familia de una sola falla) quedó en 0"
ok("la familia con una sola falla (structural) recibe masa")
oj2 = G["correr_jerarquico"](Dummy, X, folds=FOLDS[:2])
assert np.allclose(oj2[FOLDS[0][1]].sum(1), 1)
ok("correr_jerarquico acepta folds alternativos")
bm = G["barrer_mezcla"](P, oj, pesos=np.linspace(0, 1, 5))
assert len(bm) == 5 and list(bm.columns) == ["w", "final", "cost", "prob", "diag"]
assert abs(bm.final.iloc[0] - evaluar(P[:, 1:])["final_score"]) < 1e-9
ok("barrer_mezcla: w=0 reproduce el OOF plano")

print("== 10.5 temperatura ==")
assert np.allclose(G["aplicar_T"](P, 1.0), P, atol=1e-9)
ok("aplicar_T(P, 1) es la identidad")
assert G["aplicar_T"](P, 0.5).max(1).mean() > P.max(1).mean() > G["aplicar_T"](P, 2.0).max(1).mean()
ok("T<1 afila y T>1 suaviza")
T = G["calibrar_temperatura"](P, "final_score")
assert 0.4 <= T <= 3.0
assert evaluar(G["aplicar_T"](P, T)[:, 1:])["final_score"] >= evaluar(P[:, 1:])["final_score"] - 1e-6
ok(f"calibrar_temperatura devuelve T={T:.3f} y no empeora el objetivo")

print("== 10.7 z con pool ==")
cols = list(X.columns)
Xp = pd.DataFrame(rng.normal(size=(200, 12)), columns=cols)
mp = rng.choice(train.machine_id.unique(), 200)
z1 = G["z_robusto_con_pool"](X, train.machine_id.to_numpy())
z2 = G["z_robusto_con_pool"](X, train.machine_id.to_numpy(), Xp, mp)
assert z1.shape == z2.shape == X.shape and np.isfinite(z2.to_numpy()).all()
assert not np.allclose(z1.to_numpy(), z2.to_numpy()), "el pool no cambió nada"
m0 = (train.machine_id == "M0").to_numpy()
assert abs(np.median(z1.to_numpy()[m0, 0])) < 1e-9, "la mediana por máquina no quedó en 0"
ok("z_robusto_con_pool: forma, finitud, mediana 0 por máquina, y el pool cambia el resultado")
Xc = X.copy(); Xc["const"] = 3.0
zc = G["z_robusto_con_pool"](Xc, train.machine_id.to_numpy())
assert np.isfinite(zc.to_numpy()).all()
ok("columna constante (MAD=0 y std=0) no produce NaN ni inf")

print("\nTODAS LAS PRUEBAS DE LÓGICA PASARON")
