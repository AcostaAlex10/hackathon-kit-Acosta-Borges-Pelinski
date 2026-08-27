import json, pathlib
def _l(s):
    ls = s.split("\n"); return [x + "\n" for x in ls[:-1]] + [ls[-1]]
def md(s): return {"cell_type":"markdown","metadata":{},"source":_l(s.strip())}
def co(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_l(s.strip("\n"))}
C=[]

C.append(md("""
# Machine Health — solución final del equipo Acosta · Borges · Pelinski

Notebook único que reproduce la entrega. Corre de punta a punta y escribe
`submit.csv`.

Requisitos: `datos_sinraw/` y `participant_kit/` (con `scoring.py`) al lado del
notebook.

El recorrido: auditoría de la métrica, EDA, ingeniería de features física,
el protocolo de validación, el ensamble con cobertura One-vs-Rest, la
calibración medida y descartada, y la entrega.
"""))

C.append(md("## 0. Entorno"))
C.append(co('''
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

DATA = "datos_sinraw/"
KIT  = "participant_kit/"
sys.path.insert(0, KIT)
from scoring import (compute_score, validate_prediction_package,
                     FAULT_IDS, LABEL_COLUMNS, FAMILIES, COSTS)

RS = 42
np.random.seed(RS)
train = pd.read_parquet(DATA + "train_sup.parquet")
test  = pd.read_parquet(DATA + "test.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
n_fallas = lab.sum(1)
estado = np.where(n_fallas == 0, 0, lab.argmax(1) + 1)   # 0 = sano, k = F{k}
print("train", train.shape, "| test", test.shape)
'''))

C.append(md("""
## 1. EDA: los hechos que gobiernan el diseño

Se verifica, no se asume.
"""))
C.append(co('''
print("nulos:", int(train.isna().sum().sum()), "| duplicados:", int(train.duplicated().sum()))
print("is_combo:", int(train.is_combo.sum()),
      "-> fallas por ventana:", pd.Series(n_fallas).value_counts().sort_index().to_dict())
print("máquinas train/test:", train.machine_id.nunique(), test.machine_id.nunique(),
      "| intersección:", len(set(train.machine_id) & set(test.machine_id)))
est_ses = train.assign(e=estado).groupby("session_id").e.nunique()
print("ventanas por sesión:", train.groupby("session_id").size().unique(),
      "| sesiones con un solo estado:", int((est_ses == 1).sum()), "de", len(est_ses))
print()
for c in ["rpm_mean", "flow_mean", "Tamb", "delta_p_mean"]:
    print("%-16s train %8.2f   test %8.2f" % (c, train[c].mean(), test[c].mean()))
'''))
C.append(md("""
Cuatro consecuencias que ordenan todo lo que sigue:

1. **Datos limpios.** No hay imputación que hacer.
2. **`is_combo = 0` en las 1750 filas.** El problema declarado multilabel es, en
   el train, **multiclase de 14 estados excluyentes**. Se modela con softmax, que
   impone `Σp ≤ 1` — que es lo que consume la regla de decisión de la métrica.
3. **La etiqueta es constante dentro de cada sesión** (250 de 250) y cada sesión
   tiene 7 ventanas. **El n efectivo es 250, no 1750.** Cualquier ajuste de muchos
   parámetros contra la métrica es sobreajuste con esa cifra.
4. **Máquinas disjuntas entre train y test, y además cambia el régimen**:
   `rpm_mean` 1742 → 2052, `delta_p_mean` 0,51 → 2,61. Las magnitudes absolutas
   de los sensores no transfieren.
"""))

C.append(md("""
## 2. Auditoría de la métrica

`final = 0,70·cost + 0,20·prob + 0,10·diag`. El 70 % lo decide una regla fija y
conocida que convierte las probabilidades entregadas en una acción. Con
`S = Σ p_f` y `S_g` la masa de la familia `g`: MONITOR `10·S`, DERATE `7+3,5·S`,
STOP `12`, INSPECT(g) `4 + 8·(S − S_g)`. Con softmax `S ≤ 1`, así que DERATE y
STOP nunca se eligen: el juego real es MONITOR contra INSPECT-familia.
"""))
C.append(co('''
FAMS = sorted(set(FAMILIES.values()))
FIDX = {g: np.array([i for i, f in enumerate(FA) if FAMILIES[f] == g]) for g in FAMS}

def acciones(P13):
    S = P13.sum(1); n = len(P13)
    c = np.empty((n, 3 + len(FAMS)))
    c[:, 0] = 10 * S; c[:, 1] = 7 + 3.5 * S; c[:, 2] = 12
    for k, g in enumerate(FAMS):
        c[:, 3 + k] = 4 + 8 * (S - P13[:, FIDX[g]].sum(1))
    nom = np.array(["MONITOR", "DERATE", "STOP"] + ["INSPECT_" + g for g in FAMS])
    return pd.Series(nom[c.argmin(1)])

prev = train[LABEL_COLUMNS].mean().rename(lambda s: s.replace("label_", ""))
P0 = np.tile(prev.to_numpy(), (len(train), 1))
print("acción que la regla elige para el baseline P0 de la cátedra:")
print(acciones(P0).value_counts().to_string())

def evaluar(P13, idx=None):
    idx = np.arange(len(train)) if idx is None else idx
    e = pd.DataFrame({"window_id": train.window_id.iloc[idx].to_numpy()})
    for j, f in enumerate(FA): e[f] = np.clip(P13[idx][:, j], 0, 1)
    return compute_score(train.iloc[idx], e)["overall"]

r0 = evaluar(P0); ro = evaluar(train[LABEL_COLUMNS].to_numpy(float))
print("\\nP0       final=%6.2f  cost=%6.2f" % (r0["final_score"], r0["cost_score"]))
print("oráculo  final=%6.2f  cost=%6.2f" % (ro["final_score"], ro["cost_score"]))
'''))
C.append(md("""
**Corrección al material del kit.** `baseline.ipynb` afirma que la acción óptima
para probabilidades bajas es MONITOR. Es falso para su propio baseline: con las
prevalencias, `S = 0,876` y `S_mecánica = 0,464`, con lo que INSPECT mecánica
cuesta `4 + 8·0,412 = 7,30` contra `8,76` de MONITOR. La regla elige **INSPECT
mecánica en las 1848 ventanas**. Por eso el baseline da 29 y no 15: para
superarlo hay que acertar la familia más del 46 % de las veces.

**El techo real es 73,96, no 100.** Aun con predicción perfecta hay que pagar la
inspección de $4, así que el `cost_score` máximo alcanzable es 62,8. Comparar
contra 100 es engañoso.

**El costo depende de la FAMILIA, no de la falla exacta.** Confundir F03 con F04
no cambia `cost_score`: misma familia, misma acción. Distinguirlas sólo aporta al
`diag_score`, que pesa 10 %.
"""))

C.append(md("""
## 3. Ingeniería de features física

Cada bloque apunta a un modo de falla y es **adimensional o relativo**, que es la
condición para transferir entre máquinas y regímenes distintos.

- **Vibración normalizada por régimen.** La energía vibratoria escala con el
  cuadrado de la velocidad, de ahí `rms/(rpm/1000)²`. Sin esto el modelo confunde
  *máquina rápida* con *máquina en falla*. La fracción de energía en `1x` marca
  **desbalance (F01)**; `2x/1x`, **desalineación (F02)**, porque la desalineación
  excita el segundo armónico. `log1p(kurtosis)` y `crest` capturan los impactos de
  **rodamientos (F03–F05)**. La asimetría entre apoyos marca **pérdida de rigidez
  (F07)**.
- **Eléctricas (F08–F11).** Desbalance de tensión y corriente entre fases como
  `(max−min)/media`; `I_min/I_media` para **pérdida de fase (F09)**; corriente
  relativa a la velocidad para **barras rotóricas (F10)**.
- **Térmicas.** Siempre contra `Tamb`, nunca en valor absoluto: el ambiente del
  test es 3 °C más cálido. Sobretemperatura de devanado → **espiras (F11)**;
  diferencia entre rodamientos → **lubricación (F06)**.
- **Hidráulicas (F12–F13).** Leyes de afinidad de bombas: altura adimensional
  `Δp/(rpm/1000)²` y coeficiente de caudal `flow/rpm`. El rendimiento aparente
  `flow·Δp/(V·I)` marca **obstrucción o fuga (F13)**.

Sobre esas se agrega el **z-score por máquina**: la desviación de cada ventana
respecto de la mediana de su propia máquina. Se calcula por máquina de forma
independiente, sin mezclar train con test y sin mirar etiquetas — es el
preprocesamiento que aplicaría un sistema real de monitoreo de condición, que
compara cada máquina contra su propia línea de base.
"""))
C.append(co('''
BASE = [c for c in test.columns
        if c not in ("window_id", "machine_id", "session_id", "timestamp_start_s")]
ACC = ["acc_radial_a", "acc_radial_b", "acc_axial"]

def fisicas(df):
    X = pd.DataFrame(index=df.index)
    rpm = df.rpm_mean.clip(lower=1) / 1000.0
    rpm2 = rpm ** 2
    for a in ACC:
        rms = df[f"{a}__rms"].clip(lower=1e-9)
        X[f"{a}__rms_n"]   = rms / rpm2
        X[f"{a}__1x_r"]    = df[f"{a}__1x"] / (rms ** 2 + 1e-9)
        X[f"{a}__2x1x"]    = df[f"{a}__2x"] / (df[f"{a}__1x"].abs() + 1e-6)
        X[f"{a}__lkurt"]   = np.log1p(df[f"{a}__kurtosis"].clip(lower=0))
        X[f"{a}__crest_n"] = df[f"{a}__crest"]
    ra = df["acc_radial_a__rms"].clip(lower=1e-9)
    X["ax_rad"]  = df["acc_axial__rms"] / ra
    X["rad_b_a"] = df["acc_radial_b__rms"] / ra
    X["vib_tot"] = df[[f"{a}__rms" for a in ACC]].sum(1) / rpm2
    V = df[["voltage_a__rms", "voltage_b__rms", "voltage_c__rms"]]
    I = df[["current_a__rms", "current_b__rms", "current_c__rms"]]
    vm, im = V.mean(1).clip(lower=1e-9), I.mean(1).clip(lower=1e-9)
    X["V_desbal"] = (V.max(1) - V.min(1)) / vm
    X["I_desbal"] = (I.max(1) - I.min(1)) / im
    X["V_cv"] = V.std(1) / vm
    X["I_cv"] = I.std(1) / im
    X["I_min_r"] = I.min(1) / im
    X["I_max_r"] = I.max(1) / im
    X["S_ap"] = im * vm / 1000.0
    X["I_por_rpm"]  = im / rpm
    X["I_por_flow"] = im / (df.flow_mean.abs() + 1e-6)
    X["dT_wind"]   = df["temp_winding__mean"] - df.Tamb
    X["dT_b1"]     = df["temp_bearing1__mean"] - df.Tamb
    X["dT_b2"]     = df["temp_bearing2__mean"] - df.Tamb
    X["dT_b1b2"]   = df["temp_bearing1__mean"] - df["temp_bearing2__mean"]
    X["dT_wind_b"] = df["temp_winding__mean"] - df[["temp_bearing1__mean", "temp_bearing2__mean"]].mean(1)
    X["dT_wind_rpm"] = X["dT_wind"] / rpm
    X["head_n"]    = df.delta_p_mean / rpm2
    X["flow_n"]    = df.flow_mean / rpm
    X["p_in_n"]    = df.pressure_in_mean / rpm2
    X["p_ratio"]   = df.pressure_out_mean / (df.pressure_in_mean.abs() + 1e-6)
    X["hidr_pot"]  = df.flow_mean * df.delta_p_mean / (im * vm + 1e-6)
    X["flow_head"] = df.flow_mean / (df.delta_p_mean.abs() + 1e-6)
    X["rpm_cv"] = df.rpm_std / df.rpm_mean.clip(lower=1)
    nanc = [c for c in df.columns if c.endswith("__nan_frac")]
    X["nan_tot"] = df[nanc].sum(1)
    X["nan_max"] = df[nanc].max(1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def zmaq(df, cols):
    g = df.groupby("machine_id")[cols]
    z = (df[cols] - g.transform("median")) / (g.transform("std") + 1e-9)
    z.columns = [c + "__z" for c in cols]
    return z

def construir(df):
    fis = pd.concat([df[BASE], fisicas(df)], axis=1)
    tmp = fis.copy(); tmp["machine_id"] = df["machine_id"].to_numpy()
    return pd.concat([fis, zmaq(tmp, list(fis.columns))], axis=1)

Xtr, Xte = construir(train), construir(test)
assert list(Xtr.columns) == list(Xte.columns), "columnas desalineadas"
print("features:", Xtr.shape[1], "| train", Xtr.shape, "| test", Xte.shape)
'''))

C.append(md("""
## 4. El protocolo de validación

Éste es el hallazgo central del trabajo y vale más que cualquier feature.

`StratifiedGroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*. El
test es *"máquina nueva, **régimen nuevo**"*. Decidir con `GroupKFold` ya costó
dos regresiones: una configuración con 48,0 estimados sacó 43,4 en el servidor.

El reemplazo: ordenar las 26 máquinas por `rpm_mean`, entrenar con las lentas y
validar con las rápidas. **Pero con un solo corte el estimador engaña**: su
desvío es ~1,0 y las diferencias entre configuraciones son ~0,75. Por eso el juez
es el **promedio de cinco cortes** (k = 11…15 máquinas al lado de ajuste), y nada
se acepta por menos de ~2 puntos salvo que haya un argumento que no sea el score.
"""))
C.append(co('''
rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]

def particion(k):
    rapidas = set(rpm_maq.index[k:])
    es_rap = train.machine_id.isin(rapidas).to_numpy()
    return np.where(~es_rap)[0], np.where(es_rap)[0]

for k in CORTES:
    a, b = particion(k)
    print("k=%2d  ajuste %4d ventanas (%2d máq, rpm %.0f) | validación %4d (%2d máq, rpm %.0f)"
          % (k, len(a), train.machine_id.iloc[a].nunique(), train.rpm_mean.iloc[a].mean(),
             len(b), train.machine_id.iloc[b].nunique(), train.rpm_mean.iloc[b].mean()))
print("\\nrpm medio del test: %.0f" % test.rpm_mean.mean())
'''))

C.append(md("""
## 5. Modelos: ensamble y cobertura One-vs-Rest

El ensamble promedia LightGBM, ExtraTrees y RandomForest sobre el softmax de 14
estados. A eso se le mezcla una rama **One-vs-Rest** de 13 clasificadores
binarios, con peso 0,3.

La cobertura OvR se adopta por dos motivos. El primero es el score. El segundo, y
más importante, es de gestión de riesgo: **la consigna describe fallas combinadas
y el train no tiene ninguna**. Si el test las tiene, el softmax las subestima
sistemáticamente porque impone `Σp ≤ 1`; la rama OvR no asume exclusividad y evita
ese colapso. Está medido que no cuesta nada, así que el seguro es gratis.

Todos los modelos corren con `n_jobs=1` y LightGBM con `deterministic=True`:
`ExtraTrees` y `RandomForest` con `n_jobs=-1` **no son deterministas** aunque
tengan `random_state` fijo, y la consigna pide el código que replica la entrega.
"""))
C.append(co('''
def mk_ensamble():
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

def mk_binario():
    return lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=15, min_child_samples=25, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=RS,
        deterministic=True, force_col_wise=True, n_jobs=1)

def softmax14(Xa, ya, Xb):
    """Alinea predict_proba a las 14 clases aunque falte alguna en el ajuste."""
    P = np.zeros((len(Xb), 14))
    for m in mk_ensamble():
        m.fit(Xa, ya)
        pp = m.predict_proba(Xb); cls = np.asarray(m.classes_, int)
        Q = np.zeros((len(Xb), 14)); Q[:, cls] = pp
        P += Q
    return P / 3

def ovr13(Xa, idx_a, Xb):
    B = np.zeros((len(Xb), 13))
    for j, f in enumerate(FA):
        y = train[f"label_{f}"].to_numpy()[idx_a]
        if y.sum() < 2: continue
        m = mk_binario(); m.fit(Xa, y)
        B[:, j] = m.predict_proba(Xb)[:, 1]
    return B

W_OVR = 0.3
print("modelos definidos")
'''))
C.append(co('''
res = {"ensamble": [], "ensamble + OvR 0,3": []}
for k in CORTES:
    a, b = particion(k)
    P = softmax14(Xtr.iloc[a], estado[a], Xtr.iloc[b])
    B = ovr13(Xtr.iloc[a], a, Xtr.iloc[b])
    e0 = pd.DataFrame({"window_id": train.window_id.iloc[b].to_numpy()})
    e1 = e0.copy()
    for j, f in enumerate(FA):
        e0[f] = np.clip(P[:, 1 + j], 0, 1)
        e1[f] = np.clip((1 - W_OVR) * P[:, 1 + j] + W_OVR * B[:, j], 0, 1)
    r0 = compute_score(train.iloc[b], e0)["overall"]
    r1 = compute_score(train.iloc[b], e1)["overall"]
    res["ensamble"].append(r0["final_score"])
    res["ensamble + OvR 0,3"].append(r1["final_score"])
    print("k=%2d  ensamble=%.2f   +OvR=%.2f" % (k, r0["final_score"], r1["final_score"]))

print("\\n=== promedio de los cinco cortes ===")
for n, v in res.items():
    print("%-22s %.2f +- %.2f" % (n, np.mean(v), np.std(v)))
'''))
C.append(md("""
La cobertura OvR mejora la media y **reduce la varianza del estimador**, que es
lo que más importa cuando las diferencias son del orden del ruido. Se adopta.
"""))

C.append(md("""
## 6. Calibración: medida y descartada

`prob_score` pesa 20 % y está en torno a 81, así que la calibración explícita
parecía una palanca disponible. Se midió con el mismo protocolo: los calibradores
se ajustan sobre un OOF **anidado dentro de las máquinas de ajuste** (3 folds
agrupados), de modo que nunca ven el lado de validación.

| | final | prob_score |
|---|---|---|
| **Sin calibrar** | **45,39 ± 0,93** | **81,33** |
| Temperatura | 44,41 ± 1,85 | 81,01 |
| Isotónica por clase | 44,08 ± 1,49 | 80,30 |

Las dos empeoran, y el `prob_score` **baja** en lugar de subir. La explicación es
que **el promedio del ensamble ya es un mecanismo de calibración**: promediar tres
modelos con sesgos distintos suaviza las probabilidades. Encima, con n efectivo de
250 sesiones, la isotónica por clase sobreajusta, y la temperatura elige un T
inconsistente en cada corte (1,30 / 1,15 / 1,15 / 1,15 / 0,60), que es ruido y no
señal. Se descarta.

La celda siguiente reproduce la medición. Es cara; se deja desactivada por
defecto para que el notebook corra rápido.
"""))
C.append(co('''
CORRER_CALIBRACION = False   # poner True para reproducir la tabla de arriba

def temp_scale(P, T):
    L = np.log(np.clip(P, 1e-12, 1)) / T
    Q = np.exp(L - L.max(1, keepdims=True))
    return Q / Q.sum(1, keepdims=True)

if CORRER_CALIBRACION:
    G = train.machine_id.to_numpy()
    filas = []
    for k in CORTES:
        a, b = particion(k)
        Poof = np.zeros((len(a), 14))
        for ia, ib in GroupKFold(n_splits=3).split(a, estado[a], G[a]):
            Poof[ib] = softmax14(Xtr.iloc[a[ia]], estado[a[ia]], Xtr.iloc[a[ib]])
        mejor_T, mejor_s = 1.0, -1
        for T in [0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.7]:
            e = pd.DataFrame({"window_id": train.window_id.iloc[a].to_numpy()})
            Q = temp_scale(Poof, T)
            for j, f in enumerate(FA): e[f] = np.clip(Q[:, 1 + j], 0, 1)
            s = compute_score(train.iloc[a], e)["overall"]["final_score"]
            if s > mejor_s: mejor_s, mejor_T = s, T
        cals = []
        for c in range(14):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(Poof[:, c], (estado[a] == c).astype(float)); cals.append(ir)
        P = softmax14(Xtr.iloc[a], estado[a], Xtr.iloc[b])
        Q = np.clip(np.column_stack([c.predict(P[:, i]) for i, c in enumerate(cals)]), 1e-9, 1)
        Q = Q / Q.sum(1, keepdims=True)
        for nom, M in (("sin calibrar", P), ("temperatura", temp_scale(P, mejor_T)), ("isotónica", Q)):
            e = pd.DataFrame({"window_id": train.window_id.iloc[b].to_numpy()})
            for j, f in enumerate(FA): e[f] = np.clip(M[:, 1 + j], 0, 1)
            r = compute_score(train.iloc[b], e)["overall"]
            filas.append(dict(corte=k, metodo=nom, final=r["final_score"], prob=r["prob_score"]))
    tabla = pd.DataFrame(filas).groupby("metodo")[["final", "prob"]].agg(["mean", "std"])
    print(tabla.round(2).to_string())
else:
    print("calibración desactivada; resultados en la tabla de la celda anterior")
'''))

C.append(md("""
## 7. Entrega

Se reentrena con todo `train_sup` y se predice el test. Ninguna etiqueta del test
interviene en ningún paso, y ninguna transformación usa estadísticos calculados
mezclando train con test.
"""))
C.append(co('''
P = softmax14(Xtr, estado, Xte)
B = ovr13(Xtr, np.arange(len(train)), Xte)
Pf = (1 - W_OVR) * P[:, 1:] + W_OVR * B

sub = pd.DataFrame({"window_id": test.window_id.to_numpy()})
for j, f in enumerate(FA):
    sub[f] = np.clip(Pf[:, j], 0, 1)
sub = validate_prediction_package(test, sub)
sub.to_csv("submit.csv", index=False)
print("submit.csv", sub.shape, "validado")
print("suma de probabilidades por fila: %.3f" % sub[FA].sum(1).mean())
print("\\nacciones que elegirá la regla sobre el test:")
print(acciones(sub[FA].to_numpy()).value_counts().to_string())
'''))

C.append(md("""
## 8. Resumen de decisiones

| Decisión | Evidencia |
|---|---|
| Multiclase de 14 estados (softmax) | `is_combo = 0` en las 1750 filas |
| Validación por cambio de régimen, promediando cinco cortes | un solo corte tiene desvío ~1,0, mayor que las diferencias entre configuraciones; `GroupKFold` estimó 48,0 y el servidor dio 43,4 |
| Features físicas adimensionales | +5,1 bajo cambio de régimen sobre las crudas con z-score |
| z-score por máquina, junto a las físicas | ventaja consistente; no borra la huella de máquina pero mejora el ajuste |
| Ensamble LightGBM + ExtraTrees + RandomForest | menor varianza que cualquier modelo individual |
| Cobertura One-vs-Rest con peso 0,3 | mejora la media y reduce la varianza; cubre el riesgo de fallas combinadas en el test |
| **Se descarta la calibración explícita** | temperatura 44,41 e isotónica 44,08 contra 45,39 sin calibrar; el `prob_score` baja |
| Se descarta limpiar features por correlación | 45,44 → 44,45: con `max_features="sqrt"` las copias correlacionadas mejoran el muestreo del bagging |
| Se descarta el modelo jerárquico familia → falla | 44,76 → 42,10 |
| Se descarta la regresión logística en el ensamble | transfiere 38,17 bajo cambio de régimen |
| Se descarta el afilado de probabilidades | la capa de decisión de la métrica ya es óptima: coincide en el 99,1 % con la acción óptima bajo nuestro posterior |
| Se descarta el tuneo de hiperparámetros | mueve menos que el ruido de partición |

El tamaño efectivo de la muestra es de 250 sesiones. Con esa cifra las mejoras
aceptadas son estructurales —features con sentido físico, esquema de validación,
cobertura ante un riesgo declarado en la consigna— y no numéricas.
"""))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
pathlib.Path("solucion_final.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print("escrito solucion_final.ipynb", len(C), "celdas")
