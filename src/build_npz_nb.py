import json, pathlib
def _l(s):
    ls=s.split("\n"); return [x+"\n" for x in ls[:-1]]+[ls[-1]]
def md(s): return {"cell_type":"markdown","metadata":{},"source":_l(s.strip())}
def co(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_l(s.strip("\n"))}
C=[]

C.append(md("""
# Fase NPZ — extracción de features de señal cruda

Notebook **a ciegas**: fue escrito sin haber visto nunca un NPZ real, porque el
entorno donde se desarrolló no tiene acceso a Drive. Por eso está construido
para **inspeccionar antes de extraer** y para **abortar temprano** si las
features no discriminan, en lugar de procesar las 4032 ventanas a ciegas.

Se ejecuta en orden. Cada paso imprime un diagnóstico; **si un paso falla o da
un resultado pobre, el notebook lo dice y no avanza**. Pasar la salida completa
de vuelta permite corregir sin adivinar.

## Qué se busca y por qué

Medido sobre el OOF, reemplazando partes de la predicción por la verdad:

| Escenario | final |
|---|---|
| actual | 52,07 |
| + intra-familia perfecta (rodamientos F03/F04/F05) | 56,47 |
| + familia perfecta | 64,84 |

La tabla ya trae `rms`, `kurtosis`, `crest`, `1x` y `2x` por canal.
**Recalcular eso desde la señal no aporta nada.** Lo que sólo está en la señal:

- **espectro de envolvente**: donde viven las frecuencias de defecto de
  rodamiento. Es lo único que separa F03/F04/F05 entre sí.
- **bandas laterales de la corriente (MCSA)**: barras rotóricas y espiras.
- **espectro de presión**: firma de cavitación.
"""))

C.append(md("## 0. Entorno y datos"))
C.append(co('''
import os, sys, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.signal import welch, hilbert, butter, sosfiltfilt

DATA = "datos_sinraw/"        # train_sup.parquet, test.parquet, raw_signal_index.csv
RAW  = "raw_signals/"         # los NPZ  <-- AJUSTAR si quedaron en otra carpeta
KIT  = "participant_kit/"
sys.path.insert(0, KIT)
from scoring import FAULT_IDS, LABEL_COLUMNS, FAMILIES, compute_score, validate_prediction_package

RS = 42
np.random.seed(RS)
train = pd.read_parquet(DATA + "train_sup.parquet")
test  = pd.read_parquet(DATA + "test.parquet")
idx   = pd.read_csv(DATA + "raw_signal_index.csv")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)

print("train", train.shape, "| test", test.shape)
print("raw_signal_index:", idx.shape, "| columnas:", list(idx.columns))
print(idx.head(3).to_string())
npz_files = sorted(glob.glob(os.path.join(RAW, "*.npz")))
print("\\narchivos NPZ encontrados:", len(npz_files))
if not npz_files:
    print("  >>> NO SE ENCONTRARON NPZ. Ajustar la variable RAW y volver a correr.")
'''))

C.append(md("""
## 1. Inspección de una señal real

Antes de extraer nada hay que mirar qué hay adentro: longitud, rango, NaN, y
sobre todo **la frecuencia de muestreo efectiva**. El `eda_intro` del kit declara
`acc_radial_a` a 20 kHz, `current_a` a 10 kHz y `pressure_in` a 200 Hz, pero eso
se verifica contra la duración real de la ventana en lugar de asumirlo: si la
ventana dura `d` segundos y tiene `n` muestras, entonces `fs = n/d`.
"""))
C.append(co('''
FS_DECL = {"acc_radial_a": 20000, "acc_radial_b": 20000, "acc_axial": 20000,
           "current_a": 10000, "current_b": 10000, "current_c": 10000,
           "voltage_a": 10000, "voltage_b": 10000, "voltage_c": 10000,
           "pressure_in": 200, "pressure_out": 200, "flow": 100, "rpm": 50}

def resolver(f):
    """El index trae 'raw_signals/S<hash>.npz'. Se prueban las formas posibles."""
    for c in (str(f), os.path.join(RAW, os.path.basename(str(f))),
              os.path.join(os.path.dirname(RAW.rstrip("/")), str(f))):
        if os.path.exists(c): return c
    return None

fila, ruta = None, None
for _, r_ in idx.iterrows():                # primera fila cuyo archivo exista de verdad
    c = resolver(r_["file"])
    if c: fila, ruta = r_, c; break
if ruta is None:
    raise FileNotFoundError("ningun archivo del index se encontro; ajustar RAW")
print("abriendo:", ruta)

with np.load(ruta, allow_pickle=False) as z:
    claves = list(z.files)
    print("claves en el archivo:", len(claves))
    print("primeras 6:", claves[:6])
    wid = str(fila["window_id"])
    mias = [k for k in claves if k.startswith(wid + "__")]
    print("\\nclaves de la ventana %s: %s" % (wid, mias))
    for k in mias:
        x = z[k]
        canal = k.split("__", 1)[1]
        fs = FS_DECL.get(canal, np.nan)
        print("  %-16s n=%7d  dtype=%s  min=%9.4f  max=%9.4f  NaN=%d  dur~%.3f s (fs decl %s)"
              % (canal, len(x), x.dtype, np.nanmin(x), np.nanmax(x),
                 int(np.isnan(x).sum()), len(x) / fs if fs == fs else np.nan, fs))

# duracion declarada de la ventana, si el index o la tabla la traen
if "timestamp_start_s" in train.columns:
    ts = train.sort_values(["session_id", "timestamp_start_s"]).groupby("session_id").timestamp_start_s
    paso = ts.apply(lambda s: s.diff().dropna().median()).median()
    print("\\npaso entre ventanas consecutivas de una sesión: %.3f s" % paso)
    print("  -> si n/fs coincide con este paso, la fs declarada es correcta")
'''))

C.append(md("""
## 2. Extractor de features

Dos decisiones de diseño, ambas por prudencia:

**No se asume la geometría del rodamiento.** Los ratios BPFO/BPFI/BSF dependen
del número de elementos rodantes y del diámetro, que no conocemos. En lugar de
apostar a valores típicos y arriesgar que los picos caigan en el lugar
equivocado, se **barre el espectro de envolvente en órdenes de la frecuencia de
giro** (de 2 a 9, paso 0,25) y se le entrega al modelo el perfil completo. Que el
árbol encuentre dónde está el pico en lugar de adivinarlo nosotros.

**La banda resonante se busca, no se fija.** Se parte el espectro en bandas y se
elige la de mayor kurtosis, que es donde el defecto modula la resonancia
estructural. Es un kurtograma pobre pero suficiente.
"""))
C.append(co('''
def psd(x, fs, nper=4096):
    f, p = welch(x, fs=fs, nperseg=min(nper, len(x)))
    return f, p

def banda_resonante(x, fs, n=6):
    """Devuelve la banda de mayor kurtosis: ahí modula el defecto."""
    nyq = fs / 2.0
    mejor, kmej = (0.25 * nyq, 0.75 * nyq), -np.inf
    for i in range(n):
        lo, hi = max(i * nyq / n, 1.0), min((i + 1) * nyq / n, nyq * 0.999)
        if hi - lo < 50: continue
        try:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
            y = sosfiltfilt(sos, x)
        except Exception:
            continue
        k = float(((y - y.mean()) ** 4).mean() / (y.var() ** 2 + 1e-12))
        if np.isfinite(k) and k > kmej: kmej, mejor = k, (lo, hi)
    return mejor, kmej

def envolvente_psd(x, fs, banda):
    nyq = fs / 2.0
    try:
        sos = butter(4, [banda[0] / nyq, min(banda[1], nyq * 0.999) / nyq],
                     btype="band", output="sos")
        x = sosfiltfilt(sos, x)
    except Exception:
        pass
    env = np.abs(hilbert(x))
    return psd(env - env.mean(), fs, nper=8192)

ORDENES = np.arange(2.0, 9.01, 0.25)      # perfil en ordenes de fr, sin asumir geometria

def energia_en(f, p, obj, tol=0.04):
    if not np.isfinite(obj) or obj <= 0 or obj >= f[-1]: return 0.0
    m = (f > obj * (1 - tol)) & (f < obj * (1 + tol))
    if not m.any(): return 0.0
    return float(p[m].max() / (np.median(p) + 1e-20))

def feats_acc(x, fs, rpm):
    fr = max(rpm, 1.0) / 60.0
    (lo, hi), kres = banda_resonante(x, fs)
    f, p = envolvente_psd(x, fs, (lo, hi))
    d = {"env_lo": lo, "env_hi": hi, "env_kres": kres}
    for o in ORDENES:                       # perfil completo, sin asumir BPFO/BPFI/BSF
        d["env_ord_%.2f" % o] = energia_en(f, p, fr * o)
    tot = p.sum() + 1e-20
    d["env_frac_baja"] = float(p[f < 10 * fr].sum() / tot)
    d["env_entropia"]  = float(-(p / tot * np.log(p / tot + 1e-20)).sum())
    d["env_pico_ord"]  = float(f[p.argmax()] / fr) if fr > 0 else 0.0
    return d

def feats_current(x, fs, rpm, f_red=50.0):
    f, p = psd(x, fs)
    d = {}
    for s in (0.005, 0.01, 0.02, 0.03, 0.05):
        d["mcsa_s%03d" % int(s * 1000)] = float(
            (energia_en(f, p, f_red * (1 - 2 * s)) + energia_en(f, p, f_red * (1 + 2 * s))) / 2)
    p0 = p[int(np.argmin(np.abs(f - f_red)))] + 1e-20
    d["mcsa_thd"] = float(sum(p[int(np.argmin(np.abs(f - f_red * h)))] for h in (3, 5, 7)) / p0)
    tot = p.sum() + 1e-20
    d["cur_entropia"] = float(-(p / tot * np.log(p / tot + 1e-20)).sum())
    d["cur_frac_alta"] = float(p[f > 2 * f_red].sum() / tot)
    return d

def feats_pressure(x, fs):
    f, p = psd(x, fs)
    tot = p.sum() + 1e-20
    return {"pre_frac_alta": float(p[f > f[-1] * 0.5].sum() / tot),
            "pre_entropia": float(-(p / tot * np.log(p / tot + 1e-20)).sum()),
            "pre_kurt": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12))}

print("extractor definido | perfil de envolvente:", len(ORDENES), "órdenes")
'''))

C.append(md("""
## 3. Prueba sobre una muestra, ANTES de procesar todo

Se extrae sólo sobre las sesiones de rodamiento (F03, F04, F05) más un grupo de
sanas. Si el perfil de envolvente no separa esos tres modos entre sí, no tiene
sentido procesar las 4032 ventanas: hay que corregir el extractor primero.

**Criterio de decisión explícito:** si la mejor AUC por par supera 0,75, la
extracción sirve y se sigue. Entre 0,65 y 0,75 es marginal. Por debajo de 0,65,
parar y revisar (la señal puede estar decimada, la ventana ser muy corta, o la
resonancia caer fuera de la banda buscada).
"""))
C.append(co('''
FS_ACC, FS_CUR, FS_PRE = 20000, 10000, 200

m = idx.merge(train[["window_id", "session_id", "machine_id", "rpm_mean"]], on="window_id", how="inner")
ses_estado = pd.Series(estado, index=train.window_id).groupby(train.session_id.values).first()
print("ventanas de train con NPZ mapeado:", len(m), "de", len(train))

# sesiones objetivo: todas las de rodamiento + 8 sanas
obj = [s for s in ses_estado.index if ses_estado[s] in (3, 4, 5)]
sanas = [s for s in ses_estado.index if ses_estado[s] == 0][:8]
sel_ses = set(obj) | set(sanas)
sub = m[m.session_id.isin(sel_ses)]
print("sesiones de prueba:", len(sel_ses), "| ventanas:", len(sub))

def extraer_ventana(z, wid, rpm, canales):
    """Si el index no declara canales, se derivan de las claves del propio NPZ."""
    if not canales:
        canales = [k.split("__", 1)[1] for k in z.files if k.startswith(wid + "__")]
    d = {}
    for ch in canales:
        k = f"{wid}__{ch}"
        if k not in z.files: continue
        x = np.asarray(z[k], float)
        malos = ~np.isfinite(x)
        # el umbral debe ser bajo: pressure_in va a 200 Hz, o sea 200 muestras
        # por ventana de 1 s. Con 256 se perdia el canal entero en silencio.
        if malos.all() or len(x) < 64: continue
        if malos.any():
            # interpolar en lugar de comprimir: quitar muestras corre la base
            # temporal y mete armonicos falsos en el espectro
            if malos.mean() > 0.30: continue
            ii = np.arange(len(x))
            x[malos] = np.interp(ii[malos], ii[~malos], x[~malos])
        if ch.startswith("acc"):
            d.update({f"{ch}__{a}": b for a, b in feats_acc(x, FS_ACC, rpm).items()})
        elif ch.startswith("current"):
            d.update({f"{ch}__{a}": b for a, b in feats_current(x, FS_CUR, rpm).items()})
        elif ch.startswith("pressure"):
            d.update({f"{ch}__{a}": b for a, b in feats_pressure(x, FS_PRE).items()})
    return d

def extraer_conjunto(tabla, etiqueta=""):
    filas, err = [], 0
    for arch, g in tabla.groupby("file"):
        ruta = os.path.join(RAW, os.path.basename(str(arch)))
        if not os.path.exists(ruta): ruta = str(arch)
        if not os.path.exists(ruta): err += 1; continue
        try:
            with np.load(ruta, allow_pickle=False) as z:
                for _, r in g.iterrows():
                    ch = str(r["channels"]).split("|") if "channels" in r and pd.notna(r["channels"]) else []
                    d = extraer_ventana(z, str(r["window_id"]), float(r["rpm_mean"]), ch)
                    if d: d["window_id"] = str(r["window_id"]); filas.append(d)
        except Exception as e:
            err += 1
    print("%s extraidas %d ventanas | archivos con error: %d" % (etiqueta, len(filas), err))
    return pd.DataFrame(filas)

import time
t0 = time.time()
F = extraer_conjunto(sub, "[prueba]")
print("tiempo: %.1f s  ->  proyeccion para 4032 ventanas: %.1f min"
      % (time.time() - t0, (time.time() - t0) / max(len(F), 1) * 4032 / 60))
print("columnas nuevas:", F.shape[1] - 1)
porcanal = {}
for c in F.columns:
    if c == "window_id": continue
    porcanal[c.split("__")[0]] = porcanal.get(c.split("__")[0], 0) + 1
print("features por canal:", porcanal)
esperados = set()
for _, r_ in sub.head(50).iterrows():
    if "channels" in r_ and pd.notna(r_["channels"]):
        esperados |= set(str(r_["channels"]).split("|"))
faltan_ch = [c for c in esperados if c not in porcanal]
if faltan_ch:
    print("  >>> ATENCION: estos canales no produjeron NINGUNA feature:", faltan_ch)
    print("      revisar longitud minima, frecuencia de muestreo y nombre del canal")
F.head(3)
'''))
C.append(co('''
from sklearn.metrics import roc_auc_score
if len(F) == 0:
    print(">>> No se extrajo nada. Revisar RAW, el nombre de los canales y las claves del NPZ.")
else:
    Fm = F.merge(train[["window_id"]].assign(e=estado), on="window_id")
    cols = [c for c in F.columns if c != "window_id"]
    print("AUC de la mejor feature por par de rodamientos (criterio: >0.75 sirve)\\n")
    resumen = []
    for a, b in [("F03", "F04"), ("F03", "F05"), ("F04", "F05")]:
        ia, ib = FA.index(a) + 1, FA.index(b) + 1
        sel = Fm[Fm.e.isin([ia, ib])]
        if len(sel) < 20: print("%s vs %s: muestras insuficientes" % (a, b)); continue
        y = (sel.e == ia).astype(int).to_numpy()
        best, bn = 0.5, None
        for c in cols:
            v = pd.to_numeric(sel[c], errors="coerce").to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < 20 or len(np.unique(v[ok])) < 2: continue
            try: au = roc_auc_score(y[ok], v[ok])
            except ValueError: continue
            au = max(au, 1 - au)
            if au > best: best, bn = au, c
        resumen.append(best)
        print("  %s vs %s (n=%3d):  AUC=%.3f   con %s" % (a, b, len(sel), best, bn))
    if resumen:
        peor = min(resumen)
        print("\\nAUC minima entre los tres pares: %.3f" % peor)
        print("(ojo: es el MAXIMO sobre ~%d columnas, asi que esta sesgada al alza;" % len(cols))
        print(" por eso el umbral se pone en 0.75 y no en 0.70)")
        print("VEREDICTO:", "SIRVE, continuar" if peor > 0.75 else
              ("MARGINAL, ver nota" if peor > 0.65 else "NO SIRVE, revisar el extractor antes de seguir"))
        print("(referencia: con las features tabulares solas, F03 vs F04 da 0.695 y F03 vs F05 da 0.714)")
'''))
C.append(md("""
**Si el veredicto es NO SIRVE**, no seguir. Las causas probables, en orden:

1. La ventana es demasiado corta para resolver la envolvente. Mirar `n` en el
   paso 1: con 20 kHz y una ventana de 1 s hay resolución de sobra, pero si la
   señal viene decimada puede no alcanzar.
2. La resonancia cae fuera de las bandas buscadas: subir `n` en
   `banda_resonante` de 6 a 12.
3. Las claves del NPZ no siguen el formato esperado: revisar la salida del
   paso 1 y ajustar `extraer_ventana`.

**Si el veredicto es MARGINAL**, conviene seguir igual: el perfil completo de
órdenes puede aportar en combinación aunque ninguna columna sola destaque, y eso
se ve recién en el paso 5.
""")); 

C.append(md("""
## 4. Extracción completa

Sólo si el paso 3 dio verde. Se guarda en Parquet: es caro y no se quiere
repetir.
"""))
C.append(co('''
EXTRAER_TODO = True     # poner False para saltear si el paso 3 fallo

if EXTRAER_TODO:
    todas = idx.copy()
    rpm_map = pd.concat([
        train[["window_id", "rpm_mean"]], test[["window_id", "rpm_mean"]]
    ]).drop_duplicates("window_id")
    todas = todas.merge(rpm_map, on="window_id", how="left")
    todas["rpm_mean"] = todas["rpm_mean"].fillna(todas["rpm_mean"].median())
    import time; t0 = time.time()
    FULL = extraer_conjunto(todas, "[completo]")
    print("tiempo total: %.1f min" % ((time.time() - t0) / 60))
    FULL.to_parquet("raw_features.parquet", index=False)
    print("guardado raw_features.parquet", FULL.shape)
else:
    FULL = pd.read_parquet("raw_features.parquet")
    print("cargado de disco:", FULL.shape)
'''))

C.append(md("""
## 5. ¿Aportan? Medición con el protocolo de cinco cortes

Las features nuevas se **concatenan** a las 174 de base4, sin quitar nada
(limpiar por correlación ya se midió que rompe el bagging). El juez es el
promedio de cinco cortes por régimen, y el umbral de aceptación es ~2 puntos.
"""))
C.append(co('''
BASE = [c for c in test.columns
        if c not in ("window_id", "machine_id", "session_id", "timestamp_start_s")]
ACC = ["acc_radial_a", "acc_radial_b", "acc_axial"]

def fisicas(df):
    X = pd.DataFrame(index=df.index)
    rpm = df.rpm_mean.clip(lower=1) / 1000.0; rpm2 = rpm ** 2
    for a in ACC:
        rms = df[f"{a}__rms"].clip(lower=1e-9)
        X[f"{a}__rms_n"]=rms/rpm2; X[f"{a}__1x_r"]=df[f"{a}__1x"]/(rms**2+1e-9)
        X[f"{a}__2x1x"]=df[f"{a}__2x"]/(df[f"{a}__1x"].abs()+1e-6)
        X[f"{a}__lkurt"]=np.log1p(df[f"{a}__kurtosis"].clip(lower=0))
        X[f"{a}__crest_n"]=df[f"{a}__crest"]
    ra = df["acc_radial_a__rms"].clip(lower=1e-9)
    X["ax_rad"]=df["acc_axial__rms"]/ra; X["rad_b_a"]=df["acc_radial_b__rms"]/ra
    X["vib_tot"]=df[[f"{a}__rms" for a in ACC]].sum(1)/rpm2
    V=df[["voltage_a__rms","voltage_b__rms","voltage_c__rms"]]
    I=df[["current_a__rms","current_b__rms","current_c__rms"]]
    vm,im=V.mean(1).clip(lower=1e-9),I.mean(1).clip(lower=1e-9)
    X["V_desbal"]=(V.max(1)-V.min(1))/vm; X["I_desbal"]=(I.max(1)-I.min(1))/im
    X["V_cv"]=V.std(1)/vm; X["I_cv"]=I.std(1)/im
    X["I_min_r"]=I.min(1)/im; X["I_max_r"]=I.max(1)/im
    X["S_ap"]=im*vm/1000.0; X["I_por_rpm"]=im/rpm
    X["I_por_flow"]=im/(df.flow_mean.abs()+1e-6)
    X["dT_wind"]=df["temp_winding__mean"]-df.Tamb
    X["dT_b1"]=df["temp_bearing1__mean"]-df.Tamb
    X["dT_b2"]=df["temp_bearing2__mean"]-df.Tamb
    X["dT_b1b2"]=df["temp_bearing1__mean"]-df["temp_bearing2__mean"]
    X["dT_wind_b"]=df["temp_winding__mean"]-df[["temp_bearing1__mean","temp_bearing2__mean"]].mean(1)
    X["dT_wind_rpm"]=X["dT_wind"]/rpm
    X["head_n"]=df.delta_p_mean/rpm2; X["flow_n"]=df.flow_mean/rpm
    X["p_in_n"]=df.pressure_in_mean/rpm2
    X["p_ratio"]=df.pressure_out_mean/(df.pressure_in_mean.abs()+1e-6)
    X["hidr_pot"]=df.flow_mean*df.delta_p_mean/(im*vm+1e-6)
    X["flow_head"]=df.flow_mean/(df.delta_p_mean.abs()+1e-6)
    X["rpm_cv"]=df.rpm_std/df.rpm_mean.clip(lower=1)
    nanc=[c for c in df.columns if c.endswith("__nan_frac")]
    X["nan_tot"]=df[nanc].sum(1); X["nan_max"]=df[nanc].max(1)
    return X.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def zmaq(df, cols):
    g=df.groupby("machine_id")[cols]
    z=(df[cols]-g.transform("median"))/(g.transform("std")+1e-9)
    z.columns=[c+"__z" for c in cols]; return z

def construir(df, raw=None):
    fis=pd.concat([df[BASE], fisicas(df)],axis=1)
    tmp=fis.copy(); tmp["machine_id"]=df["machine_id"].to_numpy()
    X=pd.concat([fis, zmaq(tmp,list(fis.columns))],axis=1)
    if raw is not None:
        assert not raw["window_id"].duplicated().any(), "raw tiene window_id repetido: el merge desalinearia las filas"
        r=df[["window_id"]].merge(raw,on="window_id",how="left")
        r=r.drop(columns=["window_id"]).replace([np.inf,-np.inf],np.nan).fillna(0.0)
        r.index=X.index
        X=pd.concat([X,r],axis=1)
    return X

Xtr_b, Xte_b = construir(train), construir(test)
Xtr_r, Xte_r = construir(train, FULL), construir(test, FULL)
print("base:", Xtr_b.shape[1], "columnas | con NPZ:", Xtr_r.shape[1], "columnas")
faltan = (Xtr_r.iloc[:, Xtr_b.shape[1]:] == 0).all(0).sum()
print("columnas de señal que quedaron todas en cero (no se pudieron mapear):", int(faltan))
'''))
C.append(co('''
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

FAMS=["sano"]+sorted(set(FAMILIES.values()))
FAM_DE=np.array(["sano"]+[FAMILIES[f] for f in FA])
fam_id=np.array([FAMS.index(g) for g in FAM_DE]); y_fam=fam_id[estado]

def lgbm(nc,seed): return lgb.LGBMClassifier(objective="multiclass",num_class=nc,
    n_estimators=600,learning_rate=0.05,num_leaves=31,min_child_samples=20,
    subsample=0.9,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,verbose=-1,
    random_state=seed,deterministic=True,force_col_wise=True,n_jobs=1)
def et(s): return ExtraTreesClassifier(n_estimators=600,min_samples_leaf=2,max_features="sqrt",n_jobs=1,random_state=s)
def rf(s): return RandomForestClassifier(n_estimators=600,min_samples_leaf=2,max_features="sqrt",n_jobs=1,random_state=s)

def proba_k(models,Xa,ya,Xb,k):
    P=np.zeros((len(Xb),k))
    for m in models:
        m.fit(Xa,ya); pp=m.predict_proba(Xb); cls=np.asarray(m.classes_,int)
        Q=np.zeros((len(Xb),k)); Q[:,cls]=pp; P+=Q
    return P/len(models)

def ovr13(X,Xa,idx_a,Xb):
    B=np.zeros((len(Xb),13))
    for j,f in enumerate(FA):
        y=train[f"label_{f}"].to_numpy()[idx_a]
        if y.sum()<2: continue
        m=lgb.LGBMClassifier(objective="binary",n_estimators=300,learning_rate=0.05,
            num_leaves=15,min_child_samples=25,subsample=0.9,subsample_freq=1,
            colsample_bytree=0.7,reg_lambda=5.0,verbose=-1,random_state=RS,
            deterministic=True,force_col_wise=True,n_jobs=1)
        m.fit(Xa,y); B[:,j]=m.predict_proba(Xb)[:,1]
    return B

def reponderar(P14,Pfam,alpha):
    if alpha==0: return P14
    M=np.zeros((len(P14),5))
    for g in range(5): M[:,g]=P14[:,fam_id==g].sum(1)
    r=(np.clip(Pfam,1e-9,1)/np.clip(M,1e-9,1))**alpha
    Q=P14*r[:,fam_id]; return Q/Q.sum(1,keepdims=True)

def evaluar(P13,idx):
    e=pd.DataFrame({"window_id":train.window_id.iloc[idx].to_numpy()})
    for j,f in enumerate(FA): e[f]=np.clip(P13[:,j],0,1)
    return compute_score(train.iloc[idx],e)["overall"]["final_score"]

rpm_maq=train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES=[11,12,13,14,15]; W=0.3; ALPHA=0.5
res={}
for k in CORTES:
    rap=set(rpm_maq.index[k:]); er=train.machine_id.isin(rap).to_numpy()
    a,b=np.where(~er)[0],np.where(er)[0]
    linea=[]
    for nom,Xf in (("sin NPZ",Xtr_b),("con NPZ",Xtr_r)):
        Xa,Xb=Xf.iloc[a],Xf.iloc[b]
        P=proba_k([lgbm(14,RS),et(RS),rf(RS)],Xa,estado[a],Xb,14)
        B=ovr13(Xf,Xa,a,Xb)
        Pf=proba_k([lgbm(5,RS),et(RS),rf(RS)],Xa,y_fam[a],Xb,5)
        Q=reponderar(P,Pf,ALPHA)
        s=evaluar((1-W)*Q[:,1:]+W*B,b)
        res.setdefault(nom,[]).append(s); linea.append("%s=%.2f"%(nom,s))
        res.setdefault(nom+" accFam",[]).append(float((y_fam[b]==fam_id[Q.argmax(1)]).mean()))
    print("k=%2d  %s  | delta=%+.2f"%(k," ".join(linea),res["con NPZ"][-1]-res["sin NPZ"][-1]),flush=True)

print("\\n=== promedio de los cinco cortes ===")
for n in ["sin NPZ","con NPZ"]:
    print("%-10s %.2f +- %.2f | accFam %.3f"%(n,np.mean(res[n]),np.std(res[n]),np.mean(res[n+" accFam"])))
d=np.mean(res["con NPZ"])-np.mean(res["sin NPZ"])
print("\\nganancia de las features de señal: %+.2f puntos"%d)
print("VEREDICTO:", "ADOPTAR" if d>2 else ("MARGINAL, decidir con criterio" if d>0.5 else "NO ADOPTAR"))
'''))

C.append(md("""
## 6. Entrega, sólo si el paso 5 dio verde
"""))
C.append(co('''
GENERAR = True
if GENERAR:
    if "res" not in dir() or "con NPZ" not in res:
        raise RuntimeError("correr el paso 5 antes: la eleccion de features depende de su medicion")
    Xf_tr, Xf_te = (Xtr_r, Xte_r) if np.mean(res["con NPZ"]) > np.mean(res["sin NPZ"]) else (Xtr_b, Xte_b)
    print("usando:", Xf_tr.shape[1], "columnas")
    P = proba_k([lgbm(14,RS),et(RS),rf(RS)], Xf_tr, estado, Xf_te, 14)
    B = ovr13(Xf_tr, Xf_tr, np.arange(len(train)), Xf_te)
    Pf = proba_k([lgbm(5,RS),et(RS),rf(RS)], Xf_tr, y_fam, Xf_te, 5)
    Q = reponderar(P, Pf, ALPHA)
    Pfin = (1-W)*Q[:,1:] + W*B
    sub = pd.DataFrame({"window_id": test.window_id.to_numpy()})
    for j,f in enumerate(FA): sub[f]=np.clip(Pfin[:,j],0,1)
    sub = validate_prediction_package(test, sub)
    sub.to_csv("submit_npz.csv", index=False)
    print("submit_npz.csv", sub.shape, "validado | suma media %.3f"%sub[FA].sum(1).mean())
'''))

C.append(md("""
## 7. Qué pasar de vuelta

Para poder diagnosticar sin ver los datos, hace falta la salida de:

- **paso 1**: longitudes, dtypes, rangos y NaN de cada canal
- **paso 3**: el veredicto y las tres AUC por par de rodamientos
- **paso 5**: la tabla de los cinco cortes con la ganancia

Con eso alcanza para decidir si el extractor necesita corrección o si las
features de señal entran en la entrega final.
"""))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
pathlib.Path("npz_fase.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print("escrito npz_fase.ipynb", len(C), "celdas")
