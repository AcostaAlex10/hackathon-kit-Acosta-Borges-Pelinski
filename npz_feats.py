"""Extraccion de features de senal cruda para el desafio Machine Health.

REGLA DE DISENO: no recalcular lo que la tabla ya trae (rms, kurtosis, crest,
1x, 2x por canal). Solo se extrae lo que NO esta en la tabla y solo vive en la
senal:

  - espectro de ENVOLVENTE en la banda resonante -> frecuencias de defecto de
    rodamiento. Es lo unico que separa F03/F04/F05 entre si.
  - perfil de ordenes de la envolvente -> no depende de conocer la geometria
    del rodamiento, que no tenemos.
  - bandas laterales de la CORRIENTE (MCSA) -> barras rotoricas, espiras.
  - espectro de PRESION -> cavitacion (banda ancha de alta frecuencia).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, welch, butter, sosfiltfilt, decimate

FS = {"acc_radial_a": 20000, "current_a": 10000, "pressure_in": 200}

# ------------------------------------------------------------------ utilidades
def _psd(x, fs, nper):
    nper = int(min(nper, len(x)))
    f, p = welch(x, fs=fs, nperseg=nper, detrend="constant")
    return f, p

def _pico(f, p, obj, tol=0.04):
    """Energia del pico cerca de `obj`, relativa a la mediana del espectro."""
    if obj <= 0 or obj >= f[-1]:
        return 0.0
    m = (f > obj * (1 - tol)) & (f < obj * (1 + tol))
    if not m.any():
        return 0.0
    return float(p[m].max() / (np.median(p) + 1e-20))

def _entropia(p):
    p = p / (p.sum() + 1e-20)
    return float(-(p * np.log(p + 1e-20)).sum())

# ------------------------------------------------------- banda resonante
def banda_resonante(x, fs, n_bandas=6):
    """Kurtograma simplificado: la banda con mayor kurtosis es donde el
    defecto de rodamiento modula la resonancia estructural."""
    nyq = fs / 2.0
    mejor, k_mejor = (0.25 * nyq, 0.5 * nyq), -np.inf
    for i in range(n_bandas):
        lo = max(i * nyq / n_bandas, 20.0)
        hi = min((i + 1) * nyq / n_bandas, nyq * 0.995)
        if hi - lo < 100:
            continue
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        y = sosfiltfilt(sos, x)
        v = y.var()
        if v <= 0:
            continue
        k = float(((y - y.mean()) ** 4).mean() / (v ** 2))
        if k > k_mejor:
            k_mejor, mejor = k, (lo, hi)
    return mejor, k_mejor

def envolvente(x, fs, banda):
    nyq = fs / 2.0
    sos = butter(4, [banda[0] / nyq, min(banda[1], nyq * 0.995) / nyq],
                 btype="band", output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    return e - e.mean()

# --------------------------------------------------------------- vibracion
# Ordenes tipicos de defecto para un rodamiento de bolas. Son CANDIDATOS: la
# geometria real no la tenemos, por eso ademas se guarda el perfil completo.
ORDENES = {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357, "FTF": 0.398}
PERFIL = np.round(np.arange(1.0, 8.01, 0.25), 2)   # 29 ordenes

def feats_acc(x, fs, rpm):
    fr = max(rpm, 1.0) / 60.0
    d = {}
    (lo, hi), kres = banda_resonante(x, fs)
    d["env_band_lo"] = lo
    d["env_band_hi"] = hi
    d["env_kurt_res"] = kres

    env = envolvente(x, fs, (lo, hi))
    # la envolvente es de banda baja: se diezma para ganar resolucion util
    env_d = decimate(env, 10, ftype="fir", zero_phase=True)
    f, p = _psd(env_d, fs / 10.0, 4096)

    for nom, r in ORDENES.items():
        d["env_" + nom] = _pico(f, p, fr * r)
        d["env_" + nom + "_2"] = _pico(f, p, 2 * fr * r)
    # perfil de ordenes: no depende de la geometria
    for o in PERFIL:
        d["env_ord_%.2f" % o] = _pico(f, p, fr * o)

    tot = p.sum() + 1e-20
    d["env_frac_1_5fr"] = float(p[f < 5 * fr].sum() / tot)
    d["env_entropia"] = _entropia(p)
    d["env_pico_max"] = float(p.max() / (np.median(p) + 1e-20))
    d["env_ord_pico"] = float(f[int(np.argmax(p))] / (fr + 1e-9))

    # espectro directo: energia por bandas relativas al total
    f2, p2 = _psd(x, fs, 4096)
    t2 = p2.sum() + 1e-20
    for a, b, nom in [(0, 500, "b0"), (500, 2000, "b1"),
                      (2000, 5000, "b2"), (5000, 10000, "b3")]:
        m = (f2 >= a) & (f2 < b)
        d["acc_frac_" + nom] = float(p2[m].sum() / t2)
    d["acc_entropia"] = _entropia(p2)
    return d

# ---------------------------------------------------------------- corriente
def feats_cur(x, fs, rpm, f_red=50.0):
    f, p = _psd(x, fs, 8192)
    d = {}
    i0 = int(np.argmin(np.abs(f - f_red)))
    p0 = p[i0] + 1e-20
    # bandas laterales (1 +- 2s)f : firma de barras rotoricas
    for s in (0.01, 0.02, 0.03, 0.05):
        izq = _pico(f, p, f_red * (1 - 2 * s))
        der = _pico(f, p, f_red * (1 + 2 * s))
        d["mcsa_sb%d" % int(s * 100)] = float((izq + der) / 2)
        d["mcsa_asim%d" % int(s * 100)] = float(abs(izq - der) / (izq + der + 1e-9))
    for h in (3, 5, 7):
        d["mcsa_h%d" % h] = float(p[int(np.argmin(np.abs(f - f_red * h)))] / p0)
    fr = max(rpm, 1.0) / 60.0
    d["mcsa_fr"] = _pico(f, p, fr)          # modulacion mecanica en la corriente
    d["mcsa_2fr"] = _pico(f, p, 2 * fr)
    tot = p.sum() + 1e-20
    d["cur_frac_alta"] = float(p[f > 1000].sum() / tot)
    d["cur_entropia"] = _entropia(p)
    return d

# ----------------------------------------------------------------- presion
def feats_pre(x, fs):
    f, p = _psd(x, fs, 256)
    tot = p.sum() + 1e-20
    v = x.var()
    return {"pre_frac_alta": float(p[f > f[-1] * 0.5].sum() / tot),
            "pre_frac_media": float(p[(f > f[-1] * 0.2) & (f <= f[-1] * 0.5)].sum() / tot),
            "pre_entropia": _entropia(p),
            "pre_kurt": float(((x - x.mean()) ** 4).mean() / (v ** 2 + 1e-12)) if v > 0 else 0.0,
            "pre_cv": float(np.sqrt(v) / (abs(x.mean()) + 1e-9))}

# ------------------------------------------------------------------ ventana
def feats_ventana(sig: dict, rpm: float) -> dict:
    d = {}
    if "acc_radial_a" in sig:
        d.update(feats_acc(np.asarray(sig["acc_radial_a"], float), FS["acc_radial_a"], rpm))
    if "current_a" in sig:
        d.update(feats_cur(np.asarray(sig["current_a"], float), FS["current_a"], rpm))
    if "pressure_in" in sig:
        d.update(feats_pre(np.asarray(sig["pressure_in"], float), FS["pressure_in"]))
    return d

def n_features():
    import numpy as _np
    rng = _np.random.default_rng(0)
    s = {"acc_radial_a": rng.normal(size=40000),
         "current_a": rng.normal(size=20000),
         "pressure_in": rng.normal(size=400)}
    return len(feats_ventana(s, 1800.0))
