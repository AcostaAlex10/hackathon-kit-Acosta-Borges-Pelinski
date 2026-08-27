"""Features de senal cruda dirigidas a lo que la tabla NO tiene.

La tabla ya trae rms, kurtosis, crest, 1x y 2x por canal. Recalcular eso desde
el NPZ no agrega nada. Lo que la tabla NO tiene, y solo esta en la senal:

  - espectro de ENVOLVENTE: donde viven las frecuencias de defecto de rodamiento
    (BPFO/BPFI/BSF). Es lo unico que separa F03/F04/F05 entre si.
  - kurtosis espectral / banda resonante: donde buscar la envolvente.
  - bandas laterales alrededor de 1x en la CORRIENTE (MCSA): barras rotoricas.
  - espectro de presion: firma de cavitacion (banda ancha de alta frecuencia).
"""
import numpy as np
from scipy.signal import hilbert, welch, butter, sosfiltfilt

def _psd(x, fs, nper=None):
    nper = nper or min(4096, len(x))
    f, p = welch(x, fs=fs, nperseg=nper)
    return f, p

def banda_resonante(x, fs, n=8):
    """Kurtograma pobre: parte la banda en n y devuelve la de mayor kurtosis.
    Es donde el defecto de rodamiento modula la resonancia estructural."""
    nyq = fs / 2.0
    mejor, k_mejor = (0.5 * nyq, nyq), -np.inf
    for i in range(n):
        lo, hi = i * nyq / n, (i + 1) * nyq / n
        lo = max(lo, 1.0); hi = min(hi, nyq * 0.999)
        if hi - lo < 50: continue
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        y = sosfiltfilt(sos, x)
        k = float(((y - y.mean()) ** 4).mean() / (y.var() ** 2 + 1e-12))
        if k > k_mejor: k_mejor, mejor = k, (lo, hi)
    return mejor, k_mejor

def envolvente_psd(x, fs, banda=None):
    """Espectro de la envolvente, opcionalmente tras filtrar la banda resonante."""
    if banda is not None:
        nyq = fs / 2.0
        sos = butter(4, [banda[0] / nyq, min(banda[1], nyq * 0.999) / nyq],
                     btype="band", output="sos")
        x = sosfiltfilt(sos, x)
    env = np.abs(hilbert(x))
    return _psd(env - env.mean(), fs)

def pico_en(f, p, obj, tol=0.03):
    """Energia relativa en un entorno de la frecuencia objetivo."""
    if obj <= 0 or obj >= f[-1]: return 0.0
    m = (f > obj * (1 - tol)) & (f < obj * (1 + tol))
    if not m.any(): return 0.0
    return float(p[m].max() / (np.median(p) + 1e-20))

# Ratios tipicos de rodamiento de bolas (dependen de la geometria; estos son
# valores habituales para un rodamiento de 8-9 elementos, en ordenes de la
# frecuencia de giro fr). Se usan como CANDIDATOS, no como verdad.
RATIOS = {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357, "FTF": 0.398}

def feats_acc(x, fs, rpm):
    """Features de vibracion que la tabla no tiene."""
    fr = rpm / 60.0
    (lo, hi), k_res = banda_resonante(x, fs)
    f, p = envolvente_psd(x, fs, (lo, hi))
    d = {"env_banda_lo": lo, "env_banda_hi": hi, "env_kurt_res": k_res}
    for nom, r in RATIOS.items():
        d[f"env_{nom}"] = pico_en(f, p, fr * r)
        d[f"env_{nom}_2x"] = pico_en(f, p, 2 * fr * r)
    tot = p.sum() + 1e-20
    d["env_frac_baja"] = float(p[f < 10 * fr].sum() / tot)
    d["env_entropia"] = float(-(p / tot * np.log(p / tot + 1e-20)).sum())
    return d

def feats_current(x, fs, rpm, f_red=50.0):
    """MCSA: bandas laterales alrededor de la fundamental -> barras rotoricas."""
    f, p = _psd(x, fs)
    d = {}
    i0 = int(np.argmin(np.abs(f - f_red)))
    p0 = p[i0] + 1e-20
    for s in (0.01, 0.02, 0.03, 0.05):       # deslizamientos candidatos
        d[f"mcsa_sb_{int(s*100)}"] = float(
            (pico_en(f, p, f_red * (1 - 2 * s)) + pico_en(f, p, f_red * (1 + 2 * s))) / 2)
    d["mcsa_thd"] = float(sum(p[int(np.argmin(np.abs(f - f_red * h)))] for h in (3, 5, 7)) / p0)
    tot = p.sum() + 1e-20
    d["cur_entropia"] = float(-(p / tot * np.log(p / tot + 1e-20)).sum())
    return d

def feats_pressure(x, fs):
    """Cavitacion: energia de banda ancha en alta frecuencia."""
    f, p = _psd(x, fs)
    tot = p.sum() + 1e-20
    return {"pre_frac_alta": float(p[f > f[-1] * 0.5].sum() / tot),
            "pre_kurt": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12))}
