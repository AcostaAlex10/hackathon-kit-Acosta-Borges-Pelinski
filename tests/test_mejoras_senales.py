# -*- coding: utf-8 -*-
"""Prueba los extractores de la sección 10.6 con señales de contenido conocido."""
import sys, pathlib, numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from celdas_seccion10 import S

rng = np.random.default_rng(3)
EPS = 1e-12

def compute_psd(x, fs, nperseg=4096):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    if len(x) < 32:
        return np.array([]), np.array([])
    return welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend="constant")

def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0

# ------------------------------------------------------------ señales de prueba
DUR   = 2.0
F_LINE, POLOS = 50.0, 2
F_SYNC = F_LINE / POLOS            # 25 Hz de campo con p = 2
SLIP   = 0.03
F_ROT  = F_SYNC * (1 - SLIP)       # 24.25 Hz: el rotor va más lento que el campo
RPM    = F_ROT * 60.0              # 1455 rpm, coherente con el slip inyectado

def señal_corriente(fs=10_000, con_barras=True):
    t = np.arange(int(DUR * fs)) / fs
    x = 10 * np.sin(2*np.pi*F_LINE*t)
    for h, amp in [(3, .4), (5, .2), (7, .1)]:
        x += amp * np.sin(2*np.pi*h*F_LINE*t)
    if con_barras:                                     # bandas laterales de F10
        for sg in (-1, 1):
            x += .35 * np.sin(2*np.pi*F_LINE*(1 + sg*2*SLIP)*t)
    return x + .05 * rng.normal(size=len(t))

def señal_vibracion(fs=20_000, orden_rodamiento=4.3, con_subarmonico=True):
    t = np.arange(int(DUR * fs)) / fs
    x = 1.0*np.sin(2*np.pi*F_ROT*t) + .5*np.sin(2*np.pi*2*F_ROT*t)
    if con_subarmonico:                                # firma de F07
        x += .6 * np.sin(2*np.pi*0.5*F_ROT*t)
    # tren de impulsos de rodamiento excitando una resonancia en 3 kHz
    f_def = orden_rodamiento * F_ROT
    imp = np.zeros_like(t)
    imp[(np.arange(int(DUR*f_def)) * fs / f_def).astype(int)] = 1.0
    reson = np.exp(-np.arange(int(.004*fs))/(.0008*fs)) * np.sin(
        2*np.pi*3000*np.arange(int(.004*fs))/fs)
    x += 2.0 * np.convolve(imp, reson, mode="same")
    return x + .05 * rng.normal(size=len(t))

def señal_presion(fs=200):
    t = np.arange(int(DUR * fs)) / fs
    return 5 + .3*np.sin(2*np.pi*3*t) + .05*rng.normal(size=len(t))

SEÑALES = {"acc_radial_a": señal_vibracion(), "current_a": señal_corriente(),
           "pressure_in": señal_presion()}
def load_signal(row, canal):
    return SEÑALES[canal]

G = dict(np=np, pd=pd, EPS=EPS, compute_psd=compute_psd, band_power=band_power,
         load_signal=load_signal)
exec(S["fn_feat_cur"], G)
exec(S["fn_feat_vib"], G)

row = pd.Series({"rpm_mean": RPM, "window_id": "w0", "file": "x.npz"})
ok = lambda t: print(f"  OK  {t}")

print("== corriente v2 ==")
c = G["extract_current_features_v2"](row)
assert np.isfinite(list(c.values())).all(), "hay NaN o inf en las features de corriente"
ok(f"{len(c)} features, todas finitas")
assert abs(c["raw_cur2__f_line"] - F_LINE) < 1.0, c["raw_cur2__f_line"]
ok(f"fundamental estimada = {c['raw_cur2__f_line']:.2f} Hz (real {F_LINE})")
assert abs(c["raw_cur2__slip_p2"] - SLIP) < 0.02, c["raw_cur2__slip_p2"]
ok(f"slip con p=2 -> {c['raw_cur2__slip_p2']:.4f} (real {SLIP})")
sb = [c[f"raw_cur2__sb_p2k1{s}"] for s in ("m", "p")]
otras = [v for k, v in c.items() if k.startswith("raw_cur2__sb_p") and "p2k1" not in k]
assert min(sb) > max(otras) + 5, (sb, max(otras))
ok(f"las bandas laterales de barras se detectan sólo con el p correcto "
   f"({min(sb):.1f} dB vs {max(otras):.1f} dB)")
c0 = G["extract_current_features_v2"](row)
SEÑALES["current_a"] = señal_corriente(con_barras=False)
c1 = G["extract_current_features_v2"](row)
assert c0["raw_cur2__sb_p2k1m"] > c1["raw_cur2__sb_p2k1m"] + 10
ok(f"sin barras rotóricas la banda cae de {c0['raw_cur2__sb_p2k1m']:.1f} a "
   f"{c1['raw_cur2__sb_p2k1m']:.1f} dB")
SEÑALES["current_a"] = señal_corriente()
assert set(c0) == set(c1), "el conjunto de claves cambia entre filas"
ok("mismas claves siempre (columnas alineadas entre filas)")

print("== vibración v2 ==")
v = G["extract_vibration_features_v2"](row)
assert np.isfinite(list(v.values())).all()
ok(f"{len(v)} features, todas finitas")
SEÑALES["acc_radial_a"] = señal_vibracion(con_subarmonico=False)
v_sin = G["extract_vibration_features_v2"](row)
assert v["raw_vib2__05x_rel"] > 5 * v_sin["raw_vib2__05x_rel"]
ok(f"el subarmónico 0.5x (firma de F07) se detecta: "
   f"{v['raw_vib2__05x_rel']:.2e} vs {v_sin['raw_vib2__05x_rel']:.2e}")
assert v["raw_vib2__sub_sobre_1x"] > v_sin["raw_vib2__sub_sobre_1x"]
ok("sub_sobre_1x separa las dos condiciones")

print("== envolvente ==")
for orden in (3.2, 4.3, 6.5):
    SEÑALES["acc_radial_a"] = señal_vibracion(orden_rodamiento=orden)
    e = G["envolvente_features"](SEÑALES["acc_radial_a"], 20_000, F_ROT)
    assert np.isfinite(list(e.values())).all()
    det = e["raw_env__peak_orden"]
    assert abs(det - orden) < 0.35, (orden, det)
    ok(f"orden de rodamiento inyectado {orden} -> detectado {det:.2f}")
SEÑALES["acc_radial_a"] = señal_vibracion(orden_rodamiento=4.3)
e = G["envolvente_features"](SEÑALES["acc_radial_a"], 20_000, F_ROT)
assert e["raw_env__bpfo"] > e["raw_env__bpfi"] and e["raw_env__bpfo"] > e["raw_env__ftf"]
ok(f"la banda BPFO (3-7x) capta la energía: {e['raw_env__bpfo']:.3f} vs "
   f"BPFI {e['raw_env__bpfi']:.3f} y FTF {e['raw_env__ftf']:.3f}")

print("== bordes ==")
assert G["envolvente_features"](np.zeros(100), 20_000, F_ROT) == {}
assert G["envolvente_features"](SEÑALES["acc_radial_a"], 20_000, 0.0) == {}
assert G["envolvente_features"](SEÑALES["acc_radial_a"], 200, F_ROT) == {}
ok("señal corta, rpm=0 y fs por debajo de la banda: devuelven {} sin romper")
row_mal = pd.Series({"rpm_mean": 0.0})
cm = G["extract_current_features_v2"](row_mal)
assert np.isfinite(list(cm.values())).all() and set(cm) == set(c0)
ok("rpm=0 en corriente: finito y con las mismas claves")
vm = G["extract_vibration_features_v2"](row_mal)
assert np.isfinite(list(vm.values())).all()
ok("rpm=0 en vibración: finito")

print("== cross-canal ==")
x = G["extract_cross_features"](row)
assert np.isfinite(list(x.values())).all() and len(x) == 6
ok(f"{len(x)} features de coherencia, todas finitas, "
   f"coh_max={x['raw_x__coh_max']:.3f}")
com = 0.3*SEÑALES["current_a"][:20000] + señal_vibracion()[:20000:1][:20000]
SEÑALES["acc_radial_a"] = np.repeat(SEÑALES["current_a"], 2) + \
                          .1*rng.normal(size=2*len(SEÑALES["current_a"]))
x2 = G["extract_cross_features"](row)
assert x2["raw_x__coh_line"] > x["raw_x__coh_line"]
ok(f"con los canales acoplados la coherencia en f_line sube de "
   f"{x['raw_x__coh_line']:.3f} a {x2['raw_x__coh_line']:.3f}")

print("\nTODAS LAS PRUEBAS DE SEÑALES PASARON")
