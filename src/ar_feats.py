"""Identificacion output-only por AR-Burg: las resonancias de la estructura.

Conexion con la Experiencia 4 del Lab 4 (identificacion y reduccion de orden).
Alli se ajustaba H(s) porque se tenia el par entrada-salida (escalon conocido ->
respuesta medida). En los NPZ solo hay SALIDAS: no existe escalon que reproducir,
asi que el AG con fitness RMSE no tiene contra que simular.

La version que si aplica es identificacion output-only: asumiendo excitacion
aproximadamente blanca en banda (razonable en maquinaria rotativa), un modelo AR
ajustado a la salida tiene polos que SON las resonancias estructurales y sus
amortiguamientos. Es el mismo objeto que se buscaba en el Lab 4 —polos de un
modelo de orden reducido— obtenido sin la entrada.

Por que importa aca: la amplitud de vibracion escala con rpm^2, pero la
FRECUENCIA DE RESONANCIA no depende de la rpm: depende de la rigidez y la masa.
Es una propiedad estructural. Por eso transfiere entre regimenes, que es
exactamente nuestro problema (train 1742 rpm, test 2052 rpm). Y una perdida de
rigidez (F07) o un rodamiento dañado la corren.

Se resuelve con Burg (solucion cerrada, milisegundos, determinista) y no con un
AG: sobre 4032 ventanas x 3 canales el AG costaria ~30 h y llegaria al mismo
optimo. En el propio Lab 4 el AG daba RMSE 0.046 y el pulido local lo bajaba a
0.0057: el trabajo fino lo hacia el metodo local, no la evolucion.
"""
import numpy as np
from scipy.signal import decimate, lfilter

def burg(x, p):
    """Coeficientes AR por el metodo de Burg. Devuelve a con a[0]=1."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    f = x.copy(); b = x.copy()
    a = np.zeros(p + 1); a[0] = 1.0
    den = 2.0 * np.dot(x, x)
    for m in range(1, p + 1):
        num = 2.0 * np.dot(f[m:], b[m - 1:n - 1])
        den = den - f[m - 1] ** 2 - b[n - 1] ** 2 if m > 1 else den
        if abs(den) < 1e-20: break
        k = num / den
        anew = a.copy()
        for i in range(1, m + 1):
            anew[i] = a[i] - k * a[m - i]
        a = anew
        fn = f[m:] - k * b[m - 1:n - 1]
        bn = b[m - 1:n - 1] - k * f[m:]
        f = np.r_[np.zeros(m), fn]; b = np.r_[np.zeros(m - 1), bn, [0.0]][:n]
        b = np.r_[np.zeros(m), bn] if len(bn) == n - m else b
    return a

def polos_ar(x, fs, p=20, dec=None):
    """Resonancias y amortiguamientos del modelo AR ajustado a la salida."""
    if dec and dec > 1:
        x = decimate(x, dec, ftype="fir", zero_phase=True); fs = fs / dec
    a = burg(x, p)
    r = np.roots(a)
    r = r[np.abs(r) < 1.0]                      # estables
    r = r[np.imag(r) > 1e-9]                    # un polo por par conjugado
    if len(r) == 0: return np.array([]), np.array([]), np.array([])
    f = np.angle(r) * fs / (2 * np.pi)          # Hz
    z = -np.log(np.abs(r) + 1e-12)              # amortiguamiento (mayor = mas amortiguado)
    q = np.abs(r)                               # cercania al circulo unidad = resonancia aguda
    o = np.argsort(-q)                          # las mas resonantes primero
    return f[o], z[o], q[o]

def feats_ar(x, fs, p=20, dec=None, k=5):
    """k resonancias dominantes: frecuencia, amortiguamiento y agudeza."""
    f, z, q = polos_ar(x, fs, p, dec)
    d = {}
    for i in range(k):
        d["ar_f%d" % i] = float(f[i]) if i < len(f) else 0.0
        d["ar_z%d" % i] = float(z[i]) if i < len(z) else 0.0
        d["ar_q%d" % i] = float(q[i]) if i < len(q) else 0.0
    d["ar_n"] = float(len(f))
    d["ar_fmed"] = float(np.median(f)) if len(f) else 0.0
    d["ar_zmed"] = float(np.median(z)) if len(z) else 0.0
    return d
