# -*- coding: utf-8 -*-
"""Prueba las features hidráulicas de 10.5b contra una bomba sintética."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from celdas_seccion10 import S

rng = np.random.default_rng(11)
N_MAQ, POR_MAQ = 10, 60
maq = np.repeat([f"M{i}" for i in range(N_MAQ)], POR_MAQ); n = len(maq)
rpm = rng.uniform(1400, 2200, n); rpm_n = rpm / 1000.0
q_n = rng.uniform(0.5, 2.0, n)
coefs = {m: rng.uniform([8, -1, -2], [12, 1, -0.5]) for m in np.unique(maq)}
h_n = np.array([coefs[m][0] + coefs[m][1]*q + coefs[m][2]*q**2 for m, q in zip(maq, q_n)])

CAVITA = np.zeros(n, bool); CAVITA[rng.choice(n, size=n // 6, replace=False)] = True
h_obs = h_n * np.where(CAVITA, 0.78, 1.0) + 0.02 * rng.normal(size=n)

df = pd.DataFrame({"machine_id": maq, "rpm_mean": rpm,
                   "flow_mean": q_n * rpm_n, "delta_p_mean": h_obs * rpm_n**2,
                   "pressure_in_mean": (3.0 - np.where(CAVITA, 1.4, 0.0)) * rpm_n**2
                                       + 0.02 * rng.normal(size=n)})
for c in ("current_a__rms", "current_b__rms", "current_c__rms"): df[c] = rng.uniform(9, 11, n)
for c in ("voltage_a__rms", "voltage_b__rms", "voltage_c__rms"): df[c] = rng.uniform(219, 221, n)

G = dict(np=np, pd=pd)
exec(S["fn_hidraulicas"].split("def confusion_par")[0], G)
X = G["features_hidraulicas"](df)
ok = lambda t: print("  OK ", t)

assert X.shape == (n, 6) and np.isfinite(X.to_numpy()).all()
ok(f"{X.shape[1]} features, {n} filas, todas finitas")

r = X["hid_resid_H"].to_numpy()
assert r[CAVITA].mean() < r[~CAVITA].mean() - 3 * r[~CAVITA].std()
ok(f"hid_resid_H separa cavitación ({r[CAVITA].mean():+.3f}) de sana "
   f"({r[~CAVITA].mean():+.3f}, sd {r[~CAVITA].std():.3f})")

npsh = X["hid_npsh"].to_numpy()
assert npsh[CAVITA].mean() < npsh[~CAVITA].mean()
ok(f"hid_npsh baja al cavitar ({npsh[CAVITA].mean():+.3f} vs {npsh[~CAVITA].mean():+.3f})")

X2 = G["features_hidraulicas"](df.assign(delta_p_mean=h_n * rpm_n**2))
assert abs(X2["hid_resid_H"]).max() < 1e-8
ok("con una bomba perfecta el residuo es cero a precisión numérica")

X3 = G["features_hidraulicas"](pd.concat([df, df.head(3).assign(machine_id="MRARA")],
                                         ignore_index=True))
assert np.isfinite(X3.to_numpy()).all() and (X3.tail(3)["hid_resid_H"] == 0).all()
ok("máquina con 3 filas: devuelve 0 en vez de romper o sobreajustar")

assert np.isfinite(G["features_hidraulicas"](
    df.assign(flow_mean=0.0, delta_p_mean=0.0)).to_numpy()).all()
ok("caudal y altura en cero: sin NaN ni inf")
print("\nTODAS LAS PRUEBAS HIDRÁULICAS PASARON")
