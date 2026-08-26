"""Genera los datos del DESAFIO DE ENSAYO (ver ENSAYO_DESAFIO.md).

Incluye trampas deliberadas. NO leas este archivo si vas a hacer el ensayo:
arruina el ejercicio. La lista de trampas está sellada al final de
ENSAYO_DESAFIO.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/ensayo_data")
LABELS = ["Normal", "Vigilancia", "Mantenimiento urgente", "Falla inminente"]


def build(n, seed):
    r = np.random.default_rng(seed)
    horas = np.clip(r.gamma(4, 9000, n), 500, 180000)
    temp = r.normal(62, 11, n) + horas / 20000
    carga = np.clip(r.beta(4, 3, n) * 130, 5, 145)
    h2 = np.exp(r.normal(3.4, 1.0, n))
    ch4 = np.exp(r.normal(2.6, 0.95, n))
    c2h2 = np.exp(r.normal(0.5, 1.35, n))
    humedad = np.clip(r.normal(14, 7, n), 0, 60)
    rigidez = np.clip(r.normal(58, 13, n), 10, 90)
    vibr = np.abs(r.normal(2.1, 1.0, n))
    marca = r.choice(["siemens", "abb", "tadeo czerweny", "wegtrafo"], n,
                     p=[.3, .28, .27, .15])
    refrig = r.choice(["onan", "onaf", "ofaf"], n, p=[.5, .35, .15])

    riesgo = (0.9 * np.log1p(c2h2) + 0.45 * np.log1p(h2) + 0.30 * np.log1p(ch4)
              + 0.05 * (temp - 62) + 0.055 * (humedad - 14)
              - 0.035 * (rigidez - 58) + 0.55 * vibr
              + 0.018 * (carga - 80) + horas / 60000
              + (refrig == "onan") * 0.35 + r.normal(0, 1.25, n))
    # cortes FIJOS: la definicion de cada estado es la misma en train y test.
    # El shift del test se hace remuestreando filas, no moviendo los cortes,
    # para que sea un prior shift puro (p(x|y) intacto) y por lo tanto
    # corregible con el EM de Saerens.
    cortes = np.array([5.5481, 7.0770, 8.3412])
    y = np.digitize(riesgo, cortes)
    return pd.DataFrame({
        "id_unidad": np.arange(seed * 200000, seed * 200000 + n),
        "horas_servicio": horas.round(0),
        "temperatura_aceite": temp.round(1),
        "carga_pct": carga.round(1),
        "gas_h2_ppm": h2.round(2),
        "gas_ch4_ppm": ch4.round(2),
        "gas_c2h2_ppm": c2h2.round(3),
        "humedad_aceite_ppm": humedad.round(1),
        "rigidez_dielectrica_kv": rigidez.round(1),
        "vibracion_mm_s": vibr.round(2),
        "marca": marca, "refrigeracion": refrig,
        "estado": y,
    })


def ensuciar(d, seed, es_test=False):
    r = np.random.default_rng(seed)
    d = d.copy()
    n = len(d)
    # 1. temperatura: 14% cargada en Fahrenheit
    m = r.random(n) < 0.14
    d.loc[m, "temperatura_aceite"] = (d.loc[m, "temperatura_aceite"] * 9 / 5 + 32).round(1)
    # 2. gas c2h2 como texto con coma decimal y unidad
    d["gas_c2h2_ppm"] = d["gas_c2h2_ppm"].map(lambda v: str(v).replace(".", ",") + " ppm")
    # 3. centinelas y faltantes
    d.loc[r.random(n) < 0.06, "rigidez_dielectrica_kv"] = -999
    d.loc[r.random(n) < 0.09, "humedad_aceite_ppm"] = np.nan
    # 4. typos y mayúsculas
    d["marca"] = d["marca"].map(lambda s: r.choice([s, s.upper(), s.title(), s + " "]))
    # 5. columna constante
    d["distribuidora"] = "EMSA"
    # 6. columna casi vacía en test (en train está poblada)
    if es_test:
        d["inspeccion_visual"] = np.nan
    else:
        d["inspeccion_visual"] = r.choice(["ok", "leve", "observada"], n, p=[.6, .3, .1])
    return d


OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(99)

train = build(7000, 1)
# TRAMPA: prior shift PURO en el test (se remuestrea, no se redefinen clases)
_pool = build(150000, 2)
_objetivo = {0: 0.40, 1: 0.28, 2: 0.18, 3: 0.14}
_r = np.random.default_rng(500)
test = pd.concat([
    _pool[_pool.estado == c].sample(int(2500 * f), random_state=int(1000 + c))
    for c, f in _objetivo.items()
]).sample(frac=1.0, random_state=7).reset_index(drop=True)
test["id_unidad"] = np.arange(400000, 400000 + len(test))

# TRAMPA: fuga por columna de "orden de trabajo" que codifica el estado
train["orden_trabajo"] = train["estado"] * 1000 + rng.integers(0, 1000, len(train))
test["orden_trabajo"] = rng.integers(0, 4000, len(test))   # rota en el test

# TRAMPA: 5% de etiquetas mal cargadas
ruido = rng.random(len(train)) < 0.05
train.loc[ruido, "estado"] = rng.integers(0, 4, int(ruido.sum()))

# TRAMPA: las filas vienen agrupadas por estado (no mezcladas)
train = train.sort_values("estado", kind="stable").reset_index(drop=True)

sol = test[["id_unidad", "estado"]].copy()
sol["estado"] = [LABELS[i] for i in sol["estado"]]

tr = ensuciar(train, 21)
tr["estado"] = [LABELS[i] for i in tr["estado"]]
te = ensuciar(test.drop(columns="estado"), 22, es_test=True)

tr.to_csv(OUT / "train_labeled.csv", index=False)
te.to_csv(OUT / "test_features.csv", index=False)
ensuciar(build(3500, 3).drop(columns="estado"), 23).to_csv(
    OUT / "train_unlabeled.csv", index=False)
sol.to_csv(OUT / "_solucion_sellada.csv", index=False)

print("datos del ensayo en", OUT.resolve())
print("train", tr.shape, "| test", te.shape)
print(tr["estado"].value_counts(normalize=True).round(3).to_string())
