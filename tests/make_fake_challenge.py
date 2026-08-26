"""Genera un desafio SINTETICO tipo 'triaje hospitalario' con datos sucios.

Sirve para dos cosas:
  1. probar que el kit funciona antes de la hackaton
  2. ENSAYO GENERAL: cronometrate haciendo el desafio falso de punta a punta
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fake_data")
rng = np.random.default_rng(7)
N = 6000


def build(n, seed):
    r = np.random.default_rng(seed)
    edad = np.clip(r.normal(52, 20, n), 0, 99)
    fc = r.normal(85, 18, n)
    sat = np.clip(r.normal(96, 4, n), 60, 100)
    temp = r.normal(37.0, 0.9, n)
    pas = r.normal(128, 22, n)
    lact = np.exp(r.normal(0.3, 0.5, n))
    dolor = r.integers(0, 11, n)
    sexo = r.choice(["Masculino", "Femenino"], n)
    antec = r.choice(["diabetes", "hipertension", "ninguno", "epoc"], n,
                     p=[.2, .3, .4, .1])
    riesgo = (0.045 * (edad - 50) + 0.06 * (fc - 85) - 0.22 * (sat - 96)
              + 0.9 * (temp - 37) + 1.3 * np.log(lact) + 0.10 * dolor
              - 0.02 * (pas - 128) + (antec == "epoc") * 1.1
              + r.normal(0, 1.1, n))
    y = np.digitize(riesgo, np.quantile(riesgo, [0.45, 0.75, 0.92]))
    return pd.DataFrame({
        "id_paciente": np.arange(seed * 100000, seed * 100000 + n),
        "edad": edad.round(0), "frecuencia_cardiaca": fc.round(0),
        "saturacion": sat.round(1), "temperatura": temp.round(1),
        "presion_sistolica": pas.round(0), "lactato": lact.round(2),
        "escala_dolor": dolor, "sexo": sexo, "antecedente": antec,
        "nivel_urgencia": y,
    })


def dirty(df, seed):
    """Ensucia como lo haria personal cargando a mano."""
    r = np.random.default_rng(seed)
    d = df.copy()
    n = len(d)
    # temperatura: 12% cargada en Fahrenheit
    m = r.random(n) < 0.12
    d.loc[m, "temperatura"] = (d.loc[m, "temperatura"] * 9 / 5 + 32).round(1)
    # lactato como texto con coma decimal + unidad
    d["lactato"] = d["lactato"].map(lambda v: str(v).replace(".", ",") + " mmol/L")
    # centinelas y faltantes
    d.loc[r.random(n) < 0.05, "presion_sistolica"] = -999
    d.loc[r.random(n) < 0.08, "saturacion"] = np.nan
    d.loc[r.random(n) < 0.06, "escala_dolor"] = np.nan
    # typos y mayusculas en categoricas
    d["sexo"] = d["sexo"].map(lambda s: r.choice([s, s.upper(), s.lower(), s + " ",
                                                  "M" if s[0] == "M" else "F"]))
    d["antecedente"] = d["antecedente"].map(
        lambda s: r.choice([s, s.capitalize(), s.upper()]))
    # columna basura constante + columna de fuga aparente
    d["centro"] = "HOSPITAL CENTRAL"
    d["observacion"] = r.choice(["", "sin datos", "-"], n)
    return d


OUT.mkdir(parents=True, exist_ok=True)
full = build(N, 1)
test = build(2000, 2)
unl = build(3000, 3).drop(columns="nivel_urgencia")

labels = ["No urgente", "Urgente", "Muy urgente", "Crítico"]
tr = dirty(full, 11)
tr["nivel_urgencia"] = [labels[i] for i in tr["nivel_urgencia"]]
te_truth = test[["id_paciente", "nivel_urgencia"]].copy()
te_truth["nivel_urgencia"] = [labels[i] for i in te_truth["nivel_urgencia"]]
te = dirty(test.drop(columns="nivel_urgencia"), 12)

tr.to_csv(OUT / "train_labeled.csv", index=False)
te.to_csv(OUT / "test_features.csv", index=False)
dirty(unl, 13).to_csv(OUT / "train_unlabeled.csv", index=False)
te_truth.to_csv(OUT / "_solucion_oculta.csv", index=False)
print("datos falsos en", OUT.resolve())
print(tr.head(3).to_string())
