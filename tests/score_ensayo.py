"""Puntua un submit del desafio de ensayo contra la solucion sellada."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ic_kit.costs import COST_ENSAYO, LABELS_ENSAYO, confusion_cost_report, mean_cost

sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "work/submit.csv")
sol = pd.read_csv(Path(__file__).parent / "ensayo_data" / "_solucion_sellada.csv")
m = sol.merge(sub, on="id_unidad", suffixes=("_true", "_pred"))
if len(m) != len(sol):
    print("[ERROR] el submit cubre %d de %d unidades" % (len(m), len(sol)))
    sys.exit(1)

idx = {c: i for i, c in enumerate(LABELS_ENSAYO)}
yt = m.iloc[:, 1].map(idx).to_numpy()
yp = m.iloc[:, 2].map(idx).to_numpy()
c = mean_cost(yt, yp, COST_ENSAYO)
veredicto = ("EXCELENTE" if c <= 1.15 else "BUENO" if c <= 1.30 else
             "ACEPTABLE" if c <= 1.60 else "INSUFICIENTE (no aprueba)")
print("COSTO REAL = %.4f   ->   %s" % (c, veredicto))
print("accuracy   = %.4f" % float((yt == yp).mean()))
print()
print("referencias medidas:  mejor constante 1.900 | limpio+argmax 1.904")
print("                      limpio+bayes    1.196 | +EM de prior    1.119")
print("                      oraculo de prior 1.095 | con la fuga    2.877")
print()
print(confusion_cost_report(yt, yp, COST_ENSAYO, labels=LABELS_ENSAYO))
