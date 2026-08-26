"""Matrices de costo y decision optima bayesiana.

IDEA CENTRAL DEL KIT
--------------------
Cuando la metrica es una matriz de costos asimetrica, `argmax p(y|x)` NO es
la decision optima. La decision que minimiza el costo esperado es:

    pred(x) = argmin_j  sum_i  P(y=i|x) * C[i, j]

es decir `argmin (proba @ C)`. Entrenas un modelo *calibrado* que estime bien
las probabilidades, y despues aplicas esta capa de decision. En los desafios
2026 esto suele valer mas puntos que cualquier tuneo de hiperparametros.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- Desafio 1
LABELS_D1 = ["No urgente", "Urgente", "Muy urgente", "Crítico"]

# C[real, predicho]
COST_D1 = np.array(
    [
        [0, 1, 2, 4],    # real No urgente
        [2, 0, 1, 3],    # real Urgente
        [4, 2, 0, 2],    # real Muy urgente
        [10, 6, 2, 0],   # real Crítico
    ],
    dtype=float,
)


def score_d1(y_true, y_pred, C: np.ndarray = COST_D1) -> float:
    """Score del desafio 1: 1 / (1 + costo_promedio). Mayor es mejor (max 1.0)."""
    return 1.0 / (1.0 + mean_cost(y_true, y_pred, C))


# ---------------------------------------------------------------- Desafio 2
LABELS_D2 = [
    "Pecari_tajacu", "Nasua_nasua", "Harpia_harpyja", "Caiman_latirostris",
    "Phyllomedusa_distincta", "Aspidosperma_polyneuron",
    "Philodendron_bipinnatifidum", "Dicksonia_sellowiana",
    "Aechmea_distichantha", "Ilex_paraguariensis",
]
# 0-4 Animalia, 5-9 Plantae
KINGDOM_D2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def grouped_cost_matrix(groups, same_group: float = 2.0, diff_group: float = 5.0,
                        correct: float = 0.0) -> np.ndarray:
    """Matriz de costos jerarquica (ej. reino biologico del desafio 2)."""
    g = np.asarray(groups)
    k = len(g)
    C = np.where(g[:, None] == g[None, :], same_group, diff_group).astype(float)
    np.fill_diagonal(C, correct)
    return C


COST_D2 = grouped_cost_matrix(KINGDOM_D2, same_group=2.0, diff_group=5.0)


def score_d2(y_true, y_pred, C: np.ndarray = COST_D2) -> float:
    """Metrica del desafio 2: costo promedio. MENOR es mejor (min 0, max 5)."""
    return mean_cost(y_true, y_pred, C)


# ---------------------------------------------------------------- Genericos
def mean_cost(y_true, y_pred, C: np.ndarray) -> float:
    """Costo promedio por muestra dada C[real, predicho]."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(C[y_true, y_pred].mean())


def expected_cost(proba: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Costo esperado de cada decision j para cada fila. Shape (n, k)."""
    return np.asarray(proba, dtype=float) @ np.asarray(C, dtype=float)


def bayes_decision(proba: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Decision que minimiza el costo esperado. Reemplazo directo de argmax."""
    return expected_cost(proba, C).argmin(axis=1)


def decision_gain(y_true, proba: np.ndarray, C: np.ndarray) -> dict:
    """Cuanto ganas al usar bayes_decision en vez de argmax. Reportalo siempre."""
    am = np.asarray(proba).argmax(axis=1)
    bd = bayes_decision(proba, C)
    return {
        "cost_argmax": mean_cost(y_true, am, C),
        "cost_bayes": mean_cost(y_true, bd, C),
        "acc_argmax": float((np.asarray(y_true) == am).mean()),
        "acc_bayes": float((np.asarray(y_true) == bd).mean()),
        "pct_changed": float((am != bd).mean()),
    }


def confusion_cost_report(y_true, y_pred, C: np.ndarray, labels=None) -> str:
    """Tabla de donde se te va el costo. Te dice que error atacar primero."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = C.shape[0]
    labels = labels or [str(i) for i in range(k)]
    n = len(y_true)
    rows = []
    for i in range(k):
        for j in range(k):
            cnt = int(((y_true == i) & (y_pred == j)).sum())
            if cnt and C[i, j] > 0:
                rows.append((cnt * C[i, j] / n, cnt, labels[i], labels[j], C[i, j]))
    rows.sort(reverse=True)
    out = [f"Costo promedio: {mean_cost(y_true, y_pred, C):.4f}  (n={n})",
           f"{'aporte':>8} {'casos':>6} {'costo_u':>8}  real -> predicho"]
    for share, cnt, li, lj, c in rows:
        out.append(f"{share:8.4f} {cnt:6d} {c:8.1f}  {li} -> {lj}")
    return "\n".join(out)


# ------------------------------------------------- pesos de entrenamiento
def class_weights_from_cost(C: np.ndarray) -> np.ndarray:
    """Peso por clase real = costo medio de equivocarse en ella.

    Util como `sample_weight` para que el modelo se esfuerce donde duele.
    Ojo: distorsiona la calibracion, por eso el kit prueba SIEMPRE las dos
    variantes (con y sin pesos) y se queda con la mejor por CV.
    """
    w = C.sum(axis=1) / max(1, (C.shape[1] - 1))
    return w / w.mean()


# ------------------------------------------------ Desafio de ensayo (practica)
LABELS_ENSAYO = ["Normal", "Vigilancia", "Mantenimiento urgente", "Falla inminente"]

COST_ENSAYO = np.array(
    [
        [0, 1, 3, 6],      # real Normal
        [2, 0, 1, 4],      # real Vigilancia
        [5, 3, 0, 2],      # real Mantenimiento urgente
        [15, 9, 3, 0],     # real Falla inminente
    ],
    dtype=float,
)
