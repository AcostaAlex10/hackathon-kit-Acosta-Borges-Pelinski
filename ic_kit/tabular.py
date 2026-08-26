"""Pipeline tabular cost-sensitive de punta a punta.

Orden de las palancas por rentabilidad real en estos desafios:

  1. capa de decision bayesiana (costs.bayes_decision)      +++++
  2. re-pesado de probabilidades ajustado por OOF (fit_prior) ++++
  3. limpieza de datos sucios (cleaning.auto_clean)          ++++
  4. ensamble de semillas / modelos                          +++
  5. feature engineering de dominio                          +++
  6. tuneo de hiperparametros                                +
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .cleaning import is_text
from .costs import bayes_decision, mean_cost

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:                                            # pragma: no cover
    HAS_LGB = False

LGB_PARAMS = dict(
    objective="multiclass", learning_rate=0.05, num_leaves=63,
    min_child_samples=25, feature_fraction=0.85, bagging_fraction=0.85,
    bagging_freq=1, lambda_l2=1.0, n_estimators=3000, verbosity=-1,
)


# ------------------------------------------------------------- preparacion
def prepare(train: pd.DataFrame, test: pd.DataFrame, target: str,
            id_col: str | None = None, drop=(), class_order=None):
    """Alinea train/test, castea categoricas y devuelve (X, y, Xte, ids, clases).

    !! `class_order` NO ES OPCIONAL SI USAS UNA MATRIZ DE COSTOS CON NOMBRE !!
    Sin el, las clases se ordenan alfabeticamente y la matriz queda desalineada
    (ej. ['Critico','Muy urgente','No urgente','Urgente'] vs el orden clinico
    ['No urgente','Urgente','Muy urgente','Critico']). Es el error mas caro y
    mas silencioso de este tipo de competencia: el codigo corre y el score baja.
    """
    drop = set(drop) | ({id_col} if id_col else set())
    feats = [c for c in train.columns if c not in drop and c != target]
    feats = [c for c in feats if c in test.columns]

    X, Xte = train[feats].copy(), test[feats].copy()
    for c in feats:
        if is_text(X[c]) or str(X[c].dtype) == "category":
            cats = pd.Index(sorted(set(X[c].dropna().astype(str))
                                   | set(Xte[c].dropna().astype(str))))
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=cats)
            Xte[c] = pd.Categorical(Xte[c].astype(str).where(Xte[c].notna()), categories=cats)

    y_raw = train[target]
    present = sorted(pd.unique(y_raw.dropna()).tolist(), key=str)
    if class_order is not None:
        classes = list(class_order)
        falt = set(map(str, present)) - set(map(str, classes))
        if falt:
            raise ValueError("class_order no cubre estas clases del train: %s" % sorted(falt))
    else:
        classes = present
        print("[AVISO] orden de clases inferido alfabeticamente: %s\n"
              "        si tu matriz de costos tiene otro orden, pasa class_order=" % classes)
    y = pd.Series(y_raw).map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
    ids = test[id_col] if id_col and id_col in test.columns else pd.Series(np.arange(len(test)))
    return X, y, Xte, ids, classes


# ------------------------------------------------------------------- OOF
def oof_lgb(X, y, Xte, params: dict | None = None, n_splits: int = 5,
            seeds=(42, 43, 44), sample_weight=None, cat_smooth_cv: bool = True,
            verbose: bool = True):
    """Out-of-fold + prediccion de test promediando semillas.

    Devuelve (oof_proba, test_proba, info). El OOF es tu unico termometro
    honesto: NUNCA mires el test para decidir nada.
    """
    if not HAS_LGB:
        return _oof_sklearn(X, y, Xte, n_splits, seeds, sample_weight)
    p = {**LGB_PARAMS, **(params or {})}
    k = len(np.unique(y))
    p["num_class"] = k
    oof = np.zeros((len(X), k))
    pte = np.zeros((len(Xte), k))
    iters = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(**p, random_state=seed)
            sw = None if sample_weight is None else np.asarray(sample_weight)[tr]
            m.fit(X.iloc[tr], y[tr], sample_weight=sw,
                  eval_set=[(X.iloc[va], y[va])],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(0)])
            oof[va] += m.predict_proba(X.iloc[va]) / len(seeds)
            pte += m.predict_proba(Xte) / (len(seeds) * n_splits)
            iters.append(m.best_iteration_ or p["n_estimators"])
    info = {"mean_best_iter": float(np.mean(iters)), "n_models": len(iters)}
    if verbose:
        print("OOF listo:", info)
    return oof, pte, info


def _oof_sklearn(X, y, Xte, n_splits, seeds, sample_weight):
    """Fallback sin LightGBM (HistGradientBoosting de sklearn)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xn = X.copy()
    Xten = Xte.copy()
    for c in Xn.columns:
        if str(Xn[c].dtype) == "category":
            Xn[c] = Xn[c].cat.codes.replace(-1, np.nan)
            Xten[c] = Xten[c].cat.codes.replace(-1, np.nan)
    k = len(np.unique(y))
    oof = np.zeros((len(Xn), k))
    pte = np.zeros((len(Xten), k))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in skf.split(Xn, y):
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=400,
                                               learning_rate=0.06)
            sw = None if sample_weight is None else np.asarray(sample_weight)[tr]
            m.fit(Xn.iloc[tr], y[tr], sample_weight=sw)
            oof[va] += m.predict_proba(Xn.iloc[va]) / len(seeds)
            pte += m.predict_proba(Xten) / (len(seeds) * n_splits)
    return oof, pte, {"backend": "sklearn-hgb"}


# ------------------------------------------- re-pesado de probabilidades
def fit_prior(proba: np.ndarray, y, C: np.ndarray, n_rounds: int = 4,
              grid=None, verbose: bool = True) -> np.ndarray:
    """Busca w tal que decidir sobre (proba*w) minimice el costo OOF.

    Corrige de un saque: mala calibracion, desbalance y prior shift.
    Aplicalo SOLO con probabilidades OOF; despues usa el mismo w en test.
    """
    proba = np.asarray(proba, dtype=float)
    k = proba.shape[1]
    grid = grid if grid is not None else np.concatenate(
        [np.linspace(0.2, 1.0, 17), np.linspace(1.1, 6.0, 25)])
    w = np.ones(k)
    best = mean_cost(y, bayes_decision(proba * w, C), C)
    for _ in range(n_rounds):
        improved = False
        for j in range(k):
            cur = w[j]
            for g in grid:
                w[j] = g
                c = mean_cost(y, bayes_decision(proba * w, C), C)
                if c < best - 1e-12:
                    best, cur, improved = c, g, True
            w[j] = cur
        if not improved:
            break
    w = w / w.mean()
    if verbose:
        print("pesos de decision w =", np.round(w, 3), "| costo OOF =", round(best, 5))
    return w


def blend_weights(probas: list[np.ndarray], y, C: np.ndarray,
                  n_iter: int = 300, seed: int = 42, verbose: bool = True):
    """Pesos de ensamble que minimizan el costo OOF (dirichlet aleatorio + greedy)."""
    rng = np.random.default_rng(seed)
    n = len(probas)
    best_w = np.ones(n) / n
    best = mean_cost(y, bayes_decision(sum(w * p for w, p in zip(best_w, probas)), C), C)
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n))
        c = mean_cost(y, bayes_decision(sum(wi * p for wi, p in zip(w, probas)), C), C)
        if c < best:
            best, best_w = c, w
    if verbose:
        print("pesos de blend =", np.round(best_w, 3), "| costo OOF =", round(best, 5))
    return best_w, best


# ------------------------------------------------------- semisupervisado
def pseudo_label(X, y, X_unlabeled, Xte, C, params=None, conf: float = 0.9,
                 n_splits: int = 5, seeds=(42,), verbose: bool = True):
    """Auto-entrenamiento con el archivo train_unlabeled.

    Solo agrega filas donde el modelo esta MUY seguro. Verifica siempre en OOF
    que el costo baje; si no baja, descarta esta rama (pasa seguido).
    """
    oof, p_unl, _ = oof_lgb(X, y, X_unlabeled, params, n_splits, seeds, verbose=False)
    base = mean_cost(y, bayes_decision(oof, C), C)
    keep = p_unl.max(axis=1) >= conf
    if verbose:
        print("pseudo-etiquetas confiables: %d/%d (conf>=%.2f)"
              % (int(keep.sum()), len(keep), conf))
    if keep.sum() < 30:
        return None
    Xa = pd.concat([X, X_unlabeled.loc[keep]], ignore_index=True)
    ya = np.concatenate([y, p_unl[keep].argmax(axis=1)])
    oof2, pte2, _ = oof_lgb(Xa, ya, Xte, params, n_splits, seeds, verbose=False)
    new = mean_cost(ya[:len(y)], bayes_decision(oof2[:len(y)], C), C)
    if verbose:
        print("costo OOF  base=%.5f  con pseudo=%.5f  ->  %s"
              % (base, new, "USAR" if new < base else "DESCARTAR"))
    return (oof2, pte2) if new < base else None


# ------------------------------------------------------------ importancia
def permutation_cost_importance(model_predict, X, y, C, n_repeats: int = 3,
                                seed: int = 42) -> pd.DataFrame:
    """Importancia medida en la METRICA REAL, no en logloss."""
    rng = np.random.default_rng(seed)
    base = mean_cost(y, bayes_decision(model_predict(X), C), C)
    rows = []
    for c in X.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = Xp[c].to_numpy()[rng.permutation(len(Xp))]
            deltas.append(mean_cost(y, bayes_decision(model_predict(Xp), C), C) - base)
        rows.append({"feature": c, "delta_costo": float(np.mean(deltas))})
    return pd.DataFrame(rows).sort_values("delta_costo", ascending=False)
