"""Scorer vectorizado, verificado contra scoring.compute_score."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import FAULT_IDS, FAMILIES, LABEL_COLUMNS, COSTS

FA = FAULT_IDS
FAMS = sorted(set(FAMILIES.values()))              # electrical, hydraulic, mechanical, structural
FIDX = {g: np.array([i for i, f in enumerate(FA) if FAMILIES[f] == g]) for g in FAMS}

def prep_truth(df):
    """Devuelve (y13, fam_mask(n,4), missed(n,), tiene_falla(n,))."""
    y = df[LABEL_COLUMNS].to_numpy(int)
    sev = df[[f"severity_{f}" for f in FA]].to_numpy(float)
    hay = y.sum(1) > 0
    maxsev = np.where(y == 1, sev, -1).max(1)
    missed = np.where(maxsev > 0.5, COSTS["false_negative_severe"], COSTS["false_negative_mild"])
    fam = np.zeros((len(df), len(FAMS)), bool)
    for k, g in enumerate(FAMS):
        fam[:, k] = y[:, FIDX[g]].sum(1) > 0
    return y, fam, missed, hay

def fast_score(P, truth_pack):
    """P: (n,13) probabilidades. truth_pack: salida de prep_truth."""
    y, fam, missed, hay = truth_pack
    P = np.clip(np.asarray(P, float), 0, 1)
    n = len(P)
    S = P.sum(1)
    # costos esperados de cada acción segun el scorer
    cand = np.empty((n, 3 + len(FAMS)))
    cand[:, 0] = 10.0 * S                                   # MONITOR
    cand[:, 1] = COSTS["derate"] + 0.35 * 10.0 * S          # DERATE
    cand[:, 2] = COSTS["stop"]                              # STOP
    for k, g in enumerate(FAMS):
        Sg = P[:, FIDX[g]].sum(1)
        cand[:, 3 + k] = COSTS["inspect"] + COSTS["misdiagnosis"] * (S - Sg)
    a = cand.argmin(1)
    # costo real
    nfam = fam.sum(1)
    cost = np.zeros(n)
    sano = ~hay
    cost[sano & (a == 0)] = 0.0
    cost[sano & (a == 1)] = COSTS["derate"]
    cost[sano & (a == 2)] = COSTS["stop"]
    cost[sano & (a >= 3)] = COSTS["inspect"]
    m = hay & (a == 0); cost[m] = missed[m]
    m = hay & (a == 1); cost[m] = COSTS["derate"] + 0.35 * missed[m]
    m = hay & (a == 2); cost[m] = COSTS["stop"]
    for k in range(len(FAMS)):
        m = hay & (a == 3 + k)
        ok = m & fam[:, k]
        bad = m & ~fam[:, k]
        frac = np.zeros(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac[ok] = (nfam[ok] - 1) / np.maximum(nfam[ok], 1)
        cost[ok] = COSTS["inspect"] + frac[ok] * missed[ok]
        cost[bad] = COSTS["inspect"] + COSTS["misdiagnosis"]
    naive = np.where(hay, missed, 0.0)
    mean_cost, naive_cost = cost.mean(), naive.mean()
    cost_score = 100 * np.clip(1 - mean_cost / (naive_cost + 1e-12), 0, 1)
    brier = float(((P - y) ** 2).mean())
    prob_score = 100 * np.clip(1 - brier / 0.25, 0, 1)
    b = (P >= 0.5).astype(int)
    f1s = []
    for j in range(13):
        tp = ((y[:, j] == 1) & (b[:, j] == 1)).sum()
        fp = ((y[:, j] == 0) & (b[:, j] == 1)).sum()
        fn = ((y[:, j] == 1) & (b[:, j] == 0)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    diag = 100 * float(np.mean(f1s))
    return dict(final_score=0.7 * cost_score + 0.2 * prob_score + 0.1 * diag,
                cost_score=cost_score, prob_score=prob_score, diag_score=diag,
                brier=brier, mean_cost=mean_cost, naive_cost=naive_cost)
