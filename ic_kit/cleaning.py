"""Auditoria y limpieza automatica de datasets sucios.

Las consignas avisan: "varios registros fueron completados a mano, no se
descarta la existencia de errores u omisiones". Ese es el terreno donde se
gana la competencia. Este modulo detecta, en orden de rentabilidad:

  1. columnas ID / constantes / casi-constantes  (ruido o fuga)
  2. numeros guardados como texto ("36,5", "37 C", "1.234")
  3. valores centinela (-999, 999) que fingen ser datos
  4. MEZCLA DE UNIDADES (temp en C y F, peso en kg y lb) -> bimodalidad
  5. categorias con typos / mayusculas / acentos ("Masculino", "masculino ", "M")
  6. duplicados exactos y duplicados con etiqueta contradictoria
  7. drift train vs test por columna (PSI) -> features en las que no confiar
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

SENTINELS = {-999, -99, -9, 999, 9999, -1.0e9, 1.0e9}
NA_STRINGS = {"", "na", "n/a", "nan", "null", "none", "sin dato", "s/d", "-",
              "?", "desconocido", "ninguno", "unknown"}


# ----------------------------------------------------------------- helpers
def is_text(s: pd.Series) -> bool:
    """True para columnas de texto (pandas <3 usa object, pandas 3 usa str)."""
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)         and str(s.dtype) not in ("category",)         and not pd.api.types.is_datetime64_any_dtype(s)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_text(s):
    if pd.isna(s):
        return np.nan
    s = _strip_accents(str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return np.nan if s in NA_STRINGS else s


def to_number(s):
    """Convierte '36,5' / '1.234' / '37 C' / '  12 ' a float. NaN si no puede."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    t = re.sub(r"[^\d,.\-+eE]", "", str(s).strip())
    if t in ("", "-", "+", ".", ","):
        return np.nan
    if "," in t and "." in t:            # 1.234,56 -> 1234.56
        t = t.replace(".", "").replace(",", ".")
    elif "," in t:                       # 36,5 -> 36.5
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def numeric_ratio(s: pd.Series) -> float:
    v = s.dropna().astype(str).head(2000)
    if not len(v):
        return 0.0
    return float(v.map(lambda x: not np.isnan(to_number(x))).mean())


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    """Population Stability Index train vs test. >0.25 = drift fuerte."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 20 or len(b) < 20:
        return float("nan")
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    pa = np.histogram(a, bins=qs)[0] / len(a) + 1e-6
    pb = np.histogram(b, bins=qs)[0] / len(b) + 1e-6
    return float(((pa - pb) * np.log(pa / pb)).sum())


def bimodality_gap(v: np.ndarray) -> float:
    """Deteccion de mezcla de unidades: separacion entre 2 clusters 1D.

    Devuelve  |mu_b - mu_a| / desvio_intra_cluster  con el mejor corte.
    Una normal unimodal da ~2.7; una mezcla real (C vs F, kg vs lb) da >6.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 60:
        return 0.0
    lo, hi = np.quantile(v, [0.01, 0.99])
    v = np.sort(v[(v >= lo) & (v <= hi)])
    if len(v) < 60 or v.std() == 0:
        return 0.0
    best = 0.0
    for q in np.linspace(0.05, 0.95, 46):
        c = np.quantile(v, q)
        a, b = v[v <= c], v[v > c]
        if min(len(a), len(b)) < max(20, 0.03 * len(v)):
            continue
        within = np.sqrt((len(a) * a.var() + len(b) * b.var()) / len(v)) + 1e-9
        best = max(best, abs(b.mean() - a.mean()) / within)
    return float(best)


# ------------------------------------------------------------------ audit
def audit(df: pd.DataFrame, target: str | None = None,
          df_test: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ficha tecnica columna por columna. Lo PRIMERO que corres."""
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        r = {"col": c, "dtype": str(s.dtype),
             "missing_%": round(100 * float(s.isna().mean()), 2),
             "n_unique": nun, "unique_%": round(100 * nun / max(n, 1), 2)}
        flags = []
        if nun <= 1:
            flags.append("CONSTANTE")
        if nun == n and n > 50:
            flags.append("ID?")
        if is_text(s):
            nr = numeric_ratio(s)
            r["num_ratio"] = round(nr, 3)
            if nr > 0.9 and nun > 15:
                flags.append("NUMERO_COMO_TEXTO")
            elif nun <= 40:
                vals = s.dropna().astype(str)
                if len(vals) and vals.map(norm_text).nunique() < nun:
                    flags.append("TYPOS/CASE")
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            fin = v[np.isfinite(v)]
            if len(fin):
                r["min"], r["median"], r["max"] = (float(fin.min()),
                                                   float(np.median(fin)),
                                                   float(fin.max()))
                if set(np.unique(fin).tolist()) & SENTINELS:
                    flags.append("CENTINELA")
                if nun > 20 and bimodality_gap(v) > 5.0:
                    flags.append("MEZCLA_UNIDADES?")
                mad = 1.4826 * np.median(np.abs(fin - np.median(fin)))
                if mad > 0 and (np.abs(fin - np.median(fin)) / mad > 12).mean() > 0.001:
                    flags.append("OUTLIERS_EXTREMOS")
        if df_test is not None and c in df_test.columns and c != target:
            # trampa clásica: la columna está poblada en el train y vacía en el
            # test. El modelo se apoya en ella y la pierde justo en la inferencia.
            miss_te = float(df_test[c].isna().mean())
            r["missing_test_%"] = round(100 * miss_te, 2)
            if miss_te > 0.95 and s.isna().mean() < 0.5:
                flags.append("VACIA_EN_TEST")
            elif miss_te - float(s.isna().mean()) > 0.3:
                flags.append("MAS_VACIA_EN_TEST")
            if not is_text(s):
                p = psi(s, df_test[c])
                r["psi_test"] = None if np.isnan(p) else round(p, 3)
                if not np.isnan(p) and p > 0.25:
                    flags.append("DRIFT_TEST")
            else:
                new = set(df_test[c].dropna().map(norm_text)) - set(s.dropna().map(norm_text))
                if new:
                    flags.append("CATS_NUEVAS(%d)" % len(new))
        r["alertas"] = ",".join(flags)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["alertas", "missing_%"],
                                          ascending=[False, False])


def duplicate_report(df: pd.DataFrame, target: str | None = None,
                     ignore=()) -> dict:
    feats = [c for c in df.columns if c != target and c not in ignore]
    contradictory = 0
    if target and target in df.columns and feats:
        g = df.groupby(feats, dropna=False)[target].nunique()
        contradictory = int((g > 1).sum())
    return {"filas_duplicadas": int(df.duplicated().sum()),
            "duplicadas_solo_features": int(df.duplicated(subset=feats).sum()),
            "grupos_con_etiqueta_contradictoria": contradictory}


# ------------------------------------------------------------------ clean
def auto_clean(df: pd.DataFrame, target: str | None = None,
               ranges: dict | None = None, drop_cols=(),
               add_missing_flags: bool = True, verbose: bool = True):
    """Limpieza conservadora y reproducible. Devuelve (df_limpio, log).

    `ranges`: {'temperatura': (30, 45), 'edad': (0, 120)} -> fuera de rango a NaN.
    NUNCA imputa: LightGBM/CatBoost manejan NaN y el faltante suele ser señal.
    """
    log = []
    out = df.copy()
    for c in drop_cols:
        if c in out.columns:
            out.drop(columns=c, inplace=True)
            log.append("drop %s" % c)

    for c in out.columns:
        if c == target:
            continue
        s = out[c]
        if is_text(s):
            if numeric_ratio(s) > 0.9:
                out[c] = s.map(to_number)
                log.append("%s: texto -> numerico" % c)
            else:
                out[c] = s.map(norm_text)
                log.append("%s: texto normalizado (lower/sin acentos/NA)" % c)
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            mask = s.isin(list(SENTINELS))
            if mask.any():
                out.loc[mask, c] = np.nan
                log.append("%s: %d centinelas -> NaN" % (c, int(mask.sum())))

    for c, bounds in (ranges or {}).items():
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            m = (out[c] < bounds[0]) | (out[c] > bounds[1])
            if m.any():
                out.loc[m, c] = np.nan
                log.append("%s: %d fuera de [%s,%s] -> NaN"
                           % (c, int(m.sum()), bounds[0], bounds[1]))

    if add_missing_flags:
        for c in [c for c in out.columns if c != target]:
            r = float(out[c].isna().mean())
            if 0.01 < r < 0.98:
                out[c + "__isna"] = out[c].isna().astype(np.int8)
        log.append("flags __isna agregados (el faltante suele ser informativo)")

    if verbose:
        print("\n".join("  - " + x for x in log))
    return out, log


def fix_units(s: pd.Series, factor: float, offset: float = 0.0,
              threshold: float | None = None, above: bool = True) -> pd.Series:
    """Corrige mezcla de unidades. Ej. Fahrenheit -> Celsius:
        fix_units(df.temp, factor=5/9, offset=-32*5/9, threshold=60, above=True)
    """
    v = pd.to_numeric(s, errors="coerce").astype(float).copy()
    if threshold is None:
        return v * factor + offset
    m = v > threshold if above else v < threshold
    v[m] = v[m] * factor + offset
    return v
