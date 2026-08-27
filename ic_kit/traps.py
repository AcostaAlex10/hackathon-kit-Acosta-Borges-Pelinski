"""Detector de TRAMPAS en el dataset.

Un desafío bien diseñado se defiende de las soluciones automáticas (AutoML,
notebook copiado, código generado por un LLM sin mirar los datos) metiendo
trampas. Ninguna es magia: todas son detectables ANTES de entrenar, y este
módulo las busca una por una.

  1. adversarial_validation  -> ¿el test viene de otra distribución que el train?
  2. leakage_scan            -> ¿alguna columna sola predice el target casi perfecto?
  3. order_leak              -> ¿el id o el orden de las filas filtra el target?
  4. shortcut_scan_images    -> ¿alcanza el fondo/color para clasificar? (Clever Hans)
  5. label_noise_scan        -> ¿qué filas están probablemente mal etiquetadas?
  6. fit_prior_honest        -> ¿mi ajuste de umbrales es real o se sobreajustó al OOF?
  7. class_prior_shift       -> ¿el test tiene otra proporción de clases que el train?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .costs import bayes_decision, mean_cost
from .tabular import fit_prior


def _model(seed=42):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08,
                                  num_leaves=31, verbosity=-1, random_state=seed)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)


def _numeric(df):
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.Categorical(out[c].astype(str)).codes.astype(float)
    return out


# ----------------------------------------------------- 1. train vs test
def adversarial_validation(X_train, X_test, n_splits=5, seed=42, verbose=True):
    """Entrena un clasificador train-vs-test. AUC ~0.5 = misma distribución.

    Interpretación:
      AUC < 0.55   -> train y test son intercambiables. Tu CV es confiable.
      0.55 - 0.75  -> hay shift. Mirá qué features lo causan y considerá sacarlas.
      > 0.75       -> shift fuerte. El test NO es una muestra del train:
                      probablemente es la trampa central del desafío.

    Devuelve (auc, importancias, p_es_test) donde p_es_test sirve para pesar
    las filas de train que más se parecen al test.
    """
    Xa = _numeric(pd.concat([X_train, X_test], ignore_index=True))
    ya = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(_model(seed), Xa, ya, cv=skf, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(ya, p))

    m = _model(seed).fit(Xa, ya)
    imp = pd.DataFrame({"feature": Xa.columns,
                        "importancia": getattr(m, "feature_importances_",
                                               np.zeros(Xa.shape[1]))}
                       ).sort_values("importancia", ascending=False)
    if verbose:
        veredicto = ("distribuciones equivalentes, CV confiable" if auc < 0.55 else
                     "SHIFT MODERADO: revisá las features de abajo" if auc < 0.75 else
                     "SHIFT FUERTE: el test es distinto al train A PROPOSITO")
        print("adversarial validation  AUC = %.4f  -> %s" % (auc, veredicto))
        if auc >= 0.55:
            print(imp.head(8).to_string(index=False))
            print("  (una feature que domina esta lista suele ser un id, una fecha")
            print("   o un artefacto de generación: probá sacarla y re-medir)")
    return auc, imp, p[:len(X_train)]


# --------------------------------------------------------- 2. fugas
def leakage_scan(X, y, n_splits=3, seed=42, verbose=True) -> pd.DataFrame:
    """AUC/accuracy de cada columna POR SI SOLA contra el target.

    Una feature única que predice casi perfecto casi nunca es un regalo: suele
    ser una fuga (se calculó DESPUES de conocer la etiqueta). Si está en el
    train y también en el test, úsala. Si el test la tiene vacía o distinta,
    es la trampa: te lleva a un CV altísimo y un leaderboard pésimo.
    Cruzá siempre este reporte con `adversarial_validation`.
    """
    y = np.asarray(y)
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in X.columns:
        Xc = _numeric(X[[c]])
        try:
            p = cross_val_predict(_model(seed), Xc, y, cv=skf, method="predict_proba")
            acc = float((p.argmax(1) == y).mean())
            auc = float(roc_auc_score(y, p, multi_class="ovr")) if len(np.unique(y)) > 2 \
                else float(roc_auc_score(y, p[:, 1]))
        except Exception:
            acc, auc = np.nan, np.nan
        rows.append({"feature": c, "acc_sola": round(acc, 4), "auc_sola": round(auc, 4)})
    out = pd.DataFrame(rows).sort_values("auc_sola", ascending=False)
    base = float(pd.Series(y).value_counts(normalize=True).max())
    out["vs_mayoritaria"] = (out.acc_sola - base).round(4)
    if verbose:
        print("accuracy de la clase mayoritaria = %.4f" % base)
        print(out.head(10).to_string(index=False))
        sosp = out[out.auc_sola > 0.95]
        if len(sosp):
            print("\n[!] SOSPECHA DE FUGA en: %s" % list(sosp.feature))
            print("    verificá que esas columnas estén igual de pobladas en el test")
    return out


def order_leak(df, y, id_col=None, verbose=True) -> dict:
    """¿El orden de las filas o el id filtran el target?

    Pasa cuando el dataset se armó concatenando bloques por clase y después no
    se mezcló. Es una fuga que funciona en el train y NUNCA en el test.
    """
    y = np.asarray(pd.Series(y).astype("category").cat.codes)
    idx = np.arange(len(y))
    res = {"corr_indice": float(np.corrcoef(idx, y)[0, 1])}
    if id_col is not None and id_col in df.columns:
        v = pd.to_numeric(df[id_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        res["corr_id"] = float(np.corrcoef(v[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
    # rachas: si las clases vienen en bloques, hay muchas menos rachas que al azar
    cambios = int((np.diff(y) != 0).sum())
    esperado = len(y) * (1 - (pd.Series(y).value_counts(normalize=True) ** 2).sum())
    res["cambios_de_clase"] = cambios
    res["cambios_esperados_si_mezclado"] = round(float(esperado), 1)
    res["ordenado_por_clase"] = cambios < 0.5 * esperado
    if verbose:
        print("order leak:", res)
        if res["ordenado_por_clase"]:
            print("  [!] las filas vienen agrupadas por clase: NO uses el índice")
            print("      ni ninguna feature derivada del orden como predictor")
        if abs(res.get("corr_id") or 0) > 0.2:
            print("  [!] el id correlaciona con el target (r=%.3f)" % res["corr_id"])
            print("      puede ser fuga real o un artefacto: mirá si el test sigue el patrón")
    return res


# --------------------------------------------- 3. atajos en imágenes
def shortcut_scan_images(paths, y, size=16, verbose=True) -> float:
    """¿Se puede clasificar mirando SOLO color y textura gruesa? (Clever Hans)

    Reduce cada imagen a `size x size` y entrena sobre esos píxeles. Si la
    accuracy es alta, las clases se distinguen por el fondo o el tono general,
    no por el objeto. Eso significa dos cosas:
      - tu CV va a ser optimista
      - si el test tiene otros fondos, te desplomás
    Antídoto: augmentation agresiva de color y crops, y validar por grupo
    (una cámara / un sitio por fold, si podés inferirlo).
    """
    from PIL import Image
    feats = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        feats.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
    Xf = pd.DataFrame(np.array(feats))
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    p = cross_val_predict(_model(), Xf, y, cv=skf, method="predict_proba")
    acc = float((p.argmax(1) == y).mean())
    base = float(pd.Series(y).value_counts(normalize=True).max())
    if verbose:
        print("shortcut scan (imágenes de %dx%d): acc = %.4f  vs mayoritaria %.4f"
              % (size, size, acc, base))
        if acc > base + 0.35:
            print("  [!] ATAJO FUERTE: el color/fondo ya separa las clases.")
            print("      Reforzá ColorJitter, RandomResizedCrop y grayscale aleatorio.")
    return acc


# ------------------------------------------------ 4. etiquetas ruidosas
def label_noise_scan(oof_proba, y, C, top=40, verbose=True) -> pd.DataFrame:
    """Filas que el modelo cree mal etiquetadas, ordenadas por daño al costo."""
    y = np.asarray(y)
    p_asignada = np.asarray(oof_proba)[np.arange(len(y)), y]
    pred = bayes_decision(oof_proba, C)
    costo = C[y, pred]
    df = pd.DataFrame({"fila": np.arange(len(y)), "clase_asignada": y,
                       "clase_modelo": pred, "p_asignada": p_asignada,
                       "costo": costo})
    df["sospecha"] = (1 - df.p_asignada) * (1 + df.costo)
    df = df.sort_values("sospecha", ascending=False).head(top)
    if verbose:
        frac = float((p_asignada < 0.05).mean())
        print("filas con p(clase asignada) < 0.05: %.2f%%" % (100 * frac))
        if frac > 0.05:
            print("  [!] mucho ruido de etiquetas o clases genuinamente ambiguas.")
            print("      Probá label_smoothing alto y NO persigas accuracy perfecta.")
    return df


# ------------------------------------------- 5. honestidad del ajuste
def fit_prior_honest(oof_proba, y, C, n_splits=5, seed=42, verbose=True) -> dict:
    """Mide cuánto de la ganancia de `fit_prior` es real y cuánto sobreajuste.

    Ajusta los pesos en 4/5 del OOF y los evalúa en el quinto. Si la ganancia
    honesta es mucho menor que la in-sample, tus umbrales están sobreajustados
    y el leaderboard te lo va a cobrar.
    """
    oof_proba = np.asarray(oof_proba)
    y = np.asarray(y)
    w_full = fit_prior(oof_proba, y, C, verbose=False)
    c_base = mean_cost(y, bayes_decision(oof_proba, C), C)
    c_in = mean_cost(y, bayes_decision(oof_proba * w_full, C), C)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tr, va in skf.split(oof_proba, y):
        w = fit_prior(oof_proba[tr], y[tr], C, verbose=False)
        pred[va] = bayes_decision(oof_proba[va] * w, C)
    c_out = mean_cost(y, pred, C)

    res = {"costo_sin_ajuste": round(c_base, 5),
           "costo_ajuste_in_sample": round(c_in, 5),
           "costo_ajuste_honesto": round(c_out, 5),
           "ganancia_real": round(c_base - c_out, 5),
           "sobreajuste": round(c_out - c_in, 5),
           "w": np.round(w_full, 3).tolist()}
    if verbose:
        print("fit_prior: sin ajuste %.5f | in-sample %.5f | honesto %.5f"
              % (c_base, c_in, c_out))
        if c_out >= c_base:
            print("  [!] el ajuste NO generaliza: usá bayes_decision sin pesos.")
        elif res["sobreajuste"] > 0.5 * abs(res["ganancia_real"]):
            print("  [!] la mitad de la ganancia es sobreajuste: usá menos rondas.")
        else:
            print("  ajuste sano: la ganancia sobrevive fuera de muestra.")
    return res


# ------------------------------------------------ 6. prior shift de clases
def class_prior_shift(y_train, proba_test, verbose=True) -> dict:
    """Compara la proporción de clases del train con la que el modelo ve en test.

    Si difieren mucho, el desafío tiene prior shift (muy común como trampa: te
    dan un train balanceado y un test que no lo está, o al revés). La corrección
    barata es multiplicar las probabilidades por (prior_test / prior_train),
    que es exactamente lo que aproxima `fit_prior`.
    """
    y_train = np.asarray(y_train)
    k = np.asarray(proba_test).shape[1]
    p_tr = np.bincount(y_train, minlength=k) / len(y_train)
    p_te = np.asarray(proba_test).mean(axis=0)
    ratio = p_te / np.maximum(p_tr, 1e-9)
    res = {"prior_train": np.round(p_tr, 4).tolist(),
           "prior_test_estimado": np.round(p_te, 4).tolist(),
           "ratio": np.round(ratio, 3).tolist()}
    if verbose:
        print("prior train        :", res["prior_train"])
        print("prior test (estim.):", res["prior_test_estimado"])
        print("ratio              :", res["ratio"])
        if np.any((ratio > 1.5) | (ratio < 0.67)):
            print("  [!] PRIOR SHIFT detectado. `fit_prior` debería corregir parte;")
            print("      verificá con fit_prior_honest que la corrección generalice.")
    return res


# ------------------------------------------------------------ todo junto
def run_all(X_train, y, X_test, C=None, oof_proba=None, df_raw=None,
            id_col=None):
    """Corre la batería completa. Esto va en la primera hora del desafío."""
    print("=" * 70)
    print("1. ADVERSARIAL VALIDATION (train vs test)")
    print("=" * 70)
    auc, imp, _ = adversarial_validation(X_train, X_test)

    print("\n" + "=" * 70)
    print("2. FUGAS: poder predictivo de cada feature aislada")
    print("=" * 70)
    leak = leakage_scan(X_train, y)

    print("\n" + "=" * 70)
    print("3. FUGA POR ORDEN / ID")
    print("=" * 70)
    ol = order_leak(df_raw if df_raw is not None else X_train, y, id_col)

    out = {"adv_auc": auc, "adv_imp": imp, "leakage": leak, "order": ol}
    if oof_proba is not None and C is not None:
        print("\n" + "=" * 70)
        print("4. HONESTIDAD DEL AJUSTE DE DECISIÓN")
        print("=" * 70)
        out["fit_prior"] = fit_prior_honest(oof_proba, y, C)
    return out


# ------------------------------- 7. estimar el prior del test SIN etiquetas
def estimate_test_prior(proba_test, prior_train, n_iter=200, tol=1e-7,
                        verbose=True):
    """Algoritmo EM de Saerens-Latinne-Decaestecker (2002).

    Estima la proporción de clases del TEST usando sólo las probabilidades que
    el modelo predice sobre el test — nunca sus etiquetas. Es completamente
    legítimo: no mira el target, sólo la distribución de las features.

    ADVERTENCIA MEDIDA, NO TEORICA: en pruebas con datos sinteticos el EM
    COLAPSO la clase rara a prior 0 y empeoró el costo de 1.477 a 1.657. Le
    pasa cuando el clasificador es débil o está mal calibrado. NO lo apliques
    a ciegas: corré antes `validate_prior_em` sobre tu propio OOF, que te dice
    si en TU dataset el EM recupera un prior conocido.

    Devuelve (prior_test_estimado, proba_corregida).
    """
    p = np.asarray(proba_test, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    pr_tr = np.asarray(prior_train, dtype=float)
    pr = pr_tr.copy()
    for _ in range(n_iter):
        w = p * (pr / pr_tr)
        w = w / w.sum(axis=1, keepdims=True)
        nuevo = w.mean(axis=0)
        if np.abs(nuevo - pr).max() < tol:
            pr = nuevo
            break
        pr = nuevo
    w = p * (pr / pr_tr)
    w = w / w.sum(axis=1, keepdims=True)
    if verbose:
        print("prior train          :", np.round(pr_tr, 4))
        print("prior test estimado  :", np.round(pr, 4))
        print("factor de correccion :", np.round(pr / pr_tr, 3))
        if np.abs(pr / pr_tr - 1).max() > 0.3:
            print("  -> PRIOR SHIFT confirmado: usa la proba corregida para decidir")
    return pr, w


def validate_prior_em(oof_proba, y, shifts=((0.5, 2.0), (2.0, 0.5), (3.0, 0.4),
                                            (0.4, 3.0)), repeticiones=5, seed=42,
                      verbose=True) -> dict:
    """¿El EM de prior funciona en MI dataset? Se comprueba, no se supone.

    Fabrica un prior conocido remuestreando el OOF, corre `estimate_test_prior`
    y mide cuánto se acerca. Repite con varias semillas porque un solo
    remuestreo da un veredicto ruidoso: cerca del umbral, dos corridas del
    mismo dataset pueden dar respuestas opuestas.

    El veredicto sale de la mediana de la reducción relativa del error:
      >= 0.40  SIRVE           -> aplicar `estimate_test_prior`
      0.15-0.40 MARGINAL       -> sólo si `class_prior_shift` mostró shift real
      < 0.15   NO SIRVE        -> `bayes_decision` sin corregir
    """
    oof_proba = np.asarray(oof_proba, dtype=float)
    oof_proba = oof_proba / oof_proba.sum(axis=1, keepdims=True)
    y = np.asarray(y)
    k = oof_proba.shape[1]
    pr_tr = np.bincount(y, minlength=k) / len(y)
    idx_por_clase = [np.where(y == c)[0] for c in range(k)]

    filas = []
    for r in range(repeticiones):
        rng = np.random.default_rng(seed + r)
        for factores in shifts:
            f = np.ones(k)
            f[0], f[-1] = factores
            objetivo = pr_tr * f
            objetivo = objetivo / objetivo.sum()
            idx = np.concatenate([
                rng.choice(idx_por_clase[c], size=max(1, int(objetivo[c] * len(y))),
                           replace=True) for c in range(k)])
            pr_real = np.bincount(y[idx], minlength=k) / len(idx)
            pr_em, _ = estimate_test_prior(oof_proba[idx], pr_tr, verbose=False)
            err_sin = float(np.abs(pr_tr - pr_real).sum())
            err_em = float(np.abs(pr_em - pr_real).sum())
            filas.append({"shift": str(factores), "rep": r,
                          "err_sin_corregir": err_sin, "err_em": err_em,
                          "reduccion": (err_sin - err_em) / max(err_sin, 1e-9)})
    df = pd.DataFrame(filas)
    resumen = (df.groupby("shift")[["err_sin_corregir", "err_em", "reduccion"]]
               .median().round(4).reset_index())
    red = float(df.reduccion.median())
    veredicto = ("SIRVE" if red >= 0.40 else
                 "MARGINAL" if red >= 0.15 else "NO SIRVE")
    if verbose:
        print(resumen.to_string(index=False))
        print("reduccion mediana del error de prior: %.1f%% (%d escenarios)"
              % (100 * red, len(df)))
        print("VEREDICTO: %s" % veredicto)
        if veredicto == "SIRVE":
            print("  -> aplicar estimate_test_prior sobre las probabilidades de test")
        elif veredicto == "MARGINAL":
            print("  -> aplicarlo SOLO si class_prior_shift mostro un shift real,")
            print("     y verificar que el envio no cambie mas de lo razonable")
        else:
            print("  -> usar bayes_decision sin corregir el prior")
    return {"tabla": resumen, "reduccion": red, "veredicto": veredicto,
            "sirve": veredicto == "SIRVE"}
