"""Checkpoints con escritura atomica, estado del RNG y respaldo en Drive.

Colab free tier se desconecta. Un entrenamiento de 40 minutos que se corta a
los 35 no vale nada si no hay checkpoint, y en una jornada de 9 horas eso es
una porcion enorme del tiempo disponible.

Tres garantias:

  1. **Escritura atomica.** Se escribe a `.tmp` y recien despues `os.replace`.
     Si la sesion muere en medio de la escritura, el checkpoint anterior sigue
     intacto. Un `pickle.dump` directo sobre el archivo bueno lo corrompe.
  2. **Estado del RNG.** Se guarda el estado de `random` y de numpy, asi
     reanudar produce exactamente lo mismo que no haberse cortado.
  3. **Respaldo en Drive.** Si hay Drive montado, el checkpoint se espeja ahi.
     El disco de Colab se borra al reiniciar el entorno; Drive no.
"""
from __future__ import annotations

import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np

DRIVE = Path("/content/drive/MyDrive")


def _drive_disponible() -> bool:
    try:
        return DRIVE.is_dir()
    except OSError:
        return False


class Checkpoint:
    """Guarda y reanuda estado arbitrario.

        ck = Checkpoint("oof_lgb")
        estado = ck.cargar() or {"fold": 0, "oof": np.zeros((n, k))}
        for f in range(estado["fold"], n_folds):
            ...
            estado["fold"] = f + 1
            ck.guardar(estado)
        ck.limpiar()          # al terminar bien
    """

    def __init__(self, nombre: str, carpeta="work/checkpoints",
                 drive: bool = True, verbose: bool = True):
        self.nombre = nombre
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.ruta = self.carpeta / ("%s.pkl" % nombre)
        self.verbose = verbose
        self.drive = None
        if drive and _drive_disponible():
            self.drive = DRIVE / "hackathon_checkpoints"
            self.drive.mkdir(parents=True, exist_ok=True)
            self.drive = self.drive / ("%s.pkl" % nombre)

    # ------------------------------------------------------------ guardar
    def guardar(self, estado: dict, con_rng: bool = True):
        """Escritura atomica: primero .tmp, despues os.replace."""
        payload = {"estado": estado, "ts": time.time()}
        if con_rng:
            payload["rng"] = {"random": random.getstate(),
                              "numpy": np.random.get_state()}
        tmp = self.ruta.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ruta)          # atomico en el mismo volumen
        if self.drive is not None:
            try:
                shutil.copy2(self.ruta, self.drive)
            except OSError as e:
                if self.verbose:
                    print("  [aviso] no se pudo espejar en Drive: %s" % e)
        if self.verbose:
            print("  checkpoint guardado: %s (%.1f KB)"
                  % (self.ruta, self.ruta.stat().st_size / 1024))

    # ------------------------------------------------------------- cargar
    def cargar(self, restaurar_rng: bool = True):
        """Devuelve el estado guardado, o None si no hay. Prueba Drive tambien."""
        for ruta in (self.ruta, self.drive):
            if ruta is None or not ruta.exists():
                continue
            try:
                with open(ruta, "rb") as f:
                    payload = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                if self.verbose:
                    print("  [aviso] checkpoint corrupto en %s, se ignora" % ruta)
                continue
            if restaurar_rng and "rng" in payload:
                random.setstate(payload["rng"]["random"])
                np.random.set_state(payload["rng"]["numpy"])
            if self.verbose:
                edad = (time.time() - payload["ts"]) / 60
                print("  reanudando desde %s (guardado hace %.1f min)" % (ruta, edad))
            return payload["estado"]
        return None

    def existe(self) -> bool:
        return self.ruta.exists() or (self.drive is not None and self.drive.exists())

    def limpiar(self):
        """Borrar al terminar bien, para que la proxima corrida arranque limpia."""
        for ruta in (self.ruta, self.drive):
            if ruta is not None and ruta.exists():
                ruta.unlink()
        if self.verbose:
            print("  checkpoint %s limpiado" % self.nombre)


# --------------------------------------------------- OOF con reanudacion
def oof_con_checkpoint(X, y, Xte, C=None, n_splits=5, seeds=(42, 43, 44),
                       params=None, sample_weight=None, nombre="oof",
                       verbose=True):
    """Igual que `tabular.oof_lgb` pero guarda despues de cada fold.

    Si Colab se cae en el fold 7 de 15, al reejecutar la celda arranca en el 7.
    Devuelve (oof, pte, info).
    """
    from sklearn.model_selection import StratifiedKFold
    from .costs import bayes_decision, mean_cost
    from .tabular import LGB_PARAMS, HAS_LGB

    y = np.asarray(y)
    k = len(np.unique(y))
    tareas = [(s, f) for s in seeds for f in range(n_splits)]

    ck = Checkpoint(nombre, verbose=verbose)
    est = ck.cargar() or {"i": 0, "oof": np.zeros((len(X), k)),
                          "pte": np.zeros((len(Xte), k)), "iters": []}
    if est["i"] >= len(tareas):
        if verbose:
            print("  ya estaba completo")
        return est["oof"], est["pte"], {"n_models": len(est["iters"])}

    if not HAS_LGB:
        raise RuntimeError("oof_con_checkpoint necesita LightGBM; "
                           "sin el usa tabular.oof_lgb, que cae a sklearn")
    import lightgbm as lgb
    p = {**LGB_PARAMS, **(params or {}), "num_class": k}

    for i in range(est["i"], len(tareas)):
        seed, fold = tareas[i]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, va = list(skf.split(X, y))[fold]
        m = lgb.LGBMClassifier(**p, random_state=seed)
        sw = None if sample_weight is None else np.asarray(sample_weight)[tr]
        m.fit(X.iloc[tr], y[tr], sample_weight=sw,
              eval_set=[(X.iloc[va], y[va])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        est["oof"][va] += m.predict_proba(X.iloc[va]) / len(seeds)
        est["pte"] += m.predict_proba(Xte) / len(tareas)
        est["iters"].append(m.best_iteration_ or p["n_estimators"])
        est["i"] = i + 1
        ck.guardar(est)
        if verbose:
            print("  %d/%d  semilla %d fold %d" % (i + 1, len(tareas), seed, fold))

    ck.limpiar()
    info = {"n_models": len(est["iters"]),
            "mean_best_iter": float(np.mean(est["iters"]))}
    if C is not None:
        info["costo_oof"] = mean_cost(y, bayes_decision(est["oof"], C), C)
        if verbose:
            print("costo OOF (minimo costo esperado): %.5f" % info["costo_oof"])
    return est["oof"], est["pte"], info


# ------------------------------------------------------ utilidad Colab
def montar_drive(verbose=True) -> bool:
    """Monta Drive si estamos en Colab. Devuelve si quedo disponible."""
    if _drive_disponible():
        return True
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return _drive_disponible()
    except Exception:
        if verbose:
            print("  sin Drive: los checkpoints quedan solo en el disco local")
        return False
