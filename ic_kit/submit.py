"""Generacion, validacion y GESTION DEL PRESUPUESTO de envios.

Reglas de los desafios 2026: maximo 10 envios, minimo 5 min entre cada uno,
y cuenta el ULTIMO enviado (no el mejor). Eso hace que el envio final sea una
decision de riesgo: este modulo la vuelve explicita.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MAX_SUBMITS = 10
MIN_GAP_MIN = 5


# ------------------------------------------------------------- generacion
def make_submission(ids, preds, classes=None, path="submit.csv",
                    id_col="id_paciente", target_col="nivel_urgencia") -> pd.DataFrame:
    """Arma el CSV. `classes` mapea indices -> etiquetas de texto si hace falta."""
    p = np.asarray(preds)
    if classes is not None and np.issubdtype(p.dtype, np.integer):
        p = np.asarray([classes[i] for i in p])
    sub = pd.DataFrame({id_col: np.asarray(ids), target_col: p})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print("escrito %s  (%d filas)" % (path, len(sub)))
    return sub


def validate(sub: pd.DataFrame, expected_ids=None, allowed_labels=None,
             id_col=None, target_col=None, train_dist: pd.Series | None = None) -> bool:
    """Chequeos que evitan quemar un envio por una tontera. Corrio SIEMPRE."""
    id_col = id_col or sub.columns[0]
    target_col = target_col or sub.columns[-1]
    ok = True

    def bad(msg):
        nonlocal ok
        ok = False
        print("  [ERROR] " + msg)

    if sub.shape[1] != 2:
        bad("el archivo tiene %d columnas, se esperan 2" % sub.shape[1])
    if sub[id_col].duplicated().any():
        bad("hay %d ids duplicados" % int(sub[id_col].duplicated().sum()))
    if sub[target_col].isna().any():
        bad("hay %d predicciones NaN" % int(sub[target_col].isna().sum()))
    if expected_ids is not None:
        exp = set(pd.Series(expected_ids).astype(str))
        got = set(sub[id_col].astype(str))
        if exp - got:
            bad("faltan %d ids del test" % len(exp - got))
        if got - exp:
            bad("sobran %d ids que no estan en el test" % len(got - exp))
    if allowed_labels is not None:
        extra = set(sub[target_col].astype(str)) - set(map(str, allowed_labels))
        if extra:
            bad("etiquetas no validas: %s" % sorted(extra)[:5])

    dist = sub[target_col].value_counts(normalize=True).sort_index()
    print("  distribucion predicha:")
    for kk, v in dist.items():
        line = "    %-28s %6.2f%%" % (kk, 100 * v)
        if train_dist is not None and kk in train_dist.index:
            ratio = v / max(train_dist[kk], 1e-9)
            line += "   (train %5.2f%%  x%.2f)" % (100 * train_dist[kk], ratio)
            if ratio > 3 or ratio < 0.33:
                line += "  <-- OJO"
        print(line)
    print("  VALIDACION: " + ("OK" if ok else "FALLA"))
    return ok


def diff_vs(path_new: str, path_old: str, id_col=None, target_col=None) -> float:
    """% de filas que cambian respecto del envio anterior.

    <1% -> no gastes un envio. >30% -> cambio arriesgado, justificalo con OOF.
    """
    a = pd.read_csv(path_new)
    b = pd.read_csv(path_old)
    id_col = id_col or a.columns[0]
    target_col = target_col or a.columns[-1]
    m = a.merge(b, on=id_col, suffixes=("_new", "_old"))
    ch = float((m[target_col + "_new"].astype(str) != m[target_col + "_old"].astype(str)).mean())
    print("cambian %.2f%% de las filas vs %s" % (100 * ch, path_old))
    return ch


# --------------------------------------------------------------- registro
class SubmitLog:
    """Diario de envios: presupuesto, cooldown y correlacion CV vs leaderboard."""

    def __init__(self, path="work/submits.json", max_submits=MAX_SUBMITS,
                 min_gap_min=MIN_GAP_MIN, lower_is_better=False):
        self.path = Path(path)
        self.max_submits = max_submits
        self.min_gap_min = min_gap_min
        self.lower_is_better = lower_is_better
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def can_submit(self) -> tuple[bool, str]:
        if len(self.entries) >= self.max_submits:
            return False, "presupuesto agotado (%d/%d)" % (len(self.entries), self.max_submits)
        if self.entries:
            last = datetime.fromisoformat(self.entries[-1]["ts"])
            nxt = last + timedelta(minutes=self.min_gap_min)
            if datetime.now() < nxt:
                return False, "esperar hasta %s" % nxt.strftime("%H:%M:%S")
        return True, "quedan %d envios" % (self.max_submits - len(self.entries))

    def wait_until_ready(self):
        while True:
            ok, msg = self.can_submit()
            if ok or "agotado" in msg:
                print(msg)
                return ok
            print(msg, end="\r")
            time.sleep(5)

    def record(self, file: str, cv: float, notes: str = "", lb: float | None = None):
        h = hashlib.md5(Path(file).read_bytes()).hexdigest()[:8]
        if any(e["md5"] == h for e in self.entries):
            print("  [AVISO] archivo IDENTICO a un envio previo: no gastes el intento.")
        self.entries.append({"n": len(self.entries) + 1, "ts": datetime.now().isoformat(timespec="seconds"),
                             "file": str(file), "md5": h, "cv": cv, "lb": lb, "notes": notes})
        self._save()
        print("registrado envio #%d  cv=%.5f  %s" % (len(self.entries), cv, notes))

    def set_lb(self, n: int, lb: float):
        """Cargá el score real del leaderboard apenas lo veas. Es oro."""
        self.entries[n - 1]["lb"] = lb
        self._save()
        self.report()

    def report(self):
        rows = [e for e in self.entries if e.get("lb") is not None]
        print("\n#   cv        lb        gap      notas")
        for e in self.entries:
            lb = e.get("lb")
            gap = "" if lb is None else "%+.5f" % (lb - e["cv"])
            print("%-3d %-9.5f %-9s %-8s %s" % (e["n"], e["cv"],
                                                "-" if lb is None else "%.5f" % lb,
                                                gap, e.get("notes", "")))
        if len(rows) >= 3:
            cv = np.array([e["cv"] for e in rows])
            lb = np.array([e["lb"] for e in rows])
            print("correlacion CV-LB: r=%.3f  (n=%d)" % (np.corrcoef(cv, lb)[0, 1], len(rows)))
            print("gap medio: %+.5f  desvio: %.5f" % ((lb - cv).mean(), (lb - cv).std()))
            print("Si r<0.5 tu CV esta mal armado: revisa fugas o el esquema de folds.")
        best = None
        if rows:
            best = min(rows, key=lambda e: e["lb"]) if self.lower_is_better \
                else max(rows, key=lambda e: e["lb"])
            print("\nMEJOR LB hasta ahora: envio #%d (%s) -> %.5f" % (best["n"], best["notes"], best["lb"]))
            print("RECORDA: cuenta el ULTIMO envio. Reenvia ese archivo al final.")
        return best
