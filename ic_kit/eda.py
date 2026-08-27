"""EDA prearmado orientado a la METRICA, no a la exactitud.

Un EDA generico produce histogramas bonitos que no cambian ninguna decision.
Este responde las preguntas que en una competencia de costo asimetrico si
cambian lo que hacemos:

  1. Como se reparte el target y cuanto pesa cada clase en el costo maximo.
  2. Que columnas tienen anomalias (delega en `cleaning.audit`).
  3. Que variables separan la clase CARA, que no siempre es la mas frecuente.
  4. Se parecen train y test, columna por columna.
  5. Cual es el piso: que costo da la constante y que da el azar.

Las figuras se guardan con nomenclatura sistematica (`fig01_...png`) para
pegarlas directo en el informe, y hay una celda de descarga en ZIP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import audit, duplicate_report, is_text
from .costs import mean_cost


def _plt():
    import matplotlib
    matplotlib.use("Agg") if not _en_notebook() else None
    import matplotlib.pyplot as plt
    return plt


def _en_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


class EDA:
    """Genera el EDA completo y guarda las figuras numeradas.

        e = EDA(train, target="nivel_urgencia", test=test, C=COSTO, clases=CLASES)
        e.todo()
        e.zip()
    """

    def __init__(self, train: pd.DataFrame, target: str, test=None,
                 C=None, clases=None, id_col=None, figs="figs"):
        self.tr = train
        self.te = test
        self.target = target
        self.C = None if C is None else np.asarray(C, dtype=float)
        self.id_col = id_col
        self.figs = Path(figs)
        self.figs.mkdir(parents=True, exist_ok=True)
        self.n = 0
        y = train[target]
        self.clases = list(clases) if clases is not None else sorted(
            pd.unique(y.dropna()).tolist(), key=str)
        self.y = y.map({c: i for i, c in enumerate(self.clases)}).to_numpy()
        if np.isnan(self.y.astype(float)).any():
            raise ValueError("hay etiquetas del train fuera de `clases`: "
                             "revisa el orden contra la consigna")

    def _guardar(self, fig, nombre: str):
        self.n += 1
        ruta = self.figs / ("fig%02d_%s.png" % (self.n, nombre))
        fig.savefig(ruta, dpi=140, bbox_inches="tight")
        print("  %s" % ruta)
        return ruta

    # ----------------------------------------------------- 1. el target
    def target_y_costo(self):
        plt = _plt()
        cnt = pd.Series(self.y).value_counts().sort_index()
        prop = cnt / cnt.sum()
        print("Distribucion del target:")
        for i, c in enumerate(self.clases):
            print("  %-24s %5d  %5.1f%%" % (c, cnt.get(i, 0), 100 * prop.get(i, 0)))

        if self.C is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([str(c) for c in self.clases], prop.reindex(range(len(self.clases)), fill_value=0))
            ax.set_ylabel("proporcion")
            ax.set_title("Distribucion del target")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            return self._guardar(fig, "distribucion_target")

        # cuanto costo aporta cada clase real si la erraramos siempre al maximo
        peor = self.C.max(axis=1)
        aporte = prop.reindex(range(len(self.clases)), fill_value=0).to_numpy() * peor
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar([str(c) for c in self.clases],
                    prop.reindex(range(len(self.clases)), fill_value=0))
        axes[0].set_title("Frecuencia en el train")
        axes[0].set_ylabel("proporcion")
        axes[1].bar([str(c) for c in self.clases], aporte, color="indianred")
        axes[1].set_title("Costo maximo que aporta cada clase")
        axes[1].set_ylabel("proporcion x peor costo")
        for a in axes:
            a.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        print("\n%-24s %8s %8s %11s" % ("clase", "frec", "peor", "exposicion"))
        for i, c in enumerate(self.clases):
            print("%-24s %7.1f%% %8.0f %11.3f"
                  % (c, 100 * prop.get(i, 0), peor[i], aporte[i]))
        print("\nExposicion = frecuencia x peor costo de esa fila. Frecuencia y")
        print("costo unitario tiran para lados opuestos: la clase mas cara de")
        print("errar (%s, hasta %.0f) casi nunca es la mas frecuente, y es la"
              % (self.clases[int(peor.argmax())], peor.max()))
        print("que decide el resultado.")
        return self._guardar(fig, "target_vs_costo")

    # ---------------------------------------------------- 2. anomalias
    def anomalias(self):
        a = audit(self.tr, target=self.target, df_test=self.te)
        a.to_csv(self.figs.parent / "audit.csv", index=False)
        flag = a[a.alertas != ""]
        print("Columnas con alertas: %d de %d" % (len(flag), len(a)))
        if len(flag):
            cols = [c for c in ["col", "missing_%", "missing_test_%", "n_unique",
                                "alertas"] if c in flag.columns]
            print(flag[cols].to_string(index=False))
        print()
        print(duplicate_report(self.tr, self.target,
                               ignore=(self.id_col,) if self.id_col else ()))
        return a

    # --------------------------------- 3. que separa la clase mas cara
    def separadores(self, top=8):
        """Ranking de variables por cuanto separan la clase de mayor costo.

        Usa AUC one-vs-rest contra la clase cara. Una variable puede ser
        inutil en general y decisiva para la clase que cuesta 10.
        """
        plt = _plt()
        if self.C is None:
            objetivo = int(pd.Series(self.y).value_counts().idxmin())
        else:
            objetivo = int(self.C.max(axis=1).argmax())
        print("Clase objetivo (la mas cara de errar): %s" % self.clases[objetivo])
        from sklearn.metrics import roc_auc_score

        bin_y = (self.y == objetivo).astype(int)
        filas = []
        for c in self.tr.columns:
            if c == self.target or c == self.id_col:
                continue
            s = self.tr[c]
            if is_text(s):
                if s.nunique(dropna=True) > 30:
                    continue
                v = pd.Categorical(s.astype(str)).codes.astype(float)
            else:
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 30 or len(np.unique(v[ok])) < 2:
                continue
            try:
                auc = roc_auc_score(bin_y[ok], v[ok])
            except ValueError:
                continue
            filas.append({"variable": c, "auc": round(auc, 4),
                          "separacion": round(abs(auc - 0.5), 4)})
        r = pd.DataFrame(filas).sort_values("separacion", ascending=False)
        print(r.head(top).to_string(index=False))

        t = r.head(top)[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.5))
        ax.barh(t.variable, t.separacion)
        ax.set_xlabel("|AUC - 0.5| contra la clase %s" % self.clases[objetivo])
        ax.set_title("Variables que separan la clase mas cara")
        fig.tight_layout()
        self._guardar(fig, "separadores_clase_cara")

        alta = r[r.auc > 0.95]
        if len(alta):
            print("\n[!] AUC > 0.95 en solitario: %s" % list(alta.variable))
            print("    Confirmar con traps.leakage_scan antes de usarlas.")
        return r

    # ------------------------------------------------ 4. train vs test
    def train_vs_test(self, top=6):
        if self.te is None:
            print("sin conjunto de test: se omite")
            return None
        plt = _plt()
        a = audit(self.tr, target=self.target, df_test=self.te)
        if "psi_test" not in a.columns:
            return None
        d = a.dropna(subset=["psi_test"]).sort_values("psi_test", ascending=False)
        print("Drift por columna (PSI; > 0.25 es fuerte):")
        print(d[["col", "psi_test"]].head(top).to_string(index=False))

        cols = [c for c in d.col.head(4)
                if c in self.tr.columns and not is_text(self.tr[c])]
        if not cols:
            return d
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, cols):
            for df, et in ((self.tr, "train"), (self.te, "test")):
                v = pd.to_numeric(df[c], errors="coerce").dropna()
                ax.hist(v, bins=30, alpha=0.55, density=True, label=et)
            ax.set_title(c, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Columnas con mayor diferencia entre train y test")
        fig.tight_layout()
        self._guardar(fig, "drift_train_test")
        return d

    # ------------------------------------------------------- 5. el piso
    def piso(self):
        """Que costo dan las estrategias tontas. Si el modelo no las supera,
        el problema no es el modelo."""
        if self.C is None:
            print("sin matriz de costos: se omite")
            return None
        n = len(self.y)
        filas = []
        for i, c in enumerate(self.clases):
            filas.append({"estrategia": "siempre %s" % c,
                          "costo": round(mean_cost(self.y, np.full(n, i), self.C), 4)})
        rng = np.random.default_rng(42)
        filas.append({"estrategia": "azar uniforme",
                      "costo": round(mean_cost(
                          self.y, rng.integers(0, len(self.clases), n), self.C), 4)})
        prop = np.bincount(self.y, minlength=len(self.clases)) / n
        filas.append({"estrategia": "azar segun frecuencia",
                      "costo": round(mean_cost(
                          self.y, rng.choice(len(self.clases), n, p=prop), self.C), 4)})
        r = pd.DataFrame(filas).sort_values("costo")
        print(r.to_string(index=False))
        print("\nPiso a superar: %.4f (%s)" % (r.costo.iloc[0], r.estrategia.iloc[0]))
        return r

    # ------------------------------------------------------------ todo
    def todo(self):
        for titulo, fn in (("1. TARGET Y COSTO", self.target_y_costo),
                           ("2. ANOMALIAS EN LAS COLUMNAS", self.anomalias),
                           ("3. QUE SEPARA LA CLASE MAS CARA", self.separadores),
                           ("4. TRAIN CONTRA TEST", self.train_vs_test),
                           ("5. EL PISO A SUPERAR", self.piso)):
            print("\n" + "=" * 68)
            print(titulo)
            print("=" * 68)
            fn()
        print("\n%d figuras en %s/" % (self.n, self.figs))

    def zip(self, nombre="figuras_eda"):
        import shutil
        ruta = shutil.make_archive(nombre, "zip", str(self.figs))
        print("comprimido: %s" % ruta)
        try:
            from google.colab import files
            files.download(ruta)
        except Exception:
            pass
        return ruta
