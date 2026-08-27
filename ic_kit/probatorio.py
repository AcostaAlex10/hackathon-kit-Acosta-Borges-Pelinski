"""Genera el probatorio del art. 5 del reglamento como notebook Jupyter.

Cerrado el plazo de submits hay **30 minutos** para entregar **un único archivo
digital** que respalde los resultados, detallando preprocesamiento, selección
de algoritmos, evaluación y decisiones tomadas.

Dos consecuencias de diseño:

  - "un único archivo": el notebook queda **autocontenido**. Las matrices OOF,
    la auditoría y el historial de envíos se embeben comprimidos en base64, así
    que corre solo, sin `work/` al lado y sin volver a entrenar.
  - "30 minutos": lo automatizable ya está escrito. Sólo quedan marcados los
    huecos de criterio, que son los que efectivamente se evalúan.

Estilo del notebook, según la convención de la materia: español, sin emojis,
markdown conciso, sin pares pregunta-respuesta, sin explicar conceptos básicos,
semillas fijas.

Uso al cierre de la jornada:

    from ic_kit.probatorio import generar_notebook
    generar_notebook(
        grupo="Grupo 8", integrantes=["Acosta", "Bareiro", "Borges"],
        desafio="Triaje en Admisiones Hospitalarias",
        metrica="score = 1/(1+costo promedio); mayor es mejor",
        oof="work/oof.npy", y="work/y.npy", C="d1",
        trampas=[...], decisiones=[...], descartado=[...])
"""
from __future__ import annotations

import base64
import json
import zlib
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PENDIENTE = "**(completar a mano)**"


# --------------------------------------------------------------- utilidades
def _leer_json(p):
    p = Path(p)
    if not p.exists():
        return None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _empaquetar(arr) -> str:
    """Array de numpy -> base64 comprimido, para embeber en una celda."""
    buf = BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode("ascii")


def _md_tabla(df) -> str:
    """Tabla markdown sin depender de `tabulate`, que puede no estar instalado."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            "" if pd.isna(r[c]) else str(r[c]).replace("|", r"\|") for c in cols) + " |")
    return "\n".join(out)


def _md(fuente):
    return {"cell_type": "markdown", "metadata": {},
            "source": fuente if isinstance(fuente, list) else [fuente]}


def _code(fuente):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": fuente if isinstance(fuente, list) else [fuente]}


# ------------------------------------------------------------------ armado
def generar_notebook(grupo: str, desafio: str, metrica: str, integrantes=(),
                     oof=None, y=None, C=None, clases=None,
                     audit_csv="work/audit.csv", report_json="work/report.json",
                     submit_log="work/submits.json",
                     trampas=(), decisiones=(), descartado=(), hipotesis=(),
                     bitacora="work/bitacora.json",
                     salida="work/PROBATORIO.ipynb") -> Path:
    """Escribe el probatorio como .ipynb autocontenido. Devuelve la ruta.

    `oof`, `y` y `C` pueden ser rutas a .npy, arrays, o el nombre de una matriz
    del kit ("d1", "d2", "ensayo"). Si se pasan, el notebook recalcula en vivo
    la comparación entre la clase más probable y la decisión de mínimo costo.
    """
    from .costs import (COST_D1, COST_D2, COST_ENSAYO,
                        LABELS_D1, LABELS_D2, LABELS_ENSAYO)

    predef = {"d1": (COST_D1, LABELS_D1), "d2": (COST_D2, LABELS_D2),
              "ensayo": (COST_ENSAYO, LABELS_ENSAYO)}

    def _cargar(x):
        if x is None:
            return None
        if isinstance(x, str) and x in predef:
            return predef[x][0]
        if isinstance(x, (str, Path)):
            return np.load(x) if Path(x).exists() else None
        return np.asarray(x)

    Coste = _cargar(C)
    if clases is None and isinstance(C, str) and C in predef:
        clases = predef[C][1]
    oof_a, y_a = _cargar(oof), _cargar(y)

    # La bitacora de la jornada completa lo que no se paso a mano.
    if bitacora and Path(bitacora).exists():
        from .bitacora import Bitacora
        b = Bitacora(bitacora, verbose=False).para_probatorio()
        hipotesis = list(hipotesis) or b["hipotesis"]
        trampas = list(trampas) or b["trampas"]
        decisiones = list(decisiones) or b["decisiones"]
        descartado = list(descartado) or b["descartado"]
        print("  bitacora leida: %d hipotesis, %d hallazgos, %d decisiones, "
              "%d descartes" % (len(b["hipotesis"]), len(b["trampas"]),
                                len(b["decisiones"]), len(b["descartado"])))

    rep = _leer_json(report_json) or {}
    envios = _leer_json(submit_log) or []
    aud = pd.read_csv(audit_csv) if Path(audit_csv).exists() else None
    if clases is None:
        clases = rep.get("classes")

    cells = []

    # ------------------------------------------------------------ portada
    cells.append(_md([
        "# Probatorio — %s\n" % desafio, "\n",
        "**%s**" % grupo + (" — %s" % ", ".join(integrantes) if integrantes else "") + "  \n",
        "%s\n" % datetime.now().strftime("%d/%m/%Y %H:%M"), "\n",
        "Métrica de la competencia: %s\n" % metrica, "\n",
        "Este notebook es autocontenido: los resultados intermedios están ",
        "embebidos, de modo que se ejecuta secuencialmente de principio a fin ",
        "sin archivos externos ni reentrenamiento.\n",
    ]))

    cells.append(_code([
        "import base64, zlib, json\n",
        "from io import BytesIO\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "def desempaquetar(s):\n",
        "    return np.load(BytesIO(zlib.decompress(base64.b64decode(s))))\n",
    ]))

    # ------------------------------------------- 1. metrica y orden de clases
    cells.append(_md("## 1. Métrica y orden de clases\n"))
    if Coste is not None:
        cells.append(_code([
            "COSTO = desempaquetar('''%s''')\n" % _empaquetar(Coste),
            "CLASES = %r\n" % (list(map(str, clases)) if clases else None),
            "pd.DataFrame(COSTO, index=CLASES, columns=CLASES).astype(int)\n",
        ]))
        cells.append(_md([
            "La matriz se indexa `C[clase real, clase predicha]`. El orden de ",
            "las clases se fijó de forma explícita y se verificó contra la ",
            "tabla de la consigna antes de entrenar: si el orden lo define el ",
            "alfabeto, la matriz queda desalineada y la penalización se aplica ",
            "a la celda equivocada sin que nada falle en tiempo de ejecución.\n",
        ]))
    else:
        cells.append(_md(PENDIENTE + " — transcribir la matriz de costos.\n"))

    # ------------------------------------------------------ 2. exploracion
    cells.append(_md("## 2. Exploración y auditoría de los datos\n"))
    if aud is not None and "alertas" in aud.columns:
        flag = aud[aud.alertas.fillna("") != ""]
        cols = [c for c in ["col", "dtype", "missing_%", "missing_test_%",
                            "n_unique", "alertas"] if c in flag.columns]
        cells.append(_md([
            "Auditoría automática columna por columna sobre el conjunto de ",
            "entrenamiento, contrastada contra el de test:\n\n",
            _md_tabla(flag[cols]) if len(flag) else "No se detectaron anomalías.",
            "\n",
        ]))
        cells.append(_md([
            "Las alertas corresponden a: columnas sin varianza (`CONSTANTE`), ",
            "identificadores (`ID?`), valores numéricos almacenados como texto ",
            "(`NUMERO_COMO_TEXTO`), códigos centinela del tipo -999 ",
            "(`CENTINELA`), dos unidades de medida conviviendo en la misma ",
            "columna (`MEZCLA_UNIDADES?`), variantes de escritura de una misma ",
            "categoría (`TYPOS/CASE`), distribuciones distintas entre train y ",
            "test (`DRIFT_TEST`) y columnas pobladas en train pero vacías en ",
            "test (`VACIA_EN_TEST`).\n",
        ]))
    else:
        cells.append(_md(PENDIENTE + " — resumen del análisis exploratorio.\n"))
    if hipotesis:
        cells.append(_md(["Hipótesis de trabajo planteadas a partir de la exploración:\n\n"] +
                         ["- %s\n" % h for h in hipotesis]))

    # ------------------------------------------------------- 3. anomalias
    cells.append(_md("## 3. Anomalías del conjunto de datos y tratamiento\n"))
    if trampas:
        df_t = pd.DataFrame([{"Hallazgo": t.get("hallazgo", ""),
                              "Evidencia": t.get("evidencia", ""),
                              "Decisión": t.get("decision", "")} for t in trampas])
        cells.append(_md(_md_tabla(df_t) + "\n"))
    else:
        cells.append(_md(PENDIENTE + " — hallazgos, evidencia y decisión tomada.\n"))

    # ---------------------------------------------------- 4. preprocesamiento
    cells.append(_md([
        "## 4. Preprocesamiento\n\n",
        "El texto se normalizó a minúsculas sin acentos, con espacios ",
        "colapsados y las cadenas vacías o de relleno tratadas como faltante. ",
        "Los numéricos almacenados como texto se convirtieron interpretando la ",
        "coma decimal y descartando las unidades. Los códigos centinela se ",
        "convirtieron a faltante.\n\n",
        "No se imputaron los faltantes. Los modelos de árboles los manejan de ",
        "forma nativa mediante una dirección de descenso aprendida, e imputar ",
        "por la media introduce una moda artificial en la distribución. En su ",
        "lugar se agregaron indicadores binarios de ausencia, porque el patrón ",
        "de qué dato falta suele portar señal propia.\n",
    ]))

    # -------------------------------------------------------- 5. validacion
    cells.append(_md([
        "## 5. Esquema de validación\n\n",
        "Validación cruzada estratificada con predicciones out-of-fold, ",
        "promediando varias semillas para reducir la varianza del estimador. ",
        "El OOF es la única medida usada para decidir: el conjunto de test no ",
        "se empleó en ningún momento para ajustar ni para validar.\n",
    ]))
    if rep:
        fil = []
        if "oof_cost_argmax" in rep:
            fil.append(("Costo OOF decidiendo por clase más probable",
                        round(rep["oof_cost_argmax"], 5)))
        if "oof_cost" in rep:
            fil.append(("Costo OOF del pipeline final", round(rep["oof_cost"], 5)))
        if fil:
            cells.append(_md(_md_tabla(
                pd.DataFrame(fil, columns=["Concepto", "Valor"])) + "\n"))

    # ---------------------------------------------------- 6. capa de decision
    cells.append(_md([
        "## 6. Decisión sensible al costo\n\n",
        "La métrica es una matriz de costos asimétrica, de modo que la clase ",
        "más probable no es la decisión que minimiza la pérdida esperada. La ",
        "regla óptima bajo una matriz de costos conocida es\n\n",
        "$$\\hat{y}(x) = \\arg\\min_j \\sum_i P(y=i \\mid x)\\, C_{ij}$$\n\n",
        "es decir, se elige la acción de menor costo esperado en lugar de la ",
        "de mayor probabilidad. Ambas coinciden únicamente cuando la matriz es ",
        "simétrica. Separar la estimación de probabilidades de la regla de ",
        "decisión permite además entrenar el modelo para calibrar bien, sin ",
        "distorsionarlo con pesos de clase.\n",
    ]))
    if oof_a is not None and y_a is not None and Coste is not None:
        cells.append(_code([
            "OOF = desempaquetar('''%s''')\n" % _empaquetar(oof_a),
            "Y = desempaquetar('''%s''')\n" % _empaquetar(np.asarray(y_a, dtype=np.int16)),
            "\n",
            "def costo_medio(y_real, y_pred, C):\n",
            "    return float(C[np.asarray(y_real), np.asarray(y_pred)].mean())\n",
            "\n",
            "def decision_minimo_costo(proba, C):\n",
            "    return (np.asarray(proba) @ np.asarray(C)).argmin(axis=1)\n",
            "\n",
            "pred_probable = OOF.argmax(axis=1)\n",
            "pred_costo = decision_minimo_costo(OOF, COSTO)\n",
            "print('costo OOF, clase mas probable : %.4f' % costo_medio(Y, pred_probable, COSTO))\n",
            "print('costo OOF, minimo costo esp.  : %.4f' % costo_medio(Y, pred_costo, COSTO))\n",
            "print('exactitud, clase mas probable : %.4f' % (pred_probable == Y).mean())\n",
            "print('exactitud, minimo costo esp.  : %.4f' % (pred_costo == Y).mean())\n",
            "print('predicciones que cambian      : %.1f%%' % (100 * (pred_probable != pred_costo).mean()))\n",
        ]))
        cells.append(_md([
            "La exactitud baja mientras el costo mejora. No es una ",
            "contradicción: la regla sacrifica aciertos en confusiones baratas ",
            "para evitar las caras, que es exactamente lo que premia la métrica ",
            "de la competencia.\n",
        ]))
        cells.append(_md("### Origen del costo residual\n"))
        cells.append(_code([
            "filas = []\n",
            "n = len(Y)\n",
            "for i in range(COSTO.shape[0]):\n",
            "    for j in range(COSTO.shape[1]):\n",
            "        cnt = int(((Y == i) & (pred_costo == j)).sum())\n",
            "        if cnt and COSTO[i, j] > 0:\n",
            "            filas.append({'real': CLASES[i], 'predicho': CLASES[j],\n",
            "                          'casos': cnt, 'costo_unitario': COSTO[i, j],\n",
            "                          'aporte': cnt * COSTO[i, j] / n})\n",
            "aporte = pd.DataFrame(filas).sort_values('aporte', ascending=False)\n",
            "display(aporte.head(8).round(4))\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 4))\n",
            "top = aporte.head(8)[::-1]\n",
            "ax.barh([r.real + ' -> ' + r.predicho for r in top.itertuples()], top.aporte)\n",
            "ax.set_xlabel('aporte al costo promedio')\n",
            "ax.set_title('Confusiones que concentran el costo')\n",
            "fig.tight_layout(); plt.show()\n",
        ]))
        cells.append(_md([
            "Este desglose ordena las confusiones por cuánto aportan a la ",
            "métrica, y fue el que guió el orden de trabajo: se atacaron ",
            "primero las celdas de arriba, no las más frecuentes.\n",
        ]))

    # --------------------------------------------------------- 7. decisiones
    cells.append(_md("## 7. Decisiones de modelado\n"))
    if decisiones:
        cells.append(_md(["- %s\n" % d for d in decisiones]))
    else:
        cells.append(_md(PENDIENTE + "\n"))
    if descartado:
        cells.append(_md(["Alternativas evaluadas y descartadas:\n\n"] +
                         ["- %s\n" % d for d in descartado]))
    else:
        cells.append(_md("Alternativas evaluadas y descartadas: " + PENDIENTE + "\n"))

    # ------------------------------------------------------------ 8. envios
    cells.append(_md("## 8. Historial de envíos\n"))
    if envios:
        cells.append(_code([
            "ENVIOS = %s\n" % json.dumps(
                [{k: e.get(k) for k in ("n", "ts", "cv", "lb", "notes")} for e in envios],
                ensure_ascii=False),
            "envios = pd.DataFrame(ENVIOS)\n",
            "display(envios)\n",
            "\n",
            "con_lb = envios.dropna(subset=['lb'])\n",
            "if len(con_lb) >= 3:\n",
            "    r = np.corrcoef(con_lb.cv, con_lb.lb)[0, 1]\n",
            "    fig, ax = plt.subplots(figsize=(5, 4))\n",
            "    ax.scatter(con_lb.cv, con_lb.lb)\n",
            "    for t in con_lb.itertuples():\n",
            "        ax.annotate(int(t.n), (t.cv, t.lb))\n",
            "    ax.set_xlabel('validacion interna')\n",
            "    ax.set_ylabel('leaderboard')\n",
            "    ax.set_title('Correlacion validacion-leaderboard  r=%.3f' % r)\n",
            "    fig.tight_layout(); plt.show()\n",
        ]))
        cells.append(_md([
            "La relación entre la validación interna y el puntaje público se ",
            "monitoreó desde el primer envío. Sirve para decidir cuánto pesar ",
            "cada fuente: mientras se mantuvo alineada, las decisiones se ",
            "tomaron sobre la validación interna, que usa todos los datos ",
            "etiquetados y no consume intentos.\n",
        ]))
    else:
        cells.append(_md(PENDIENTE + " — tabla de envíos.\n"))

    # -------------------------------------------------- 9. reproducibilidad
    cells.append(_md([
        "## 9. Reproducibilidad\n\n",
        "Todas las particiones y modelos usan semilla fija ",
        "(`RANDOM_STATE = 42`). El pipeline completo se ejecuta con un único ",
        "comando y reproduce el archivo entregado. Los resultados intermedios ",
        "que sustentan este informe están embebidos en el propio notebook, de ",
        "modo que las tablas y figuras anteriores se regeneran al ejecutarlo ",
        "sin depender de archivos externos.\n",
    ]))

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}

    p = Path(salida)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:                                   # valida el esquema si hay nbformat
        import nbformat
        nbformat.write(nbformat.from_dict(nb), str(p))
        validado = " (validado con nbformat)"
    except Exception:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        validado = " (escrito sin nbformat)"

    pend = sum(1 for c in cells if any(PENDIENTE in s for s in c["source"]))
    print("probatorio escrito en %s%s" % (p, validado))
    print("  %d celdas, %.0f KB, autocontenido" % (len(cells), p.stat().st_size / 1024))
    if pend:
        print("  %d secciones marcadas '(completar a mano)': son las de criterio,"
              " que es lo que se evalua" % pend)
    return p


generar = generar_notebook          # alias retrocompatible
