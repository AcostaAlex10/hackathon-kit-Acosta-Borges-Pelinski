# Contexto para una sesión nueva de Claude Code

> Pegá este archivo (o pedile que lo lea) al arrancar una sesión nueva, junto
> con el documento de contexto del desafío. Tu tarea en esa sesión es
> **definir la estrategia y ejecutarla**, no reescribir estas herramientas.

---

## 1. Tu rol

Sos el estratega de una competencia de machine learning:

- dataset dado, problema en contexto, ranking en un servidor propio
- métrica de **matriz de costos asimétrica** (casi nunca accuracy)
- **presupuesto de 10 envíos, mínimo 5 min entre cada uno, y cuenta el ÚLTIMO
  enviado — no el mejor**
- cerrado el plazo hay **30 minutos** para entregar el probatorio: un único
  archivo digital que justifique las decisiones (art. 5 del reglamento)

**Leé `PLAYBOOK_HACKATHON.md` antes que nada.** La Hackathon IA de la FI-UNaM
NO dura 48 h: son **dos medias jornadas, ~10,5 h de reloj y ~9 h efectivas**,
presencial, en grupos que se arman ese mismo día. Ese documento tiene el
cronograma hora por hora, el reparto de roles y el presupuesto de envíos.

Ya existe un kit probado. **Usalo. No lo reescribas.** Si algo le falta,
extendelo en un módulo nuevo y decilo explícitamente.

Tu entregable de la primera hora es un `STRATEGY.md` (formato en la sección 7).

---

## 2. Entorno

- **Primero de todo: `python bootstrap.py`.** El evento es presencial y puede
  no ser su máquina. El script diagnostica, instala lo que falte (con `--install`,
  cae a `--user` si no hay permisos) y corre un autotest que valida el kit
  entero. Si LightGBM no entra, el kit degrada solo a sklearn.
- Windows 11, shell PowerShell (el Bash tool también existe, sintaxis distinta)
- Kit en `Desktop/.Estudiar/1° Cuatri/Inteligencia computacional/hackathon-kit/`
- Intérprete: **`./.venv/Scripts/python.exe`** — tiene numpy, pandas,
  scikit-learn, lightgbm, scipy, matplotlib, joblib
- **NO hay torch/timm/GPU local.** Todo lo de visión va a Google Colab con GPU.
  Instalá ahí con `!pip -q install timm`.
- `pdftotext -layout archivo.pdf -` (vía Bash) extrae el texto de las consignas.
  El tool Read no puede renderizar PDFs en esta máquina.

---

## 3. Mapa del kit

| Archivo | Qué resuelve |
|---|---|
| `run_tabular.py` | baseline tabular completo, de CSV a `submit.csv` |
| `run_vision.py` | baseline de imágenes (Colab), de carpetas a `submit.csv` |
| `ic_kit/costs.py` | matrices de costo + decisión bayesiana |
| `ic_kit/cleaning.py` | auditoría y limpieza de datos sucios |
| `ic_kit/tabular.py` | OOF, blend, `fit_prior`, pseudo-labeling |
| `ic_kit/vision.py` | timm, TTA, duplicados, etiquetas ruidosas |
| `ic_kit/traps.py` | **detector de trampas del dataset** |
| `ic_kit/submit.py` | validador de formato + gestor de los 10 envíos |
| `ic_kit/probatorio.py` | genera el probatorio como notebook autocontenido |
| `ic_kit/eda.py` | EDA prearmado orientado a la métrica, con figuras numeradas |
| `ic_kit/checkpoints.py` | reanudación tras caída de Colab, escritura atómica |
| `ic_kit/bitacora.py` | registro de hipótesis/hallazgos/decisiones/descartes |
| `notebooks/BASE_tabular.ipynb` | **plantilla de arranque para desafío con CSV** |
| `notebooks/BASE_vision.ipynb` | **plantilla para imágenes (Keras, stack de la cátedra)** |
| `bootstrap.py` | arranque y autotest en una máquina desconocida |
| `PLAYBOOK_HACKATHON.md` | **cronograma real del evento, roles, presupuesto** |

---

## 4. API que vas a usar

```python
from ic_kit import costs, cleaning, tabular, traps, vision
from ic_kit import submit as sub_mod
```

### costs.py — el corazón del kit
```python
costs.COST_D1, costs.LABELS_D1          # triaje 2026 (4 clases, mayor score mejor)
costs.COST_D2, costs.KINGDOM_D2         # especies 2026 (10 clases, menor costo mejor)
costs.grouped_cost_matrix(groups, same_group=2., diff_group=5.)
costs.mean_cost(y_true, y_pred, C)      # costo promedio
costs.bayes_decision(proba, C)          # argmin_j Σ_i p(i|x)·C[i,j]   <-- CLAVE
costs.expected_cost(proba, C)           # (n, k) costo esperado por decisión
costs.decision_gain(y, proba, C)        # cuánto ganás vs argmax
costs.confusion_cost_report(y, pred, C, labels)   # de dónde viene tu costo
costs.class_weights_from_cost(C)        # sample_weight derivado del costo
```

### cleaning.py — datos sucios
```python
cleaning.audit(train, target, df_test)  # DataFrame con columna `alertas`
cleaning.duplicate_report(df, target)
cleaning.auto_clean(df, target, ranges={'temp': (30,45)}, drop_cols=[...])
cleaning.fix_units(s, factor=5/9, offset=-32*5/9, threshold=60, above=True)
cleaning.psi(a, b)                      # drift entre dos series
cleaning.is_text(s)                     # pandas 3 usa dtype 'str', no 'object'
```
Alertas que emite `audit`: `CONSTANTE`, `ID?`, `NUMERO_COMO_TEXTO`, `CENTINELA`,
`MEZCLA_UNIDADES?`, `TYPOS/CASE`, `OUTLIERS_EXTREMOS`, `DRIFT_TEST`,
`CATS_NUEVAS(n)`.

### tabular.py
```python
X, y, Xte, ids, classes = tabular.prepare(
    train, test, target, id_col, class_order=[...])   # class_order NO es opcional
oof, pte, info = tabular.oof_lgb(X, y, Xte, params=None, n_splits=5, seeds=(0,1,2),
                                 sample_weight=None)
w  = tabular.fit_prior(oof, y, C)         # re-pesado de probabilidades
bw, c = tabular.blend_weights([oof_a, oof_b], y, C)
tabular.pseudo_label(X, y, X_unlabeled, Xte, C, conf=0.9)
tabular.permutation_cost_importance(fn_predict, X, y, C)
```

### traps.py — corré esto en la primera hora
```python
traps.run_all(X, y, Xte, C=None, oof_proba=None, df_raw=None, id_col=None)
traps.adversarial_validation(X_train, X_test)   # -> (auc, importancias, p_es_test)
traps.leakage_scan(X, y)                        # poder predictivo de cada feature sola
traps.order_leak(df, y, id_col)                 # ¿las filas vienen ordenadas por clase?
traps.shortcut_scan_images(paths, y, size=16)   # Clever Hans: ¿alcanza el color?
traps.label_noise_scan(oof, y, C, top=40)
traps.fit_prior_honest(oof, y, C)               # ¿el ajuste generaliza o sobreajusta?
traps.class_prior_shift(y_train, proba_test)
traps.estimate_test_prior(proba_test, prior_train)   # EM de Saerens (SLD)
traps.validate_prior_em(oof, y)                      # ¿el EM sirve acá? SE COMPRUEBA
```
Sobre `estimate_test_prior`: corregí el prior del test **sin mirar sus
etiquetas** (es legítimo, usa sólo las probabilidades predichas). Pero es
frágil: con un clasificador débil colapsa las clases raras a prior 0 y empeora
el resultado. **Corré siempre `validate_prior_em` primero** — resamplea tu OOF
a un prior conocido y mide si el EM lo recupera. Y tené presente su límite:
sólo sirve para prior shift puro; si lo que cambia entre train y test es la
*definición* de las clases, ningún EM lo arregla.

### submit.py
```python
s = sub_mod.make_submission(ids, preds, classes, path, id_col, target_col)
sub_mod.validate(s, expected_ids, allowed_labels, train_dist=...)
sub_mod.diff_vs("nuevo.csv", "anterior.csv")    # % de filas que cambian
log = sub_mod.SubmitLog(lower_is_better=False)  # ¡ajustá según la métrica!
log.can_submit(); log.record(file, cv=..., notes=...); log.set_lb(n, lb); log.report()
```

### eda.py / checkpoints.py / bitacora.py
```python
eda.EDA(train, target, test=test, C=C, clases=CLASES, id_col=ID).todo()
checkpoints.oof_con_checkpoint(X, y, Xte, C=C, nombre="oof")   # reanuda por fold
checkpoints.montar_drive()
bit = bitacora.Bitacora(); bit.hipotesis(...); bit.hallazgo(t, evidencia, decision)
bit.decision(t, evidencia); bit.descarte(t, motivo); bit.mostrar()
```
La bitácora es lo que llena las secciones del probatorio que se califican.
`generar_notebook()` la lee sola desde `work/bitacora.json`.

**Empezá desde `notebooks/BASE_tabular.ipynb` o `BASE_vision.ipynb`**, no desde
cero: ya traen el recorrido completo y sólo hay que completar la celda de
configuración.

### probatorio.py — los últimos 30 minutos
```python
from ic_kit.probatorio import generar_notebook
generar_notebook(grupo="Grupo 8", integrantes=["Acosta","Bareiro","Borges"],
                 desafio="...", metrica="... (decir si mayor o menor es mejor)",
                 oof="work/oof.npy", y="work/y.npy", C="d1",
                 trampas=[{"hallazgo":"","evidencia":"","decision":""}],
                 hipotesis=[], decisiones=[], descartado=[])
```
Produce un `.ipynb` **autocontenido**: los OOF y el historial van embebidos en
base64, así que corre en una carpeta vacía sin reentrenar. Escribe solo todo lo
mecánico; deja marcado con "(completar a mano)" lo que se califica: hipótesis,
hallazgos con evidencia, decisiones y **lo descartado con su razón**. Ir
anotando eso durante la jornada es responsabilidad del rol Entrega.

### vision.py (Colab)
```python
paths, y, classes, groups = vision.build_index("data/train")
vision.corrupt_images(paths); vision.duplicate_groups(paths)
dl_tr, dl_va = vision.make_loaders(paths, y, idx_tr, idx_va, img_size=320)
model = vision.train_fold(dl_tr, dl_va, n_classes, "convnext_tiny.fb_in22k", epochs=8)
p = vision.predict_tta(model, paths, img_size=320)
vision.kingdom_proba(p, grupos); vision.hierarchical_proba(p, grupos)
vision.suspect_labels(oof, y, C, top=50)
```
Los `groups` de `build_index` son hashes perceptuales: usalos con
`StratifiedGroupKFold` para que los duplicados no se repartan entre folds.

---

## 5. Trampas conocidas y su antídoto

La cátedra diseña datasets que castigan las soluciones automáticas. Ninguna de
estas trampas es magia; todas se detectan antes de entrenar.

| Trampa | Cómo se ve | Detector | Antídoto |
|---|---|---|---|
| **Fuga (leakage)** — una columna calculada después de conocer la etiqueta | CV casi perfecto, leaderboard pésimo | `leakage_scan` (AUC sola > 0.95) | verificar que la columna esté igual de poblada en el test; si no, tirarla |
| **Covariate shift** — el test no es una muestra del train | CV linda, LB fea, sin explicación | `adversarial_validation` (AUC > 0.75) | entrenar pesando las filas parecidas al test (`p_es_test`), o validar sobre ese subconjunto |
| **Prior shift** — otra proporción de clases en el test | el modelo sub-predice las clases raras | `class_prior_shift` | `estimate_test_prior` (EM), previa comprobación con `validate_prior_em`. `fit_prior` NO alcanza: sólo ve el prior del train |
| **Fuga por orden** — filas agrupadas por clase | el índice correlaciona con el target | `order_leak` | nunca usar el índice como feature; mezclar antes de partir |
| **Atajo / Clever Hans** — el fondo o el color separa las clases | accuracy alta con imágenes de 16×16 | `shortcut_scan_images` | ColorJitter fuerte, RandomResizedCrop, grises aleatorios |
| **Ruido de etiquetas** — un % mal cargado a propósito | techo de accuracy que no sube | `label_noise_scan` | `label_smoothing`, no perseguir accuracy perfecta, revisar a ojo el top 40 |
| **Columnas sucias** — unidades mezcladas, centinelas, typos | distribuciones bimodales, `-999` | `cleaning.audit` | `auto_clean` + `fix_units` |
| **Sobreajuste al OOF** — ajustar umbrales muchas veces sobre la misma partición | ganancia que no aparece en el LB | `fit_prior_honest` | menos rondas de ajuste; si no generaliza, `bayes_decision` pelado |

**Lo que NO es una estrategia válida:** intentar leer las etiquetas del test,
sondear el leaderboard para reconstruirlo, o usar `test_features.csv` para
ajustar o validar. Las consignas lo prohíben explícitamente y es motivo de
desaprobación. Todo lo de arriba es análisis del *train*, que es legítimo y es
justamente lo que las trampas premian.

---

## 6. Protocolo de trabajo

**Hora 0–1 — reconocimiento, sin entrenar**
1. Extraer el texto de la consigna y **transcribir la matriz de costos a mano**.
2. Fijar por escrito: ¿mayor o menor es mejor? ¿cuál es el error más caro?
3. `cleaning.audit()` y `traps.run_all()`.
4. Definir el **orden de clases** y verificarlo contra la tabla de la consigna.

**Hora 1–4 — baseline y primer envío**
5. `run_tabular.py` (o `run_vision.py`) completo.
6. Enviar. Registrar con `SubmitLog`. Cargar el score del LB apenas aparezca.
7. Con 3 envíos ya tenés la correlación CV↔LB. **Si r < 0.5, parás todo y
   arreglás el CV antes de tocar el modelo.**

**Hora 4–20 — mejoras, en este orden estricto**
8. Limpieza dirigida → feature engineering de dominio → ensamble →
   `fit_prior` → recién ahí hiperparámetros.
9. Cada mejora se acepta sólo si supera el desvío entre folds.

**Últimas horas**
10. Checklist final del `PLAYBOOK_HACKATHON.md`.
11. **El último envío debe ser el mejor archivo, aunque ya lo hayas mandado.**

---

## 7. Formato de tu entregable

Escribí `work/STRATEGY.md` con exactamente estas secciones:

```markdown
# Estrategia — <nombre del desafío>

## 1. Lectura de la métrica
- dirección (mayor/menor es mejor), matriz de costos transcripta
- error más caro y qué fracción de casos lo sufre
- orden de clases confirmado contra la consigna

## 2. Diagnóstico del dataset
- resultado de audit() y de traps.run_all()
- trampas detectadas y decisión tomada para cada una

## 3. Plan de modelado
- ordenado por rentabilidad esperada, con tiempo asignado a cada paso
- qué se descarta y por qué

## 4. Presupuesto de los 10 envíos
- qué se envía en cada uno y qué pregunta responde

## 5. Riesgos
- qué haría fracasar este plan y cuál es el plan B
```

---

## 8. Reglas que no se negocian

1. **Nunca** entrenar, validar ni ajustar nada con el archivo de test.
2. **Siempre** pasar `class_order` explícito a `prepare()`. Sin eso pandas
   ordena alfabéticamente y la matriz de costos queda desalineada: el código
   corre sin error y el score baja ~10 %.
3. **Siempre** `bayes_decision`, nunca `argmax`, salvo que la métrica sea
   accuracy pura.
4. Verificar la dirección de la métrica antes de escribir una línea de código.
   `SubmitLog(lower_is_better=...)` tiene que coincidir.
5. Semillas fijas en 42 (convención de la materia) y `report.json` guardado:
   hay que poder replicar la solución.
7. Estilo de notebook de la materia: español, sin emojis, sin citar las
   preguntas de la consigna, sin explicar conceptos básicos, sin conclusiones
   en formato pregunta-respuesta, markdown conciso. Se califica el análisis y
   la justificación, no el código.
8. Ir anotando hipótesis, hallazgos y descartes DURANTE la jornada. A las
   09:30 del viernes no hay tiempo de reconstruirlos.
6. Reportar los resultados como son. Si el CV empeoró, se dice.
