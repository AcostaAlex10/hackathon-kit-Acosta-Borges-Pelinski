# Guía del equipo — hackathon-kit

Para Acosta, Borges y Pelinski. Si tenés 5 minutos, leé la sección 1 y la 7.
Si tenés 30, leé todo y hacé el ensayo.

---

## 1. Qué es esto y por qué existe

Un conjunto de herramientas para competencias donde te dan un dataset, un
problema en contexto y una **métrica de matriz de costos asimétrica** — o sea,
equivocarse de una manera cuesta mucho más que de otra.

Tres cosas que este kit resuelve y que casi nadie hace bien bajo presión:

**Uno.** Con costos asimétricos, predecir la clase más probable **no es óptimo**.
Lo óptimo es elegir la clase de menor costo esperado. En nuestro desafío de
ensayo eso solo bajó el costo de 1,904 a 1,196 — sin tocar el modelo.

**Dos.** Los datasets vienen con trampas deliberadas. Una columna que en el
train predice perfecto y en el test está rota te da la mejor validación cruzada
del aula y el peor puntaje del ranking. Medido: CV de 0,214 contra un costo
real de 2,877.

**Tres.** Son 10 envíos, cuenta el último, y hay 30 minutos para el probatorio.
El kit lleva la contabilidad y genera el probatorio solo.

---

## 2. Arranque

### En una máquina cualquiera (aula, notebook propia)

```bash
git clone https://github.com/AcostaAlex10/hackathon-kit-Acosta-Borges-Pelinski.git
cd hackathon-kit-Acosta-Borges-Pelinski
python bootstrap.py --install
```

`bootstrap.py` reporta qué Python y qué paquetes hay, instala lo que falte (si
no hay permisos de administrador reintenta con `--user`), y corre un autotest
de 20 segundos que valida el kit entero. **Si termina con `AUTOTEST OK`, estás
listo.** Si LightGBM no entra, no importa: el kit degrada solo a
scikit-learn y sigue funcionando.

### En la nube (Colab)

Abrí `notebooks/arranque_colab.ipynb` en Colab y corré la primera celda. Clona
el repo, instala y corre el autotest. Obligatorio si el desafío es de imágenes:
ahí necesitás GPU.

---

## 3. Qué hace cada módulo

### `ic_kit/costs.py` — el corazón
Matrices de costo y la regla de decisión. Lo único que hay que entender sí o sí:

```python
from ic_kit import costs
pred = costs.bayes_decision(proba, C)      # en vez de proba.argmax(1)
```

Elige, para cada fila, la clase que minimiza el costo esperado. También:
`mean_cost` (la métrica), `confusion_cost_report` (te dice de qué confusión
concreta viene tu costo, ordenada — atacás de arriba hacia abajo) y
`decision_gain` (cuánto ganás contra `argmax`).

### `ic_kit/cleaning.py` — datos sucios
`audit(train, target, test)` devuelve una ficha por columna con alertas:
`CONSTANTE`, `ID?`, `NUMERO_COMO_TEXTO` (un `"36,5 °C"` guardado como texto),
`CENTINELA` (los `-999`), `MEZCLA_UNIDADES?` (grados C y F en la misma
columna), `TYPOS/CASE`, `DRIFT_TEST`, `VACIA_EN_TEST`.

`auto_clean()` arregla lo mecánico. No imputa a propósito: los árboles manejan
faltantes solos y el patrón de qué falta suele tener señal.

### `ic_kit/traps.py` — el detector de trampas
**Esto se corre en la primera hora, antes de entrenar.**

```python
from ic_kit import traps
traps.run_all(X, y, Xte, df_raw=train, id_col="id")
```

- `leakage_scan` — mide qué tan bien predice cada columna **por sí sola**. Un
  AUC de 0,99 no es un regalo, es una fuga.
- `adversarial_validation` — entrena un clasificador train-contra-test. Si los
  distingue (AUC > 0,75), el test no es una muestra del train y tu CV miente.
- `order_leak` — ¿las filas vienen ordenadas por clase?
- `class_prior_shift` + `estimate_test_prior` — ¿el test tiene otra proporción
  de clases? Se corrige sin mirar sus etiquetas.
- `validate_prior_em` — **comprueba** si esa corrección sirve en tus datos antes
  de aplicarla. Es frágil y a veces empeora; esto lo detecta de antemano.
- `fit_prior_honest` — ¿el ajuste de umbrales generaliza o se sobreajustó?
- `shortcut_scan_images` — para imágenes: ¿alcanza con el color del fondo?

### `ic_kit/tabular.py` — el modelo
`prepare()`, `oof_lgb()` (validación cruzada con varias semillas),
`blend_weights()` (pesos de ensamble que minimizan el **costo**, no el
logloss), `fit_prior()`, `pseudo_label()`.

### `ic_kit/submit.py` — los 10 envíos
`validate()` chequea el CSV antes de subirlo. `SubmitLog` lleva el presupuesto,
el cooldown de 5 minutos, avisa si el archivo es idéntico a uno anterior, y
calcula la correlación entre nuestra validación y el ranking.

### `ic_kit/probatorio.py` — los últimos 30 minutos
Genera el probatorio como notebook **autocontenido**: los resultados van
embebidos, así que corre en una carpeta vacía sin reentrenar.

### `ic_kit/eda.py` — el EDA prearmado
`EDA(train, target, test, C, clases).todo()` responde de una las preguntas que
bajo costo asimétrico cambian decisiones: cómo se reparte el target y cuánta
**exposición al costo** aporta cada clase, qué columnas traen anomalías, qué
variables separan la **clase cara** (que casi nunca es la más frecuente), si
train y test se parecen, y qué costo dan las estrategias triviales — el piso
que hay que superar. Guarda las figuras numeradas (`fig01_...png`) para pegar
en el informe, y `.zip()` las descarga.

### `ic_kit/checkpoints.py` — que Colab no nos cueste 40 minutos
`oof_con_checkpoint()` es igual a `oof_lgb` pero guarda después de cada fold.
Si la sesión se corta en el fold 12 de 15, reejecutar la celda retoma en el 12.
Escritura atómica (a `.tmp` y después `os.replace`, así una caída no corrompe
el checkpoint anterior), guarda el estado del generador aleatorio, y espeja en
Drive si está montado — el disco de Colab se borra al reiniciar el entorno.

### `ic_kit/bitacora.py` — lo que después se califica
El probatorio se evalúa por las **hipótesis, hallazgos, decisiones y
descartes**, no por las métricas. Eso no se reconstruye a las 09:30 del
viernes. Se anota en una línea mientras pasa:

```python
bit.hipotesis("El lactato concentra la señal de gravedad")
bit.hallazgo("Fuga en orden_trabajo", "AUC en solitario = 0.92", "descartada")
bit.decision("Mínimo costo esperado en vez de argmax", "1.345 -> 0.997")
bit.descarte("Pseudo-etiquetado", "no bajó el costo OOF")
```

`generar_notebook()` lee la bitácora sola y arma esas secciones.

### `ic_kit/vision.py` — desafíos de imágenes
timm, TTA, detección de duplicados por hash perceptual (para que el CV no
mienta) y detección de etiquetas mal puestas.

---

## 4. Los notebooks base

Son la plantilla de arranque, con el mismo formato que usa la cátedra:

| Notebook | Cuándo |
|---|---|
| `notebooks/BASE_tabular.ipynb` | desafío con CSV |
| `notebooks/BASE_vision.ipynb` | desafío con imágenes (Keras, como la cátedra) |
| `notebooks/arranque_colab.ipynb` | esqueleto mínimo si preferís armarlo a mano |

Traen ya resueltos: clonado del repo, EDA completo, detección de anomalías,
limpieza, modelo con reanudación, decisión de costo, envío validado y
probatorio. **Lo único que se toca al empezar es la celda `Configuración`**:
target, id, `CLASES` en el orden de la consigna y la matriz de costos. Tienen
`assert` que frenan si el orden no coincide.

El de visión mantiene Keras/TF para no cambiar de stack en plena competencia, e
incluye reanudación por época, descongelado en dos etapas, aumentación fuerte
de color (la defensa contra el atajo de fondo) y promediado con espejado en la
predicción.

---

## 5. Los dos scripts que hacen todo

```bash
# Tabular: de CSV a submit.csv, con auditoría y decisión de costo incluidas
python run_tabular.py --train data/train.csv --test data/test.csv \
    --target nivel_urgencia --id id_paciente --cost d1 --out work/submit.csv

# Imágenes (Colab con GPU)
python run_vision.py --train data/train --submit data/only_submit \
    --cost d2 --model convnext_tiny.fb_in22k --size 320 --epochs 8
```

Para un desafío nuevo, la matriz de costos se pone en `ic_kit/costs.py` o se
pasa como CSV con `--cost ruta.csv`, **más `--classes "a;b;c"` en el mismo
orden que las filas de la matriz**. Ver sección 7.

---

## 6. Practicá antes: el ensayo

Hay un desafío falso completo con 10 trampas plantadas y solución sellada.

```bash
python tests/make_ensayo.py
# leer ENSAYO_DESAFIO.md, resolverlo, y después:
python tests/score_ensayo.py work/submit.csv
```

Cronometrate en 3 horas. No es para ganarlo: es para que el flujo salga sin
pensar el día de la hackathon. **No abras la sección 7 de `ENSAYO_DESAFIO.md`
antes de terminar**, tiene las respuestas.

---

## 7. Los cinco errores que nos pueden costar el puesto

**El orden de las clases.** Si las etiquetas son texto, pandas las ordena
alfabéticamente (`Crítico, Muy urgente, No urgente, Urgente`) y la matriz de
costos está en orden clínico (`No urgente, Urgente, Muy urgente, Crítico`). El
código corre sin un solo error y el puntaje baja ~10 %. Por eso `prepare()`
exige `class_order` explícito. **Verificarlo contra la tabla de la consigna
antes de entrenar.**

**Usar `argmax`.** Ver sección 1.

**Confiar en una CV que no se validó.** Si la correlación entre nuestra
validación y el ranking es baja, hay una fuga. Se para todo y se arregla el CV
antes de tocar el modelo.

**Tocar el test.** Está prohibido por consigna y además nos ilusiona. El OOF es
el único juez.

**Perder el mejor archivo.** Cuenta el **último** envío, no el mejor. El último
envío del día tiene que ser el mejor archivo, aunque ya lo hayamos mandado.

---

## 8. Cómo trabajamos entre nosotros

Ramas: `herramientas` es el kit estable. Cada uno tiene la suya —`Acosta`,
`Borges`, `Pelinski`— para experimentar sin pisarse.

```bash
git checkout Borges
git pull origin herramientas      # traer lo último del kit
# ... trabajar, commitear ...
git push origin Borges
```

Lo que funcione y esté probado se lleva a `herramientas` con un merge.
**Nadie commitea directo a `herramientas` durante la competencia**: el kit
tiene que quedar estable mientras los tres experimentamos.

Los datos del desafío **no van al repo** (están en `.gitignore`). Se comparten
aparte.

Reparto de roles sugerido para la jornada, en `PLAYBOOK_HACKATHON.md`. La regla
que importa: **quien maneja los envíos no es quien entrena.** El que entrena
siempre quiere mandar una más.

---

## 9. Mapa de documentos

| Archivo | Cuándo leerlo |
|---|---|
| `GUIA_EQUIPO.md` | ahora (este) |
| `PLAYBOOK_HACKATHON.md` | antes de la jornada: cronograma, roles, envíos |
| `ENSAYO_DESAFIO.md` | para practicar |
| `CONTEXTO_CLAUDE.md` | para pasarle a una sesión de Claude Code |
| `README.md` | referencia rápida |
| `PLAYBOOK.md` | sólo para desafíos de cátedra de 48 h |
