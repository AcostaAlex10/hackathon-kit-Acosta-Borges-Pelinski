# Estrategia 1 — Desafío Machine Health

Documento de contexto y estrategia para la Hackathon IA FI-UNaM (27-28/08/2026).
Equipo Acosta · Borges · Pelinski.

Sirve para dos cosas: como registro de lo que ya sabemos y como prompt de
arranque para una sesión nueva de Claude Code. Todas las cifras salen de
mediciones propias sobre `train_sup`, con la métrica oficial del kit de la
cátedra (`scoring.py`) y validación cruzada agrupada por `machine_id`.
Ninguna usó el test para entrenar, validar ni ajustar.

**Revisión 2**, tras el benchmark multimodelo de la mañana del jueves
(`eda_y_benchmark.ipynb`). Lo que cambió: sabemos dónde está el techo tabular y
sabemos que el algoritmo no es la palanca. La sección 3 se reescribió entera.

---

## 0. Contexto para una sesión nueva

Planta industrial con máquinas rotativas (motor eléctrico + bomba hidráulica)
instrumentadas con sensores de vibración, corriente, tensión, temperatura,
presión y caudal. Las señales se agregan en ventanas temporales y producen
descriptores estadísticos y espectrales. Hay 13 modos de falla (F01-F13) en
cuatro familias: mecánica (F01-F06), estructural (F07), eléctrica (F08-F11) e
hidráulica (F12-F13).

La entrega es un CSV con una fila por `window_id` del test y 14 columnas:
`window_id,F01,...,F13`, con la probabilidad de cada falla en [0,1].
Servidor de ranking: `http://192.168.25.239:8000/`.

Archivos: `train_sup` (1750 ventanas etiquetadas), `test` (1848), 
`train_unlabeled` (434), `raw_signal_index` (4032 = todas), más las señales
crudas en NPZ. 46 features tabulares utilizables.

---

## 1. Lectura de la métrica

`final_score = 0.70 * cost_score + 0.20 * prob_score + 0.10 * diag_score`.
**Mayor es mejor**, rango [0,100].

- `cost_score = 100 * max(0, 1 - mean_cost / naive_cost)`. Simula decisiones de
  mantenimiento (MONITOR, INSPECT familia, DERATE, STOP) y mide cuánto se ahorra
  contra no hacer nada.
- `prob_score = 100 * max(0, 1 - brier / 0.25)`. Calibración.
- `diag_score = 100 * macro_f1`, con umbral fijo en 0.5.

Tres consecuencias que cambian el plan:

**El 70 % del score paga por la familia, no por la falla.** `INSPECT` se elige
por familia. Acertar F03 en vez de F04 no cambia un centavo: las dos son
mecánicas y la inspección mecánica cuesta $4 igual. Medido: accuracy de estado
0.461 contra accuracy de familia 0.635.

**El techo no es 100, es 73.96.** Con las etiquetas exactas, `cost_score` se
planta en 62.80 porque igual hay que pagar la inspección. `naive_cost = 9.42`;
el costo mínimo alcanzable es `0.876 * 4 = 3.504`. La banda de la consigna
("> 55 excelente") recién se entiende con este techo a la vista.

**La capa de decisión ya está óptima. No tocarla.** La métrica corre su propio
`choose_action` sobre las probabilidades que enviamos. Esa regla coincide en el
**99.1 %** de las ventanas con la acción óptima calculada bajo nuestro propio
posterior. Es el caso raro en que la palanca histórica del kit
(`costs.bayes_decision`) no aplica: ya viene incorporada. Distorsionar las
probabilidades para forzar acciones cuesta más en Brier de lo que devuelve en
costo:

    afilado p^alpha:  alpha=1 -> 43.21   alpha=2 -> 42.69   alpha=6 -> 42.07

La única forma de mover `cost_score` es predecir mejor.

---

## 2. Diagnóstico del dataset

**No es multilabel. Es un problema de 14 clases.** La consigna insiste en fallas
simultáneas y estados combinados. En las 1750 ventanas etiquetadas no hay una
sola con dos fallas: 217 sanas y 1533 con exactamente una. `is_combo` vale 0 en
todas las filas.

    E[#fallas] = 0.876 = P(al menos una falla)
    familias distintas por ventana: {0: 217, 1: 1533}

Por eso el modelo es un softmax de 14 estados (sano + F01-F13), no 13
clasificadores binarios independientes. Las probabilidades salen coherentes
entre sí y suman <= 1, que es justo lo que consume `choose_action`.

**Train y test no comparten ni una máquina ni una sesión.** 26 máquinas en
train, 22 en test, intersección vacía. La validación adversarial da AUC 0.988,
pero no es una fuga: el test opera en otro punto de trabajo (rpm medio 2052
contra 1742; Tamb 29.9 contra 26.8). Revisamos las columnas que más separan y
ninguna está rota ni vacía en el test.

El antídoto es normalizar por máquina, y es la mejora más grande medida:

    + z-score por machine_id   43.21 -> 47.17   (acc_familia 0.574 -> 0.635)
    + z-score por session_id   43.21 -> 42.22   (empeora: normaliza la falla misma)

El modelo tiene que aprender desviaciones respecto del comportamiento propio de
cada máquina, no valores absolutos. La validación va con `StratifiedGroupKFold`
agrupando por `machine_id`, sin excepción.

**Las señales crudas son el techo que queda.** El zip que recibimos se llama
`datos_sinraw` y no las trae, pero `raw_signal_index` mapea las 4032 ventanas
(train + test + unlabeled) a sus NPZ: `acc_radial_a` a 20 kHz, `current_a` a
10 kHz, `pressure_in` a 200 Hz. La consigna dice que los descriptores tabulares
alcanzan "al menos esa es la promesa que dejó el becario antes de abandonar su
puesto en la empresa": el chiste está puesto a propósito. La tabla trae un solo
armónico por canal (1x y 2x), y las frecuencias que separan las fallas de
rodamiento entre sí (BPFO, BPFI, BSF) no están ahí y sí en la señal.

**Prevalencias** (suman 0.876, sano = 0.124):

    F01 .096  F02 .096  F03 .092  F04 .056  F05 .064  F06 .060  F07 .052
    F08 .064  F09 .048  F10 .048  F11 .056  F12 .092  F13 .052

De las ventanas con falla, el 57.5 % es severa (`max_severity > 0.5`), que es lo
que separa el falso negativo de $15 del de $5.

---

## 3. Plan de modelado

### Lo medido

Todo en OOF, `StratifiedGroupKFold(5)` agrupado por `machine_id`, features
originales más su z-score por máquina.

| Configuración | Final | Cost | Prob | Diag |
|---|---|---|---|---|
| P0 prevalencia (baseline oficial) | 29.05 | 20.00 | 75.25 | 0.00 |
| LGBM 14 clases, sin normalizar | 43.21 | 34.21 | 78.46 | 35.71 |
| + afilado alpha=2 (descartado) | 42.69 | 33.98 | 75.28 | 38.48 |
| + z-score por máquina | 47.17 | 39.13 | 79.75 | 38.24 |
| **Ensamble LGB + RF + RegLogística** | **49.36** | 41.47 | 82.58 | 38.10 |
| Oráculo (etiquetas exactas) | 73.96 | 62.80 | 100.0 | 100.0 |

### El algoritmo no es la palanca

Seis algoritmos como clasificadores multiclase de 14 estados, misma partición,
mismas features:

| Modelo | final | cost | prob | diag | acc_familia | sd_folds |
|---|---|---|---|---|---|---|
| LightGBM | 47.17 | 39.13 | 79.75 | 38.24 | 0.635 | 1.79 |
| XGBoost | 46.95 | 38.68 | 80.71 | 37.37 | 0.637 | 3.15 |
| CatBoost | 46.29 | 37.43 | 82.72 | 35.41 | 0.605 | 2.93 |
| RandomForest | 45.53 | 37.37 | 83.14 | 27.47 | 0.605 | 2.31 |
| ExtraTrees | 45.39 | 38.22 | 82.37 | 21.61 | 0.595 | 2.83 |
| RegLogística | 42.71 | 34.20 | 76.08 | 35.58 | 0.583 | 6.21 |

Los seis caen dentro de 4.5 puntos, un margen más chico que el desvío entre
folds de varios de ellos. Repetido con otra semilla de partición (2024), **el
ranking se reordena por completo**: RandomForest pasa de cuarto a primero
(45.79), CatBoost de tercero a último (44.26), y LightGBM baja de 47.17 a 45.62
sin que cambie nada del modelo.

Un ranking que se da vuelta al cambiar la partición no mide calidad de
algoritmo, mide ruido. **Buscar un modelo mejor es tiempo perdido**, y el tuneo
de hiperparámetros más todavía: mueve menos que el cambio de familia de modelo,
que ya vimos que no mueve nada.

### Lo que sí funciona: ensamble con diversidad deliberada

Nadie gana en las tres componentes. Los boosters (LightGBM, XGBoost) ganan en
`diag_score` porque discriminan mejor; los bagging y CatBoost ganan en
`prob_score` porque salen mejor calibrados. El promedio simple de
probabilidades captura las dos cosas:

| Ensamble | semilla 42 | semilla 2024 | ventaja |
|---|---|---|---|
| LightGBM solo | 47.17 | 45.62 | — |
| LGB + XGB | 47.77 | 45.85 | +0.61 / +0.23 |
| LGB + XGB + Cat | 47.85 | 46.84 | +0.69 / +1.22 |
| LGB + XGB + Cat + RF | 47.87 | 47.29 | +0.70 / +1.67 |
| **LGB + RF + RegLogística** | **49.36** | **47.65** | **+2.19 / +2.03** |

El ganador salió de buscar entre las 41 combinaciones posibles sobre la misma
partición, así que se lo trató como sospechoso de sobreajuste de selección y se
lo verificó con la semilla de control. **Sobrevivió casi intacto** (+2.19 y
+2.03): la ventaja es real.

Por qué funciona: la regresión logística es el peor modelo individual de los
seis y es la que hace andar el ensamble. Es la única lineal, y sus errores están
descorrelacionados de los de los árboles. En un promedio, la diversidad vale más
que la calidad individual.

### El techo tabular

**Alrededor de 48 puntos** (48.50 promediando las dos particiones, 49.36 en el
mejor caso), contra 29.05 del piso y 73.96 del oráculo: unos dos tercios de lo
alcanzable.

Que seis algoritmos con sesgos inductivos muy distintos converjan al mismo
número es la definición operativa de un techo: el límite no está en el modelo,
está en la información que traen las features.

Dónde se traba, concretamente: la accuracy de **familia** se estanca en
0.60-0.64 en los seis modelos, y `cost_score` —el 70 % del score— paga
exactamente por ella. Las seis fallas mecánicas se confunden entre sí porque la
tabla trae un solo armónico por canal (1x y 2x), y las frecuencias que las
separan (BPFO, BPFI, BSF) no están ahí.

### Orden de trabajo, revisado

1. **Ensamble por promedio simple.** LightGBM + RandomForest + regresión
   logística. Sale de modelos ya entrenados: es la mejora más barata que queda.
   Incluir el modelo lineal aunque sea el peor individual.
2. **Features espectrales desde los NPZ.** BPFO, BPFI, BSF, bandas de
   envolvente. Es lo único que ataca la confusión intra-familia mecánica y la
   única vía para romper 50.
3. **Features de dominio eléctricas.** Desequilibrio de tensión y de corriente
   entre fases: F08-F11 son las cuatro eléctricas y hoy se confunden.
4. **Features hidráulicas.** `delta_p` contra caudal, para separar cavitación
   (F12) de obstrucción (F13).
5. **Calibración.** `prob_score` pesa 20 % y está en 80-83.
6. **Umbral de diagnóstico.** `macro_f1` corta en 0.5 fijo; `diag_score` pesa
   10 % y está en 36, que es donde más margen relativo queda. No requiere
   features nuevas.

### Lo descartado y por qué

- *Afilado o distorsión de probabilidades para forzar acciones:* medido, empeora
  el score. La capa de decisión ya está óptima (99.1 % de coincidencia).
- *13 clasificadores binarios independientes:* el train no tiene combos; el
  softmax de 14 estados aprovecha la restricción y da probabilidades coherentes.
- *Normalización por `session_id`:* empeora (43.21 -> 42.22). La sesión tiene
  una sola condición, así que normalizar por sesión borra la falla.
- *Búsqueda de un algoritmo mejor:* seis probados, el ranking no es estable
  entre particiones. Cerrado.
- *Tuneo de hiperparámetros:* mueve menos que el cambio de familia de modelo.
  No se invierte tiempo hasta tener features nuevas.
- *Mezcla jerárquica (modelo de 5 familias + reparto interno):* aporta +0.32,
  dentro del ruido. Queda superada por el ensamble, que da +2 y sí replica.

---

## 4. Presupuesto de envíos

**Ojo: la consigna y el reglamento no dicen lo mismo.** La consigna habla de
"múltiples envíos" y de que cuenta el último, pero no menciona ni el tope de 10
ni los 5 minutos de espera del reglamento. `ic_kit/submit.py` está configurado
para el régimen viejo (`MAX_SUBMITS=10`, `MIN_GAP_MIN=5`). Confirmar con la
cátedra antes del primer envío y ajustar `SubmitLog` en consecuencia. Con la
métrica actual va `SubmitLog(lower_is_better=False)`.

Plan mientras no se confirme, asumiendo el régimen restrictivo:

| Envío | Cuándo | Qué pregunta responde |
|---|---|---|
| 1 | apenas esté | ensamble LGB+RF+RegLog (49.36 OOF): calibrar el gap OOF vs ranking |
| 2-3 | jueves tarde | features espectrales de los NPZ |
| 4-5 | jueves tarde | features eléctricas e hidráulicas de dominio |
| 6 | jueves 17:30 | cierre del día, red de seguridad |
| 7-8 | viernes | calibración y umbral de diagnóstico |
| 9 | viernes | ensamble final ampliado |
| 10 | viernes, antes del cierre | reenvío del mejor archivo |

Validar cada CSV con `scoring.validate_prediction_package` antes de subir.
Cuenta el último envío, no el mejor.

---

## 5. Riesgos

**El test podría tener fallas combinadas.** El modelo de 14 clases asume una
falla por ventana porque eso es lo que muestra el train. Si el test trae combos
—y la consigna los describe con detalle— el softmax nunca reparte masa entre dos
familias a la vez. Mitigación barata: mezclar una fracción chica de un modelo de
13 binarios independientes, que sí los admite, y verificar que el OOF lo tolere
sin costo. Es el riesgo más serio de todo el plan porque es el único que no
podemos medir con los datos que tenemos.

**Dataset chico y OOF ruidoso.** 1750 filas y 46 features. El benchmark dejó
medido cuánto ruido tiene la validación: cambiar sólo la semilla de partición
mueve el `final_score` de un mismo modelo hasta 1.5 puntos, y el desvío entre
folds va de 1.7 a 6.2 según el algoritmo. **Ninguna mejora menor a ~2 puntos se
acepta sin verificarla con una segunda semilla de partición**, que es
exactamente lo que salvó al ensamble de ser descartado como ruido. Pocas
rondas, cambios grandes.

**Las señales crudas pueden no llegar a tiempo.** Si los NPZ no se pueden bajar
o procesar, el plan se queda en los pasos 2 a 5, que valen bastante menos. Plan
B: agotar el feature engineering de dominio sobre la tabla y volcar el tiempo
restante en calibración y umbral, que son los dos componentes donde todavía
queda margen barato.

**Reproducibilidad.** La consigna pide adjuntar el código que replica la
solución enviada, vía un formulario dedicado. Mantener `random_state=42` en
todos lados y el pipeline corriendo de punta a punta desde el repo.

---

## 6. Reglas que la consigna fija

- Prohibido usar el test para entrenar, etiquetar o ajustar hiperparámetros.
- Prohibido el etiquetado manual de ventanas de test por inspección humana o
  feedback experto.
- El uso de las señales crudas es opcional.
- Al finalizar hay que adjuntar el código que replica la solución.
- Del briefing del docente: prohibido consultar o delegar la resolución a
  personas ajenas al equipo; permitido el uso libre de IA, siempre que la
  resolución la lleven adelante los integrantes registrados.
