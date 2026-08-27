# Estrategia 1 — Desafío Machine Health

Documento de contexto y estrategia para la Hackathon IA FI-UNaM (27-28/08/2026).
Equipo Acosta · Borges · Pelinski.

Sirve para dos cosas: como registro de lo que ya sabemos y como prompt de
arranque para una sesión nueva de Claude Code. Todas las cifras salen de
mediciones propias sobre `train_sup`, con la métrica oficial del kit de la
cátedra (`scoring.py`) y validación cruzada agrupada por `machine_id`.
Ninguna usó el test para entrenar, validar ni ajustar.

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

Medido hasta ahora, en OOF:

| Configuración | Final | Cost | Prob | Diag |
|---|---|---|---|---|
| P0 prevalencia (baseline oficial) | 29.05 | 20.00 | 75.25 | 0.00 |
| LGBM 14 clases, GroupKFold por máquina | 43.21 | 34.21 | 78.46 | 35.71 |
| + afilado alpha=2 (descartado) | 42.69 | 33.98 | 75.28 | 38.48 |
| + z-score por máquina | 47.17 | 39.13 | 79.75 | 38.24 |
| + mezcla jerárquica w=0.2 | 47.49 | 39.53 | 80.10 | 37.99 |
| Oráculo (etiquetas exactas) | 73.96 | 62.80 | 100.0 | 100.0 |

La receta actual: LightGBM multiclase de 14 estados sobre las 46 features
originales más su z-score por `machine_id` (mediana y desvío de la propia
máquina), `StratifiedGroupKFold(5)` agrupando por `machine_id`,
`random_state=42`. Se envía `oof[:, 1:]` como probabilidades F01-F13.

Orden de trabajo por rentabilidad esperada:

1. **Features espectrales desde los NPZ.** BPFO, BPFI, BSF, bandas de
   envolvente. Es lo de mayor techo y lo único que ataca la confusión dentro de
   la familia mecánica, que es la más poblada.
2. **Features de dominio eléctricas.** Desequilibrio de tensión y de corriente
   entre fases: F08-F11 son las cuatro eléctricas y hoy se confunden entre sí.
3. **Features hidráulicas.** `delta_p` contra caudal, para separar cavitación
   (F12) de obstrucción (F13).
4. **Calibración.** `prob_score` ya da 80 y sube barato; pesa 20 %.
5. **Umbral de diagnóstico.** `macro_f1` corta en 0.5 fijo y hoy se pierde las
   fallas raras. Pesa 10 %, pero está casi entero sobre la mesa.
6. **Ensamble de semillas.** Último, y sólo si sobra tiempo.

**Lo descartado y por qué:**

- *Afilado o distorsión de probabilidades para forzar acciones:* medido, empeora
  el score. La capa de decisión ya está óptima (99.1 % de coincidencia).
- *13 clasificadores binarios independientes:* el train no tiene combos; el
  softmax de 14 estados aprovecha la restricción y da probabilidades coherentes.
- *Normalización por `session_id`:* empeora (43.21 -> 42.22). La sesión tiene
  una sola condición, así que normalizar por sesión borra la falla.
- *Mezcla jerárquica (modelo de 5 familias + reparto interno):* aporta +0.32,
  que está dentro del ruido entre folds. **Todavía no se acepta**: medir el
  desvío entre folds antes de subirla.

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
| 1 | apenas esté | receta actual (47.49 OOF): calibrar el gap OOF vs ranking |
| 2-3 | jueves tarde | features espectrales de los NPZ |
| 4-5 | jueves tarde | features eléctricas e hidráulicas de dominio |
| 6 | jueves 17:30 | cierre del día, red de seguridad |
| 7-8 | viernes | calibración y umbral de diagnóstico |
| 9 | viernes | ensamble final |
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

**Dataset chico.** 1750 filas y 46 features. Cada ronda de ajuste sobre el mismo
OOF gasta parte de la validación. Pocas rondas, cambios grandes, y el desvío
entre folds como juez: ninguna mejora se acepta si no lo supera.

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
