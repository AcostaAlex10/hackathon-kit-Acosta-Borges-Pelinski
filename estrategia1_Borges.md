# Estrategia 1 — Desafío Machine Health

Documento de contexto y estrategia para la Hackathon IA FI-UNaM (27-28/08/2026).
Equipo Acosta · Borges · Pelinski.

Sirve para dos cosas: como registro de lo que ya sabemos y como prompt de
arranque para una sesión nueva de Claude Code. Todas las cifras salen de
mediciones propias sobre `train_sup`, con la métrica oficial del kit de la
cátedra (`scoring.py`) y validación cruzada agrupada por `machine_id`.
Ninguna usó el test para entrenar, validar ni ajustar.

**Revisión 4**. Lo que cambió: el envío de v4 **bajó** el score (48,11 → 47,37)
y está diagnosticado por qué; y el techo de las señales crudas está cuantificado,
lo que corrige a la baja el descarte que veníamos arrastrando.

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

### 3.1 El estimador es el problema, no el modelo

Tres envíos al servidor: **28**, **43,42** y **48,1150**. El de 43,42 venía de
una configuración que daba ~48 en `GroupKFold`. Esa brecha define todo el
trabajo posterior.

`StratifiedGroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*.
El test es *"máquina nueva, régimen nuevo"*: sus 22 máquinas son todas distintas
a las 26 del train **y** operan en otro punto de trabajo (`rpm_mean` 1742→2052,
`delta_p_mean` 0,51→2,61, cinco veces mayor).

El reemplazo es entrenar con las máquinas lentas y validar con las rápidas. Pero
**con un solo corte ese estimador engaña**. Medido sobre cinco cortes
(k = 11…15 máquinas lentas a ajuste):

| Configuración | k=11 | k=12 | **k=13** | k=14 | k=15 | media | sd |
|---|---|---|---|---|---|---|---|
| Intento3 tal cual | 42,98 | 43,70 | **45,90** | 43,48 | 43,06 | 43,83 | 1,07 |
| sin duplicados | 44,19 | 43,34 | **43,82** | 41,75 | 43,62 | 43,34 | 0,84 |
| sin redundancia | 43,13 | 42,65 | **46,55** | 44,79 | 44,10 | 44,24 | 1,37 |
| + secuencia | 44,60 | 44,10 | **44,71** | 43,47 | 43,19 | 44,01 | 0,60 |

**El desvío del estimador es 1,0 y las diferencias entre configuraciones son
0,75.** El corte k=13, que era el único que usábamos, es el más optimista de los
cinco en dos de las cuatro configuraciones.

**Regla operativa:** el juez es el promedio de cinco cortes, y nada se acepta
por menos de ~2 puntos salvo que haya un argumento que no sea el score.

### 3.2 Qué sobrevivió del handoff

| Propuesta | Medido (5 cortes) | Veredicto |
|---|---|---|
| Limpiar redundancia (17 pares > 0,95) | 43,83 → 44,24 | se adopta **por parsimonia**: 174 → 142 columnas |
| Componentes de secuencia + factor de potencia | 43,83 → 44,01 | **descartada**: dentro del ruido |
| Modelo jerárquico familia → falla | 44,76 → 42,10 | **descartada**: empeora |
| Cobertura de combos (rama One-vs-Rest) | 44,76 → 45,21 | se adopta **como seguro**, no por el score |

### 3.3 Elección de ensamble

| Ensamble | GroupKFold | Régimen (5 cortes) |
|---|---|---|
| lgb+et | 54,06 | 44,89 ± 1,78 |
| **lgb+et+rf** | 54,06 | **44,76 ± 0,95** |
| lgb | 51,92 | 44,24 ± 1,37 |
| lgb+et+rf+lr | 53,05 | 42,84 ± 0,63 |
| lgb+rf+lr | 53,27 | 42,19 ± 1,62 |
| lr solo | 49,14 | 38,17 ± 1,52 |

Los cuatro punteros están empatados dentro del ruido. Lo que **sí** supera el
ruido: **toda combinación con regresión logística cae dos puntos**, porque
transfiere mal a régimen nuevo. Se elige `lgb+et+rf` por tener el desvío más
bajo del grupo empatado — criterio de estabilidad, no de score.

### 3.4 La regresión de v4, diagnosticada

El envío de v4 dio **47,3733** contra los **48,1150** de base4. Bajó. La causa
está aislada, midiendo cada cambio por separado sobre los cinco cortes con el
ensamble real (no con LightGBM solo, que fue mi error original):

| Configuración | final | cost | prob | diag |
|---|---|---|---|---|
| 174 columnas (base4) | **45,44** | 37,23 | 81,33 | 31,17 |
| 142 columnas (v4, sin redundancia) | **44,45** | 36,16 | 81,18 | 29,03 |
| 142 + sesión | 44,76 | 36,46 | 81,25 | 29,83 |
| 142 + sesión + OvR (= v4 enviado) | 45,21 | 36,69 | 81,30 | 32,64 |

**Quitar las 12 columnas redundantes cuesta un punto entero.** El error de
método: medí ese cambio con LightGBM solo y lo apliqué a un ensamble con
ExtraTrees y RandomForest. Con `max_features="sqrt"`, las copias correlacionadas
**aumentan la probabilidad de que una variable útil entre en cada split**: para
un ensamble de bagging la redundancia no es ruido, es muestreo. La parsimonia
que buscaba era una mejora para un modelo y un daño para otro.

Los otros dos cambios quedan exonerados:

- *Promedio por sesión:* neutro (+0,02 sobre 174 columnas). Además se verificó
  que las sesiones del **test** también son homogéneas, sin usar etiquetas:
  la fracción de ventanas en la clase modal predicha es 0,911 en test contra
  0,925 en train, y 0,321 barajando las sesiones como control.
- *Cobertura OvR:* positiva (+0,40 sobre 174 columnas).

**Receta corregida**, medida sobre los cinco cortes con las 174 columnas:

| Configuración | final | sd |
|---|---|---|
| lgb+et+rf (como base4) | 45,44 | 1,00 |
| lgb+et+rf + sesión | 45,46 | 1,11 |
| **lgb+et+rf + OvR 0,3** | **45,84** | **0,40** |
| lgb+et+rf + sesión + OvR 0,3 | 45,77 | 1,44 |

Es una mejora de +0,4 sobre base4: real pero chica. **No justifica un envío por
sí sola**; conviene juntarla con lo que salga de las señales crudas.

### 3.5 El techo de las señales crudas, cuantificado

Veníamos diciendo que el retorno de los NPZ era bajo porque sólo F03/F04/F05
son difíciles y ésas pesan en `diag_score`, que es el 10 %. **Eso está mal**, y
el error es mío por repetirlo sin medirlo. Simulando sobre el OOF:

| Escenario | final | cost | prob | diag |
|---|---|---|---|---|
| actual | 52,07 | 43,97 | 83,49 | 45,92 |
| + intra-familia perfecta (rodamientos) | **56,47** | 43,97 | 92,30 | 72,33 |
| + familia perfecta | **64,84** | 59,03 | 87,12 | 60,98 |

Resolver los rodamientos vale **+4,4 puntos**, no ~1,7. El razonamiento viejo
sólo contaba `diag_score` y se olvidaba de que concentrar la masa en la falla
correcta **también mejora el Brier**: `prob_score` sube de 83,5 a 92,3, y eso
pesa 20 %.

Acertar la familia sigue valiendo más (**+12,8**), y es donde `cost_score`
tiene los 25 puntos que le faltan para el techo de 62,8.

### 3.6 Lo descartado y por qué

- *Afilado de probabilidades:* la capa de decisión de la métrica ya es óptima
  (99,1 % de coincidencia con la acción óptima bajo nuestro posterior).
- *13 binarios independientes como modelo principal:* el train no tiene combos;
  el softmax aprovecha la restricción. Sí se usan como rama de cobertura.
- *Normalización por `session_id`:* borra la falla, que es constante en la sesión.
- *Rango percentil por máquina:* 51,9 en `GroupKFold` y 41,8 bajo régimen.
- *Búsqueda de algoritmo y tuneo de hiperparámetros:* mueven menos que el ruido.
- *Componentes de secuencia, modelo jerárquico:* medidos arriba, no pasan.
- *Decidir con un solo corte de régimen:* medido, el corte k=13 es el más
  optimista en la mitad de los casos.

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
| 1-3 | ya hechos | 28, 43,42 y **48,1150**. Calibraron la brecha estimador↔servidor |
| 4 | próximo | v4: redundancia limpia + cobertura OvR (estimado 47-49) |
| 5-6 | jueves tarde | calibración; más cortes de régimen |
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

**Todo estimador que tenemos es ruidoso, y ahora sabemos cuánto.** El n
efectivo es de 250 sesiones, no 1750 ventanas. `GroupKFold` mueve hasta 1,5
puntos con sólo cambiar la semilla; el corte por régimen tiene desvío 1,0 entre
cortes. **Ninguna mejora menor a ~2 puntos se acepta sin promediar cinco
cortes.** Ya nos costó dos veces: primero eligiendo `lgb+rf+lr` con
`GroupKFold`, después leyendo el corte k=13 como si fuera el valor verdadero.

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
