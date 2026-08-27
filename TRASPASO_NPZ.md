# Traspaso — fase de señales crudas (NPZ)

Handoff para una sesión nueva de Claude Code. **Leer entero antes de tocar
nada.** Todo lo de acá está medido y ejecutado, no estimado. No re-derivar.

Rama de trabajo: `Borges`. Último commit al escribir esto: `0339de7`.

---

## 0. Lo primero al retomar

El contenedor es efímero y los datos no van al repo (`.gitignore`). Pedirle al
usuario que suba:

| Archivo | Qué trae | Estado |
|---|---|---|
| `datos_sinraw.zip` | `train_sup`, `test`, `train_unlabeled`, `raw_signal_index`, `data_dictionary` (CSV y Parquet) | ya lo tuvimos |
| `participant_kit.zip` | **`scoring.py`** (crítico), `baseline.py`, notebooks del kit | ya lo tuvimos |
| **`raw_signals/`** | **los NPZ. NUNCA los tuvimos.** Es lo que falta para esta fase | **falta** |

Los NPZ son un archivo por sesión: `raw_signals/S<hash>.npz`, con claves
internas `window_id__channel`. `raw_signal_index` mapea cada `window_id` a su
archivo y canales. Son 4032 ventanas (1750 train + 1848 test + 434 unlabeled),
7 ventanas por sesión.

Canales y frecuencias, que **no coinciden entre sí**:

| Canal | fs |
|---|---|
| `acc_radial_a` | 20 kHz |
| `current_a` | 10 kHz |
| `pressure_in` | 200 Hz |

Si el volumen es un problema, **con 30-40 sesiones etiquetadas alcanza** para
validar que la extracción sirve antes de correrla entera.

Reconstrucción del entorno:

```bash
pip install -q pandas numpy scikit-learn pyarrow scipy lightgbm
# datos_sinraw/ y participant_kit/ al lado del notebook
```

---

## 1. Estado de la competencia

Envíos hechos y su score real en el servidor:

| # | Qué era | Score |
|---|---|---|
| 1 | primer intento | 28 |
| 2 | crudas + z-score, ensamble | 43,42 |
| 3 | **`base4.ipynb`: físicas + z (174 col), ensamble** | **48,1150** ← mejor |
| 4 | `machine_health_v4.ipynb` | 47,3733 ← bajó |

**El notebook vigente es `base4.ipynb`** (el que dio 48,1150). `machine_health_v4.ipynb`
está en el repo pero **no se usa**: ver §4.

---

## 2. Los hechos que gobiernan el problema

No re-derivar ninguno.

1. **`is_combo = 0` en las 1750 filas.** El problema declarado multilabel es, en
   el train, **multiclase de 14 estados excluyentes**. Se modela con softmax.
2. **La etiqueta es constante dentro de cada sesión** (250 de 250) y cada sesión
   tiene exactamente 7 ventanas. **El n efectivo es 250, no 1750.**
   Verificado también en el **test**, sin usar etiquetas: la fracción de
   ventanas en la clase modal predicha es 0,911 en test contra 0,925 en train y
   0,321 barajando las sesiones como control.
3. **Train y test no comparten ninguna máquina** (26 vs 22, intersección vacía)
   **y el régimen cambia**: `rpm_mean` 1742 → 2052, `delta_p_mean` 0,51 → 2,61
   (cinco veces), `Tamb` 26,8 → 29,9.
4. **El 70 % del score se juega en la FAMILIA** (5 clases: sano, mecánica,
   estructural, eléctrica, hidráulica), no en la falla exacta. Confundir F03 con
   F04 no cambia `cost_score`: misma familia, misma acción INSPECT.
5. **El techo real es 73,96, no 100.** Aun con predicción perfecta hay que pagar
   la inspección de $4, así que `cost_score` máximo es 62,8.
6. **El baseline de la cátedra no es pasivo.** `baseline.ipynb` dice que P0
   elige MONITOR; en realidad elige **INSPECT mecánica en las 1848 ventanas**
   (`4 + 8·0,412 = 7,30 < 8,76`). Para superarlo hay que acertar la familia más
   del 46 % de las veces. Es material defendible para el probatorio.
7. **La capa de decisión de la métrica ya es óptima.** Su `choose_action`
   coincide en el 99,1 % de las ventanas con la acción óptima bajo nuestro
   propio posterior. No hay nada que ganar distorsionando probabilidades.

---

## 3. Protocolo de validación — LO MÁS IMPORTANTE

`StratifiedGroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*.
El test es *"máquina nueva, **régimen nuevo**"*. Decidir con `GroupKFold` ya nos
costó dos veces.

El reemplazo: ordenar las 26 máquinas por `rpm_mean`, entrenar con las lentas y
validar con las rápidas. **Pero con un solo corte ese estimador engaña.** Medido
sobre cinco cortes (k = 11…15 máquinas lentas a ajuste):

| Configuración | k=11 | k=12 | **k=13** | k=14 | k=15 | media | sd |
|---|---|---|---|---|---|---|---|
| base4 tal cual | 42,98 | 43,70 | **45,90** | 43,48 | 43,06 | 43,83 | 1,07 |
| sin duplicados | 44,19 | 43,34 | **43,82** | 41,75 | 43,62 | 43,34 | 0,84 |
| sin redundancia | 43,13 | 42,65 | **46,55** | 44,79 | 44,10 | 44,24 | 1,37 |
| + secuencia | 44,60 | 44,10 | **44,71** | 43,47 | 43,19 | 44,01 | 0,60 |

**El desvío del estimador es 1,0 y las diferencias entre configuraciones son
0,75.** El corte k=13, que era el único que se usaba, es el más optimista de los
cinco en dos de las cuatro configuraciones.

### Reglas que salen de esto

1. **El juez es el promedio de cinco cortes**, nunca uno solo, nunca `GroupKFold`.
2. **Nada se acepta por menos de ~2 puntos**, salvo que haya un argumento que no
   sea el score (ver la cobertura OvR en §4).
3. **Medir con el ensamble que se va a enviar**, no con un modelo suelto. Este
   error causó la regresión de v4 (§4).
4. Reportar siempre media ± sd, no el número pelado.

```python
rpm_maq = train.groupby("machine_id").rpm_mean.mean().sort_values()
CORTES = [11, 12, 13, 14, 15]
def particion_regimen(k):
    rapidas = set(rpm_maq.index[k:])
    es_rap = train.machine_id.isin(rapidas).to_numpy()
    return np.where(~es_rap)[0], np.where(es_rap)[0]
```

### Calibración estimador ↔ servidor

base4 da **43,83** en este esquema y sacó **48,1150** en el servidor: el
estimador subestima en unos **+4 puntos** y de forma consistente. Sirve para
proyectar, no para prometer.

---

## 4. La regresión de v4 — no repetir el error

v4 dio 47,3733 contra 48,1150. Causa aislada midiendo cada cambio por separado
**con el ensamble real**:

| Configuración | final | cost | prob | diag |
|---|---|---|---|---|
| 174 columnas (base4) | **45,44** | 37,23 | 81,33 | 31,17 |
| 142 columnas (v4, sin redundancia) | **44,45** | 36,16 | 81,18 | 29,03 |
| 142 + sesión | 44,76 | 36,46 | 81,25 | 29,83 |
| 142 + sesión + OvR (= v4 enviado) | 45,21 | 36,69 | 81,30 | 32,64 |

**Quitar las 12 columnas redundantes (correlación > 0,95) cuesta un punto.** Se
midió con LightGBM solo y se aplicó a un ensamble con ExtraTrees y RandomForest:
con `max_features="sqrt"` las copias correlacionadas **aumentan la probabilidad
de que una variable útil entre en cada split**. Para bagging la redundancia no
es ruido, es muestreo.

**No volver a limpiar features por correlación** en este ensamble.

Los otros dos cambios de v4 quedan exonerados y **conviene conservarlos**:

| Configuración (174 columnas) | final | sd |
|---|---|---|
| lgb+et+rf (como base4) | 45,44 | 1,00 |
| lgb+et+rf + sesión | 45,46 | 1,11 |
| **lgb+et+rf + OvR 0,3** | **45,84** | **0,40** |
| lgb+et+rf + sesión + OvR 0,3 | 45,77 | 1,44 |

La **cobertura OvR** mezcla el softmax con una rama One-vs-Rest de 13 binarios:
`P[:, 1:] = 0.7 * P[:, 1:] + 0.3 * B`. Se adopta **como seguro**, no por el
score: cubre el riesgo de que el test tenga fallas combinadas (la consigna las
describe, el train no tiene ninguna), y está medido que no cuesta nada.

---

## 5. Dónde está el score — el mapa que ordena la fase NPZ

Simulado sobre el OOF, reemplazando partes de la predicción por la verdad:

| Escenario | final | cost | prob | diag |
|---|---|---|---|---|
| actual | 52,07 | 43,97 | 83,49 | 45,92 |
| + intra-familia perfecta (rodamientos) | **56,47** | 43,97 | 92,30 | 72,33 |
| + familia perfecta | **64,84** | 59,03 | 87,12 | 60,98 |

**Corrección importante:** veníamos diciendo que resolver F03/F04/F05 valía poco
porque sólo pesa en `diag_score` (10 %). **Es falso.** Vale **+4,4 puntos**,
porque concentrar la masa en la falla correcta también mejora el Brier:
`prob_score` sube de 83,5 a 92,3 y eso pesa 20 %.

Acertar la familia vale **+12,8** y es donde `cost_score` tiene los 25 puntos
que le faltan para su techo de 62,8. La accuracy de familia actual es ~0,68.

**Los dos objetivos de la fase NPZ, en orden de valor:**

1. **Mejorar la discriminación de FAMILIA** (techo +12,8).
2. **Separar F03/F04/F05 entre sí** (techo +4,4).

---

## 6. Qué extraer de los NPZ, y qué NO

### Lo que NO hay que hacer

**No recalcular desde la señal lo que la tabla ya tiene.** La tabla trae por
canal: `rms`, `kurtosis`, `crest`, `1x`, `2x`. Recalcularlos da el mismo escalar
y no aporta nada. En particular:

- La FFT para 1x y 2x **ya está**: F01 vs F02 separa con **AUC 1,000**.
- `log1p(kurtosis)` y crest **ya están** como columnas, y **no mueven
  F03/F04/F05**: esas se separan por el espectro de **envolvente**, que es otra
  cuenta.

**No hacer deep learning.** Con 250 muestras efectivas, tres frecuencias de
muestreo distintas y covariate shift entre regímenes, una 1D-CNN o un
Transformer van a memorizar las 26 máquinas. El ensamble de árboles sobre
features físicas ya transfiere bien. Esto está decidido, no re-discutir.

### Lo que sí, y qué ataca cada cosa

| Feature | Cómo | Qué ataca | Objetivo |
|---|---|---|---|
| **Espectro de envolvente** en la banda resonante | filtrar la banda de mayor kurtosis, Hilbert, PSD de la envolvente, energía relativa en BPFO/BPFI/BSF·fr | F03/F04/F05 entre sí | +4,4 |
| **MCSA**: bandas laterales de corriente | PSD de `current_a`, picos en `f(1±2s)` para varios deslizamientos | F10 barras rotóricas, F11 espiras → familia **eléctrica** | familia |
| **Espectro de presión** | fracción de energía en alta frecuencia, kurtosis | F12 cavitación → familia **hidráulica** | familia |
| **Entropía espectral** por canal | dispersión del espectro | sano vs falla | familia |

Todo esto está implementado en **`raw_feats.py`** (rama `Borges`).

### Estado de `raw_feats.py`

**Validado** contra señales sintéticas con defecto conocido: se fabricó un
rodamiento con defecto en pista exterior y el detector da un pico BPFO **190
veces** mayor que en la señal sana, sin disparar BPFI.

**No validado** contra los NPZ reales — nunca los tuvimos.

**Limitación conocida y probablemente el primer problema a resolver:** los
ratios de frecuencia de defecto dependen de la **geometría del rodamiento**, que
no tenemos. `raw_feats.RATIOS` usa valores típicos (BPFO ≈ 3,585·fr, BPFI ≈
5,415·fr, BSF ≈ 2,357·fr). Si la cátedra simuló otra geometría, los picos caen
en el lugar equivocado y las features no sirven.

**Variante robusta, a usar si la primera no rinde:** en vez de tres números,
barrer el espectro de envolvente en órdenes de 2 a 8 (paso 0,1) y darle al
modelo el perfil completo. Que el árbol encuentre dónde está el pico en lugar de
adivinar la geometría. Es más columnas pero no depende de un supuesto.

### Orden de trabajo sugerido

1. Cargar un NPZ y **mirar la señal**: longitud, fs real, si hay NaN, si la
   ventana es continua. Antes de extraer nada.
2. Correr `raw_feats` sobre **30-40 sesiones etiquetadas** y medir si las
   features de envolvente separan F03/F04/F05 (AUC por par). **Si no separan,
   pasar a la variante robusta antes de procesar las 4032 ventanas.**
3. Extraer sobre todo el dataset. Guardar en Parquet: es caro y no se quiere
   repetir.
4. Concatenar a las 174 columnas de base4, **sin quitar nada**.
5. Medir con los cinco cortes, con el ensamble `lgb+et+rf` + OvR 0,3.
6. Aceptar sólo si supera ~2 puntos.

---

## 7. Reglas duras

1. **Nunca decidir con `GroupKFold` ni con un solo corte de régimen.**
2. **Nunca** entrenar, ajustar ni calibrar con el test. Nunca etiquetar a mano.
3. **Reproducibilidad (art. 5):** `ExtraTrees` y `RandomForest` con `n_jobs=-1`
   **no son deterministas** aunque tengan `random_state` fijo. Usar `n_jobs=1`
   y, en LightGBM, `deterministic=True, force_col_wise=True, n_jobs=1`. La
   consigna pide el código que replica la entrega.
4. **Verificar el submit** con `validate_prediction_package` antes de subir.
5. Semillas en 42.
6. **Reportar los resultados como son.** Si el CV empeoró, se dice. Este
   documento existe porque dos veces se aceptó una mejora que no lo era.
7. Estilo del probatorio: español, sin emojis, sin citar la consigna, conciso.
   Se califica el análisis y la justificación, no el código.

---

## 8. Descartes ya medidos — no repetirlos

| Descarte | Evidencia |
|---|---|
| Limpiar features por correlación > 0,95 | 45,44 → 44,45. Rompe el bagging (§4) |
| Componentes de secuencia + factor de potencia | 43,83 → 44,01, dentro del ruido |
| Modelo jerárquico familia → falla | 44,76 → 42,10, empeora |
| Ensamble con regresión logística | lgb+et+rf 44,76 vs +lr 42,84; lr sola transfiere 38,17 |
| Rango percentil por máquina | 51,9 en GroupKFold y 41,8 bajo régimen |
| Afilado de probabilidades contra la métrica | +0,31 contra desvío de 1,7; la capa de decisión ya es óptima |
| Normalización por `session_id` | 43,21 → 42,22: la sesión tiene una sola condición, normalizar borra la falla |
| Encogimiento hacia la prevalencia | efecto mixto y menor al ruido |
| Features agregadas de sesión como columnas | empeoran |
| Búsqueda de algoritmo y tuneo de hiperparámetros | mueven menos que el ruido de partición |
| Curado de datos | no hay nada que curar: cero nulos, cero duplicados, sin centinelas |
| Deep learning sobre la señal cruda | ver §6 |

---

## 9. Archivos en la rama `Borges`

| Archivo | Qué es |
|---|---|
| `estrategia1_Borges.md` | estrategia completa, revisión 4. Leer después de esto |
| `raw_feats.py` | extracción de señal cruda, validada en sintético |
| `machine_health_v3.ipynb` | features físicas + doble validación. Ejecutado |
| `machine_health_v4.ipynb` | **no usar**: es el que bajó a 47,37 (§4) |
| `eda_y_benchmark.ipynb` | benchmark multimodelo de la mañana |
| `TRASPASO_NPZ.md` | este documento |

`base4.ipynb` —el que dio 48,1150— **lo tiene el usuario, no está en el repo.**
Pedírselo al arrancar.

---

## 10. Material para el probatorio

Ya documentado y defendible:

- **El kit de la cátedra se equivoca** sobre qué acción elige su propio
  baseline (§2.6).
- **El techo real es 73,96, no 100** (§2.5).
- **El estimador de validación fue el hallazgo central**: tres esquemas
  sucesivos, cada uno corrigiendo un sesgo del anterior, con el ruido medido.
  Vale más que cualquier feature.
- **Dos regresiones diagnosticadas con evidencia**: elegir ensamble por
  `GroupKFold` y limpiar redundancia en un ensamble de bagging. El descarte con
  evidencia también es resultado.
