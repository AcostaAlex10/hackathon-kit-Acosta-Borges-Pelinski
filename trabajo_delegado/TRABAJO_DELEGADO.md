# Trabajo delegado — Machine Health

Handoff para una sesión nueva. **Leer esto entero antes de tocar nada.** Todo lo de
acá está medido y ejecutado, no estimado. No re-derivar.

---

## 0. Lo primero al retomar

Los datos y el kit oficial **no están en el repo** (van en `.gitignore` y el contenedor
es efímero). Pedirle al usuario que suba de nuevo:

- `datos_sinraw.zip` → `train_sup`, `test`, `train_unlabeled`, `raw_signal_index`,
  `data_dictionary.csv` (CSV y Parquet)
- `participant_kit.zip` → **`scoring.py`** (crítico), `baseline.py`, notebooks del kit

Reconstrucción:
```bash
pip install -q pandas numpy scikit-learn pyarrow scipy lightgbm
# descomprimir en work/desafio/ como:
#   work/desafio/datos_zip/datos_sinraw/*
#   work/desafio/kit_zip/participant_kit/*
cd work/desafio && python3 generar_submit.py     # ~2 min, regenera el submit
```

Scripts ya escritos en `src/` (rama `Acosta`):

| Script | Qué hace |
|---|---|
| `generar_submit.py` | entrena y escribe `submit.csv`. Reproducible byte a byte |
| `fastscore.py` | scorer vectorizado, **verificado contra el oficial (diferencia 0)**. El scorer oficial es lentísimo en Python puro; usar este para iterar |
| `comparar.py` | compara bloques de features en los dos esquemas de validación |
| `eda_ana.py` | EDA dirigido (discriminación por familia, huella de máquina, redundancia) |
| `build_nb.py` | genera el notebook |

---

## 1. Estado actual

- Envíos hechos: uno dio **28**, otro dio **43,42**. Puntero (en su momento): **42**.
- Submit propuesto sin enviar: `submits/submit_fisicas_z_lgb_et.csv`.
  Estimación conservadora **~45**.
- Notebook ejecutado: `trabajo_delegado/solucion_machine_health.ipynb`
  (14/14 celdas con salida, cero errores).

**Configuración actual:** features físicas + z-score por máquina (174 columnas),
ensamble LightGBM + ExtraTrees, promedio geométrico por sesión.

---

## 2. Los cuatro hechos que gobiernan el problema

1. **`is_combo = 0` en las 1750 filas.** El problema declarado como multilabel es, en
   el train, **multiclase de 14 estados excluyentes**. Se modela con softmax.
2. **Train y test no comparten ninguna máquina** (26 vs 22, intersección vacía) **y el
   régimen cambia**: `rpm_mean` 1742 → 2052, `Tamb` 26,8 → 29,9.
3. **La etiqueta es constante dentro de cada sesión** (250 de 250), y cada sesión tiene
   exactamente 7 ventanas. **El n efectivo es 250 sesiones, no 1750 ventanas.** Con esa
   cifra, ajustar muchos parámetros contra la métrica es sobreajuste garantizado.
4. **El 70 % del score se juega en acertar la FAMILIA** (5 clases), no la falla exacta.
   Distinguir F03 de F04 no aporta un centavo al `cost_score`; sólo al `diag_score`,
   que pesa 10 %.

---

## 3. El esquema de validación (lo más importante)

`GroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*. El test es
*"máquina nueva, **régimen nuevo**"*. **Hay que validar con los dos esquemas.**

El segundo: ordenar las 26 máquinas por `rpm_mean`, entrenar con las 13 lentas
(1437 rpm) y validar con las 13 rápidas (2057 rpm ≈ los 2052 del test).

**Brecha calibrada: ~4,6 puntos.** Una configuración con 48,0 en `GroupKFold` obtuvo
43,42 en el servidor. Sirve para estimar cualquier configuración nueva.

| Features | GroupKFold (3 sem.) | Cambio de régimen |
|---|---|---|
| Crudas | 44,11 | 38,09 |
| Crudas + z-score | 47,55 | 40,22 |
| Físicas | 49,82 | **45,35** |
| Físicas + z-score ← actual | **52,28** | 44,53 |
| Físicas + rango percentil | 51,88 | 41,80 |

| Ensamble (sobre físicas + z) | GroupKFold | Cambio de régimen |
|---|---|---|
| **lgb + et** ← elegido | 53,53 | **45,33** |
| lgb + rf + lr | **54,31** | 43,07 |
| lgb + et + rf + lr | 53,98 | 43,48 |

Con promedio por sesión: **54,53 / 45,74**.

> **Dos trampas confirmadas.** El rango percentil por máquina y el ensamble
> `lgb+rf+lr` son de lo mejor en `GroupKFold` y de lo peor bajo cambio de régimen.
> **Nunca decidir sólo con `GroupKFold`.**

---

## 4. Hallazgos nuevos de la última sesión

### 4.1 El z-score no borra la huella de máquina

Accuracy prediciendo `machine_id` (26 clases, azar = 0,038):

| Features | Accuracy |
|---|---|
| Crudas | **1,000** |
| Físicas | **1,000** |
| **Sólo columnas z-score** | **1,000** |

Cada ventana lleva una huella perfecta de su máquina y el z-score **sólo recentra**:
la forma de la distribución sigue identificando la máquina. Por eso da +2,5 en
`GroupKFold` y no mejora la transferencia. Las que más identifican la máquina:
`voltage_a__rms`, `voltage_c__rms`, `V_rms_mean`, `delta_p_mean` — constantes de
instalación, no estado de salud.

### 4.2 Los rodamientos son el único agujero real

Separabilidad intra-familia (AUC de la mejor feature sola):

| Par | AUC |
|---|---|
| F01 vs F02 (desbalance / desalineación) | **1,000** |
| F12 vs F13 (cavitación / obstrucción) | 0,920 |
| F08 vs F09 (desequilibrio / pérdida fase) | 0,896 |
| F10 vs F11 (barras / espiras) | 0,882 |
| **F03 vs F05** (pista ext. / elemento rodante) | **0,714** |
| **F03 vs F04** (pista ext. / pista int.) | **0,695** |

Casi todo está resuelto. **Sólo F03/F04/F05 son difíciles**, que es donde harían falta
las frecuencias BPFO/BPFI/BSF de las señales crudas. Como esas tres sólo pesan en el
`diag_score` (10 %), **el retorno de procesar los NPZ es bajo**. Esto invierte la
prioridad que se venía asumiendo.

### 4.3 Lo eléctrico está subaprovechado

Discriminación por familia, medida **sólo entre ventanas con falla**:

| Familia | Mejor discriminador | AUC |
|---|---|---|
| Estructural | `acc_axial__crest` | 0,916 |
| Eléctrica | `acc_radial_a__crest` (¡vibración!) | 0,821 |
| Hidráulica | `rad_b_a` | 0,770 |
| Mecánica | `acc_radial_a__rms` | 0,691 |

**Las fallas eléctricas se detectan mejor por vibración que por features eléctricas.**
Tiene sentido físico, pero indica margen sin explotar.

---

## 5. Defectos conocidos del notebook — NO están arreglados

El notebook adjunto **corre y es reproducible**, pero tiene esto pendiente:

1. **`acc_*__crest_n` es copia exacta de `acc_*__crest`** (correlación 1,000). Se
   olvidó normalizarlo, y el crest factor ya es adimensional: la columna sobra.
   Son 3 columnas duplicadas.
2. **`S_ap` correlaciona 1,000 con `I_rms_mean`** (la tensión es casi constante). Sobra.
3. **24 pares con correlación > 0,95** en total. Hay para limpiar.
4. **35 de 87 features** tienen |AUC−0,5| < 0,02 contra sano/falla (todas las
   `*__nan_frac`, tensiones y corrientes crudas por fase). Ojo: pueden servir igual
   para separar familias; **medir antes de tirar**.
5. El texto del notebook cita números del ensamble en unas tablas y de LightGBM solo en
   otras. Está aclarado, pero conviene unificar.

---

## 6. Qué conviene hacer, en orden

1. **Limpiar la redundancia** (defectos 1–3 de arriba). Barato, reduce varianza.
   Verificar con los dos esquemas que no baje.
2. **Explotar lo eléctrico** (§4.3). Faltan las **componentes de secuencia**
   (positiva / negativa / cero), que son el indicador estándar de desequilibrio
   trifásico, y el **factor de potencia**. Es la palanca con mejor relación
   esfuerzo/retorno que queda.
3. **Modelo jerárquico familia → falla.** Primero las 5 familias (que es el 70 % del
   score), después la falla dentro de la familia. Alinea el modelo con la métrica y
   ataca directamente el cuello de botella (exactitud de familia ≈ 0,63).
4. **Cobertura ante combos.** La consigna dice que existen fallas combinadas pero el
   train no tiene ninguna. Si el test las tiene, el softmax las subestima y el
   `cost_score` cae. Cobertura barata: mezclar el softmax con una rama One-vs-Rest de
   13 binarios, con el peso elegido por CV agrupada. Se puede sondear sin etiquetas
   mirando la distribución de `Σp` y la entropía sobre `train_unlabeled`.
5. **NPZ sólo al final**, sabiendo que mueve el 10 % del score (§4.2).

---

## 7. Descartes ya medidos — no repetirlos

| Descarte | Evidencia |
|---|---|
| Rango percentil por máquina | 51,9 en `GroupKFold` y 41,8 bajo cambio de régimen |
| Ensamble `lgb+rf+lr` | 54,31 / 43,07. La regresión logística transfiere pésimo (39,12) |
| Afilado de probabilidades contra la métrica | +0,31 contra desvío entre folds de 1,7 |
| Encogimiento hacia la prevalencia | efecto mixto y menor al ruido |
| Features agregadas de sesión como columnas | empeoran |
| Curado de datos | no hay nada que curar: cero nulos, cero duplicados, sin centinelas |
| Tuneo de hiperparámetros | el cambio de familia de modelo mueve menos que el ruido de partición |

---

## 8. Reglas duras

1. **Nunca decidir sólo con `GroupKFold`.** Siempre reportar los dos esquemas.
2. **Nunca** entrenar, ajustar ni calibrar con el test. Nunca etiquetar test a mano.
3. **Reproducibilidad:** `ExtraTrees` y `RandomForest` con `n_jobs=-1` **no son
   deterministas** aunque tengan `random_state` fijo (comprobado: `n_jobs=1` da el
   mismo hash, `n_jobs=2` no). Usar `n_jobs=1` y, en LightGBM,
   `deterministic=True, force_col_wise=True, n_jobs=1`. La consigna pide el código que
   replica la entrega.
4. **Verificar el submit** con `validate_prediction_package` antes de subir.
5. Semillas en 42. Reportar los resultados como son: si el CV empeoró, se dice.
6. Estilo del probatorio: español, sin emojis, sin citar la consigna, conciso. Se
   califica el análisis y la justificación, no el código.

---

## 9. Material para el probatorio

Dos hallazgos defendibles que ya están documentados:

- **El kit de la cátedra se equivoca.** `baseline.ipynb` dice que P0 elige MONITOR;
  en realidad elige **INSPECT mecánica en las 1848 ventanas**
  (`4 + 8·0,412 = 7,30 < 8,76`). Por eso el baseline da 29 y no 15: para superarlo hay
  que acertar la familia más del 46 % de las veces.
- **El techo real es 73,96, no 100.** Aun con predicción perfecta hay que pagar la
  inspección de $4, así que el `cost_score` máximo es 62,8. Comparar contra 100 es
  engañoso.

---

## 10. Cabos sueltos del repo

- Quedó un archivo con el nombre roto: `    /notebooks  /solucion_machine_health.ipynb`
  (espacios al principio y en el medio), artefacto de una subida desde Colab. Hay otro
  parecido con `eda_y_benchmark_vinculo_con_drive.ipynb`, duplicado. Conviene borrarlos.
- El `README.md` de `main` y el de `Acosta` **divergieron**: el de `main` borra la
  descripción del kit y varias referencias. Alguien debería unificarlos.
- `notebooks/solucion_machine_health.ipynb` en `Acosta` fue reemplazado por una versión
  de Colab. La copia verificada y reproducible es la de esta carpeta.
