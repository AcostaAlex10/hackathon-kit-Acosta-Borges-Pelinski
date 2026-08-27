# ¿Conviene gastar la tercera bala? — análisis para el equipo

**Contexto.** Tenemos dos envíos hechos: uno dio **28** (por debajo del baseline de
29) y el de Ana dio **43,42**. El puntero estaba en 42. La pregunta es si el submit
que preparamos ahora mejora lo de Ana lo suficiente como para gastar un envío.

**Respuesta corta: sí, subilo.** Estimación **~45**, es decir **+1,5 a +2** sobre
43,42. La ventaja viene de las features físicas, no de la normalización.

Todos los números de abajo salen del notebook `solucion_machine_health.ipynb`
ejecutado de punta a punta, no de estimaciones.

---

## 1. Lo primero: nuestro CV estaba mintiendo, y ahora sabemos cuánto

`GroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*. Pero el test es
*"máquina nueva, **régimen nuevo**"*: las 22 máquinas de test son todas distintas a
las 26 de train (intersección vacía) **y** operan en otro punto de trabajo.

| Variable | train | test |
|---|---|---|
| `rpm_mean` | 1742 | **2052** |
| `flow_mean` | 44,8 | 50,1 |
| `Tamb` | 26,8 | 29,9 |
| `delta_p_mean` | 0,5 | 2,6 |

Por eso se agregó una **segunda validación**: se ordenan las 26 máquinas por rpm, se
entrena con las 13 lentas (1437 rpm) y se valida con las 13 rápidas (2057 rpm ≈ los
2052 del test). Es la que reproduce la relación real entre train y test.

**La brecha medida:** el enfoque de Ana estima ~48 en `GroupKFold` y obtuvo **43,42**
en el servidor. Esa diferencia de ~4,6 puntos es el precio de validar con el esquema
equivocado, y es lo que nos permite estimar cualquier otra configuración.

---

## 2. Comparación de enfoques, mismo protocolo

Todo con LightGBM, `StratifiedGroupKFold(5)` con 3 semillas para la columna izquierda.

| Features | GroupKFold | Cambio de régimen |
|---|---|---|
| Crudas | 44,11 | 38,09 |
| Crudas + z-score por máquina ← **enfoque de Ana** | 47,55 | 40,22 |
| Físicas | 49,82 | **45,35** |
| **Físicas + z-score** ← **lo que proponemos** | **52,28** | 44,53 |
| Físicas + rango percentil por máquina | 51,88 | 41,80 |

### Tres conclusiones

**1. Las features físicas son la palanca real.** Ganan en los dos esquemas contra
cualquier alternativa que no las tenga: **+5,1 bajo cambio de régimen** sobre el
enfoque de Ana. Ana y nosotros estábamos atacando cosas distintas, y **se suman**.

**2. El rango percentil por máquina es una trampa de validación.** Es de lo mejor en
`GroupKFold` (51,88) y de lo peor bajo cambio de régimen (41,80). Destruye la
magnitud, que es justamente lo que aportan las features físicas. **Si alguien lo
prueba y ve el número alto en `GroupKFold`, que no se entusiasme.**

**3. El z-score por máquina NO mejora la transferencia a máquinas nuevas.** Esto
corrige una afirmación anterior mía. El orden entre `físicas` y `físicas + z-score`
bajo cambio de régimen **se da vuelta según el modelo**:

- con LightGBM solo: `físicas` 45,35 contra `físicas+z` 44,53
- con el ensamble: `físicas` 44,76 contra `físicas+z` 45,33

Menos de un punto y cambia de signo: está dentro del ruido. El z-score sí da una
ventaja consistente en `GroupKFold` (+2,5), y por eso se conserva, pero **no hay que
venderlo como que ayuda a generalizar a máquinas nuevas**.

---

## 3. Por qué las features físicas funcionan

No es fuerza bruta: cada bloque apunta a un modo de falla y está construido para ser
**adimensional o relativo**, que es la condición para transferir entre regímenes.

- **Vibración normalizada por régimen.** La energía vibratoria escala con el cuadrado
  de la velocidad, de ahí `rms/(rpm/1000)²`. Sin esto el modelo confunde *máquina
  rápida* con *máquina en falla* — que es exactamente el shift de la sección 1.
  La fracción de energía en `1x` marca **desbalance (F01)**; `2x/1x`, **desalineación
  (F02)**, porque la desalineación excita el segundo armónico. `log1p(kurtosis)` y
  `crest` capturan los impactos de **rodamientos (F03–F05)**. La asimetría entre
  apoyos marca **pérdida de rigidez (F07)**.
- **Eléctricas (F08–F11).** Desbalance de tensión y corriente entre fases como
  `(max−min)/media` para **F08**; `I_min/I_media` para **pérdida de fase (F09)**;
  corriente relativa a la velocidad para **barras rotóricas (F10)**.
- **Térmicas.** Siempre como diferencia contra `Tamb`, nunca en valor absoluto,
  porque el ambiente del test es 3 °C más cálido. Sobretemperatura de devanado →
  **cortocircuito entre espiras (F11)**; diferencia entre rodamientos →
  **lubricación deficiente (F06)**.
- **Hidráulicas (F12–F13).** Leyes de afinidad de bombas: altura adimensional
  `Δp/(rpm/1000)²` y coeficiente de caudal `flow/rpm`. El rendimiento aparente
  `flow·Δp/(V·I)` marca **obstrucción o fuga (F13)**.

---

## 4. Configuración final elegida

Features físicas + z-score por máquina (174 columnas), ensamble **LightGBM +
ExtraTrees**, promedio geométrico por sesión.

| Ensamble | GroupKFold | Cambio de régimen |
|---|---|---|
| **lgb + et** | 53,53 | **45,33** |
| lgb + rf + lr | 54,31 | 43,07 |
| lgb + et + rf + lr | 53,98 | 43,48 |
| lgb + et + lr | 54,02 | 42,83 |

Con promedio por sesión: **54,53 / 45,74**.

> Nota importante: `lgb+rf+lr` es el mejor en `GroupKFold` (54,31) y de los peores
> bajo cambio de régimen (43,07). Es el mismo tipo de trampa que el rango percentil.

**El promedio por sesión** se apoya en un hecho estructural verificado: cada sesión
tiene exactamente 7 ventanas y **la etiqueta es constante en las 250 sesiones**. O sea
que el tamaño efectivo de la muestra es **250 sesiones, no 1750 ventanas** — el riesgo
de sobreajuste es mucho mayor de lo que sugiere el tamaño nominal. Promediar las 7
ventanas elimina ruido sin usar nada prohibido: sólo `session_id`, que viene en el test.

---

## 5. Cosas que probamos y descartamos, con evidencia

Vale registrarlas para el probatorio: el descarte con evidencia también es resultado.

| Descarte | Por qué |
|---|---|
| Rango percentil por máquina | 51,9 en `GroupKFold` y 41,8 bajo cambio de régimen |
| Afilado de probabilidades contra la métrica | +0,31 contra un desvío entre folds de 1,7 → ruido |
| Encogimiento hacia la prevalencia | efecto mixto y menor al ruido: con `w=0,10` mejora 0,25 bajo cambio de régimen y empeora 0,41 en `GroupKFold`. Se descarta por parsimonia, no por empeorar |
| Tuneo de hiperparámetros | el cambio de familia de modelo mueve menos que el ruido de partición |
| Features agregadas de sesión (media/std por sesión como columnas) | empeoran: más dimensión, mismo ruido |
| Curado de datos | no hay nada que curar: cero nulos, cero duplicados, sin centinelas |

---

## 6. Dos hallazgos que conviene que todos sepan

### 6.1 El kit de la cátedra tiene un error, y es defendible señalarlo

`baseline.ipynb` afirma: *"la acción óptima para probabilidades bajas es MONITOR"*.
**Es falso para su propio baseline.** Con las prevalencias, `S = 0,876` y
`S_mecánica = 0,464`, así que INSPECT mecánica cuesta `4 + 8·0,412 = 7,30` contra
`8,76` de MONITOR. La regla elige **INSPECT mecánica en las 1848 ventanas**.

Esto importa porque fija la vara real: **el baseline no es pasivo**, es un modelo que
siempre inspecciona la familia más frecuente. Para superarlo hay que acertar la
familia **más del 46 % de las veces**. Explica por qué el baseline da 29 y no 15.

### 6.2 El 70 % del score se juega en la FAMILIA, no en la falla exacta

Con una sola familia verdadera presente, el costo real es:

| Estado real | MONITOR | INSPECT correcta | INSPECT incorrecta |
|---|---|---|---|
| Sana | 0 | 4 | 4 |
| Falla leve | 5 | 4 | 12 |
| Falla severa | 15 | 4 | 12 |

**Distinguir F03 de F04 no aporta un centavo al `cost_score`.** Sólo suma al
`diag_score`, que pesa 10 %. Todo el esfuerzo debería ir a separar las cinco familias
(sana / mecánica / estructural / eléctrica / hidráulica), no las 13 fallas.

Y el techo: el oráculo (predicción perfecta) da **73,96**, no 100, porque aun
acertando todo hay que pagar la inspección de $4. Comparar contra 100 es engañoso.

---

## 7. Reproducibilidad: un problema que teníamos y no sabíamos

La consigna pide adjuntar el código que replica la solución enviada. **Nuestros
notebooks no lo cumplían.**

**`ExtraTrees` (y `RandomForest`) con `n_jobs=-1` no son deterministas aunque tengan
`random_state` fijo.** Comprobado: con `n_jobs=1` dos corridas dan el mismo hash, con
`n_jobs=2` dan hashes distintos. Si un notebook usa `n_jobs=-1`, **el submit no se
puede reproducir**.

Lo que se hizo:
- `n_jobs=1` en ExtraTrees y RandomForest
- `deterministic=True, force_col_wise=True, n_jobs=1` en LightGBM

Verificación en tres niveles:
1. El script corrido dos veces → mismo md5
2. **El notebook y el script generan el submit idéntico byte a byte**
   (`b39b2a76c93767b88eb469e0b4f6df57`)
3. El notebook ejecuta 14/14 celdas con salida y cero errores

**Sugerencia para el equipo:** revisar los notebooks de Ana por lo mismo. Si usan
`n_jobs=-1`, el submit de 43,42 no es reproducible y eso es un problema para la
entrega del art. 5, independientemente del score.

---

## 8. Recomendación

**Subir el submit.** El riesgo es acotado: la configuración **contiene** la de Ana
(mismo z-score, mismo tipo de ensamble) y le suma las features físicas, que ganan en
los dos esquemas de validación. Lo peor razonable es empatar en ~43; lo esperable es
**45-46**.

Lo que **no** prometo: los 50 que sugeriría extrapolar la brecha de `GroupKFold`. El
estimador conservador (cambio de régimen) dice 45,74 y es el que hay que creer,
porque es el único esquema que reproduce la condición real del test.

### Si el resultado vuelve a decepcionar

La próxima hipótesis a testear es que **el test sí tiene combinaciones de fallas**. La
consigna las menciona explícitamente, pero `is_combo = 0` en las 1750 filas de train.
Si el test las tiene, un softmax las subestima sistemáticamente y el `cost_score` cae,
porque INSPECT de una sola familia deja fracción sin cubrir. La cobertura barata es
mezclar el softmax con una rama One-vs-Rest de 13 binarios, con el peso elegido por CV
agrupada. Se puede sondear sin etiquetas mirando la distribución de `Σp` y la entropía
sobre `train_unlabeled`.

---

## Archivos

| Qué | Dónde (rama `Acosta`) |
|---|---|
| Notebook ejecutado | `notebooks/solucion_machine_health.ipynb` |
| Submit propuesto | `submits/submit_fisicas_z_lgb_et.csv` |
| Script que reproduce el submit (~2 min) | `src/generar_submit.py` |
| Scorer vectorizado (verificado contra el oficial, diferencia 0) | `src/fastscore.py` |
| Comparación de enfoques | `src/comparar.py` |

Para correr el notebook hay que poner `datos_sinraw/` y `participant_kit/` al lado.

**Pendiente de decidir:** si el notebook va también a `main` junto al de Ana. Y el
`README.md` de `main` y el de `Acosta` divergieron — el de `main` borra la descripción
del kit y varias referencias; alguien debería unificarlos.
