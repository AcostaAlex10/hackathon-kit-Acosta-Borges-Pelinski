# Respuesta al plan de Ana — qué ya está medido y qué falta

Ana: tu plan apunta bien y **tu sospecha sobre el z-score era correcta**. Varias de
las preguntas ya las corrí, así que las paso con números para que no las repitas, y
marco las que quedan genuinamente abiertas.

Todo esto sale de `notebooks/solucion_machine_health.ipynb` y `src/eda_ana.py`
(rama `Acosta`), ejecutados, no de estimaciones.

---

## 1. Tu sospecha sobre el z-score: tenías razón, y ahora está medida en CV

Dijiste, textual:

> "KS train/test 0.139 → 0.059 demuestra que las distribuciones se acercan, **pero no
> demuestra que la normalización ayude a predecir fallas**. Tenemos que medirlo
> mediante CV, no solamente mediante KS."

Correcto. Lo medí en CV y el resultado te da la razón:

| Features | GroupKFold (3 semillas) | Cambio de régimen |
|---|---|---|
| Crudas | 44,11 | 38,09 |
| Crudas + z-score ← el enfoque actual | 47,55 | 40,22 |
| **Físicas** | 49,82 | **45,35** |
| **Físicas + z-score** | **52,28** | 44,53 |
| Físicas + rango percentil | 51,88 | 41,80 |

**El z-score da +2,5 en `GroupKFold` pero NO mejora la transferencia a máquinas
nuevas.** Con LightGBM solo, `físicas` gana a `físicas+z` bajo cambio de régimen
(45,35 vs 44,53); con el ensamble el orden se invierte (44,76 vs 45,33). Menos de un
punto y cambia de signo según el modelo: está dentro del ruido.

Lo conservamos por la ventaja en `GroupKFold`, pero **no hay que venderlo como que
ayuda a generalizar**. Tu mecanismo hipotético ("elimina información de máquina pero
también información útil de la condición de operación") es exactamente lo que se ve.

### Y hay algo peor: el z-score ni siquiera elimina la huella de máquina

Tu pregunta 7 era "¿hay alguna feature que permita identificar `machine_id`?".
La respuesta es **sí, y de forma perfecta**:

| Features | Accuracy prediciendo `machine_id` (26 clases, azar = 0,038) |
|---|---|
| Crudas | **1,000** |
| Físicas | **1,000** |
| **Sólo z-score (sin las originales)** | **1,000** |

Cada ventana lleva una huella perfecta de su máquina, y **el z-score no la borra**:
sólo recentra, pero la *forma* de la distribución sigue identificando a la máquina.
Eso explica por qué no mejora la transferencia.

Las features que más identifican la máquina: `voltage_a__rms` (0,124),
`voltage_c__rms` (0,122), `V_rms_mean` (0,086), `voltage_b__rms` (0,074),
`delta_p_mean` (0,065). Son tensiones y presión: constantes de instalación, no
estado de salud.

> Nota metodológica: esto usa CV aleatoria, donde ventanas de la misma máquina caen a
> ambos lados. Es la condición que hace *memorizable* la máquina, que es justamente lo
> que queríamos comprobar.

---

## 2. La brecha CV ≈48 vs leaderboard ≈43: resuelta

`GroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*. El test es
*"máquina nueva, **régimen nuevo**"*:

| Variable | train | test |
|---|---|---|
| `rpm_mean` | 1742 | **2052** |
| `flow_mean` | 44,8 | 50,1 |
| `Tamb` | 26,8 | 29,9 |
| `delta_p_mean` | 0,5 | 2,6 |

Agregué una segunda validación que lo reproduce: ordenar las 26 máquinas por rpm,
entrenar con las 13 lentas (1437 rpm) y validar con las 13 rápidas (2057 rpm ≈ los
2052 del test). **La brecha de ~4,6 puntos es el costo de validar con el esquema
equivocado**, y sirve para estimar cualquier configuración nueva.

Sobre tu tabla de componentes, acá está completa (ensamble sobre físicas + z):

| Modelo | GroupKFold | Cambio de régimen |
|---|---|---|
| LightGBM | 52,28 | 44,53 |
| ExtraTrees | 49,09 | 43,95 |
| RandomForest | 50,13 | 44,58 |
| RegLogística | 51,02 | **39,12** |
| **lgb + et** | 53,53 | **45,33** |
| lgb + rf + lr | **54,31** | 43,07 |
| lgb + et + rf + lr | 53,98 | 43,48 |

**Ojo con `lgb+rf+lr`**: es el mejor en `GroupKFold` y de los peores bajo cambio de
régimen. La regresión logística aporta diversidad en distribución, pero es la que peor
transfiere (39,12). Es la misma trampa que el rango percentil.

---

## 3. Tu punto 3 (feature engineering físico): ya está hecho y medido

Tu lista y la que implementé coinciden casi exactamente. Está en el notebook,
son 42 features nuevas, **+5,1 bajo cambio de régimen** sobre crudas+z. Es la
palanca más grande que encontramos.

- **Mecánicas**: `rms/(rpm/1000)²`, `1x/rms²`, `2x/1x`, `log1p(kurtosis)`, `crest`,
  axial/radial, asimetría entre apoyos.
- **Eléctricas**: desbalance de tensión y corriente `(max−min)/media`, CV entre fases,
  `I_min/I_media`, potencia aparente, corriente relativa a rpm y a caudal.
- **Térmicas**: todo contra `Tamb` (nunca absoluto — el test está 3 °C más cálido),
  más la diferencia entre rodamientos.
- **Hidráulicas**: `Δp/(rpm/1000)²`, `flow/rpm`, `p_in/(rpm/1000)²`, relación de
  presiones, rendimiento `flow·Δp/(V·I)`.

### Dos features mías que están mal y hay que arreglar

La auditoría de redundancia que pediste (tu punto de correlación) me delató:

- `acc_*__crest_n` es **copia exacta** de `acc_*__crest`, correlación 1,000. Me olvidé
  de normalizarlo; el crest factor ya es adimensional, así que la columna sobra.
- `S_ap` (potencia aparente) correlaciona 1,000 con `I_rms_mean`, porque la tensión es
  casi constante. Sobra.

En total **24 pares con correlación > 0,95**. Hay para limpiar.

---

## 4. Tu EDA dirigido: resultados

### ¿Qué separa sano de falla?
Vibración radial, sobre todo el **crest factor**: `acc_radial_a__crest` AUC 0,770,
`acc_radial_b__crest` 0,707, `acc_radial_b__rms` 0,703.

### ¿Qué separa una familia de otra? (esto es el 70 % del score)
Medido **sólo entre ventanas con falla**, para no confundir "¿hay falla?" con "¿cuál?":

| Familia | n | Mejores discriminadores |
|---|---|---|
| Mecánica | 812 | `acc_radial_a__rms` 0,691 · `acc_radial_a__1x` 0,689 |
| Eléctrica | 378 | `acc_radial_a__crest` 0,821 · `acc_radial_a__rms` 0,782 |
| Hidráulica | 252 | `rad_b_a` 0,770 · `acc_radial_b__2x` 0,725 · `ax_rad` 0,724 |
| Estructural | 91 | `acc_axial__crest` **0,916** · `acc_axial__1x` 0,791 |

**Dato llamativo:** las fallas *eléctricas* se detectan mejor por **vibración** que por
features eléctricas. Tiene sentido físico (un desequilibrio de tensión genera
vibración a 2× la frecuencia de línea), pero sugiere que las features eléctricas
todavía no están bien explotadas. **Ahí hay margen.**

### ¿Qué separa fallas dentro de la misma familia?

| Par | n | Mejor discriminador | AUC |
|---|---|---|---|
| F01 vs F02 (desbalance/desalineación) | 336 | `acc_radial_a__2x`, `ax_rad` | **1,000** |
| F12 vs F13 (cavitación/obstrucción) | 252 | `acc_radial_b__crest` | 0,920 |
| F08 vs F09 (desequilibrio/pérdida fase) | 196 | `I_cv`, `I_desbal` | 0,896 |
| F10 vs F11 (barras/espiras) | 182 | `acc_axial__2x` | 0,882 |
| **F03 vs F05** (pista ext./elemento rodante) | 273 | `acc_radial_a__kurtosis` | **0,714** |
| **F03 vs F04** (pista ext./pista int.) | 259 | `acc_radial_a__1x` | **0,695** |

**F01 vs F02 es perfectamente separable** (AUC 1,000). Las intra-familia en general
están bien resueltas… **salvo los rodamientos**. F03/F04/F05 son las únicas realmente
difíciles, y es exactamente donde harían falta las frecuencias BPFO/BPFI/BSF de las
señales crudas. Eso confirma la intuición previa, pero **acota mucho el problema**:
sólo 3 de 13 fallas necesitan NPZ, y como pesan sólo en el `diag_score` (10 %),
**el retorno de ir a las señales crudas es bajo**.

### ¿Features que no aportan?
35 de 87 tienen |AUC−0,5| < 0,02 contra sano/falla: todas las `*__nan_frac`, las
tensiones y corrientes crudas por fase. Ojo: pueden servir igual para separar
familias aunque no detecten falla, así que yo no las tiraría sin medirlo.

---

## 5. Qué haría ahora, en orden

1. **Limpiar la redundancia** (24 pares > 0,95, empezando por las dos features mías
   que están duplicadas). Barato y reduce varianza.
2. **Explotar mejor lo eléctrico.** Que las fallas eléctricas se detecten por
   vibración indica que las features eléctricas están subaprovechadas. Faltan
   componentes de secuencia (positiva/negativa/cero), que es el indicador estándar
   de desequilibrio, y el factor de potencia.
3. **Atacar F03/F04/F05.** Son el único agujero real. Antes de ir a NPZ, probaría un
   modelo jerárquico: primero familia (5 clases, el 70 %), después la falla dentro de
   la familia. Alinea el modelo con la métrica.
4. **NPZ sólo al final**, y sabiendo que sólo mueve el 10 % del score.

**Lo que NO haría:** tunear hiperparámetros (el cambio de familia de modelo mueve
menos que el ruido de partición), ni perseguir el número de `GroupKFold` sin mirar el
de cambio de régimen.

---

## 6. Descartes ya medidos, para que no los repitamos

| Descarte | Evidencia |
|---|---|
| Rango percentil por máquina | 51,9 en `GroupKFold` y 41,8 bajo cambio de régimen |
| Afilado de probabilidades contra la métrica | +0,31 contra desvío entre folds de 1,7 |
| Encogimiento hacia la prevalencia | efecto mixto y menor al ruido |
| Features agregadas de sesión como columnas | empeoran |
| Curado de datos | no hay nada que curar: cero nulos, cero duplicados, sin centinelas |

---

## 7. Dos cosas operativas

**Reproducibilidad.** `ExtraTrees` y `RandomForest` con `n_jobs=-1` **no son
deterministas** aunque tengan `random_state` fijo (comprobado: `n_jobs=1` da el mismo
hash, `n_jobs=2` no). Si tu notebook usa `n_jobs=-1`, el submit de 43,42 no se puede
reproducir, y la consigna pide adjuntar el código que replica la entrega. Vale la pena
revisarlo.

**El baseline no es pasivo.** `baseline.ipynb` dice que P0 elige MONITOR; en realidad
elige **INSPECT mecánica en las 1848 ventanas** (`4+8·0,412 = 7,30 < 8,76`). Por eso
da 29 y no 15: para superarlo hay que acertar la familia más del 46 % de las veces.
Es un hallazgo defendible para el probatorio.
