# Rama `Pelinski` — Machine Health

Cuaderno de trabajo de **Pelinski** para el desafío Machine Health de la
hackathon IC415. **Es la rama donde nació el notebook que ganó.**

> **Sobre la autoría.** Las tres ramas personales (`Acosta`, `Borges`,
> `Pelinski`) organizan líneas de trabajo distintas, pero **el trabajo fue de a
> tres en todas**. Las hipótesis se discutieron en grupo, cada medición se
> revisó entre los tres y las tres ramas recibieron aportes de los otros dos. La
> separación es de organización, no de autoría.
>
> La cadena completa de intentos del equipo está en [`main`](../../tree/main).

---

## Los notebooks, en orden

| # | Notebook | Qué es | Celdas |
|---|---|---|---|
| 0 | [`0_primer_intento.ipynb`](notebooks/0_primer_intento.ipynb) | primer envío | 46 |
| 1 | [`1_nzp1.ipynb`](notebooks/1_nzp1.ipynb) | **el que se entregó: 50,6966** — señales crudas NPZ sobre las 174 columnas físicas | 135 |
| 2 | [`2_limpio.ipynb`](notebooks/2_limpio.ipynb) | reorganización de nzp1 con tres bugs corregidos, más la sección 10 de palancas | 174 |
| 3 | [`3_limpio1.ipynb`](notebooks/3_limpio1.ipynb) | la misma línea corrida en Colab, hasta la matriz de 347 columnas | 186 |

La guía de estudio del notebook entregado está en
[`Machine_Health_NPZ_guia.pdf`](Machine_Health_NPZ_guia.pdf).

---

## 1 · `nzp1` — la entrega de 50,6966

Parte de las 174 columnas (46 descriptores originales + z-score por máquina + 42
variables físicas) y les suma **features espectrales extraídas de los NPZ**.

Tres canales por ventana de 2 segundos: `acc_radial_a` a 20 kHz, `current_a` a
10 kHz, `pressure_in` a 200 Hz. De cada uno se extraen potencias por banda,
armónicos 1x–5x de la frecuencia de rotación, descriptores espectrales
(centroide, ancho de banda, entropía) y variabilidad entre segmentos.

**El problema que apareció, y cómo se resolvió.** Antes de aceptar las features
crudas se midió cuánta de su varianza es de la falla y cuánta es de la máquina
(`between_machine_signal`). Seis de ellas —todas ratios armónicos y potencias
relativas— dieron cocientes de 2,3 a 4,3: **identificaban la máquina, no la
falla**. Se descartaron esas seis, y las 27 restantes se normalizaron con
**z-score robusto por máquina**, con mediana y MAD en vez de media y desvío:

```
z_robusto = (x − mediana) / (1,4826 × MAD)
```

Las potencias espectrales son mucho más sesgadas que los descriptores tabulares,
y el desvío estándar se deja arrastrar por los outliers donde el MAD no.

**Lo que aportó cada canal**, agregado de a uno (OOF, métrica oficial):

| Matriz | Cols | Mejor individual |
|---|---|---|
| `fis_z` sola | 174 | CatBoost 50,86 |
| + vibración NPZ normalizada | 201 | CatBoost 50,88 |
| + corriente NPZ | 227 | LightGBM **52,84** |
| + presión NPZ | 249 | XGBoost **53,07** |

**La corriente es el aporte más claro**: LightGBM pasa de 50,42 a 52,84, y
`diag_score` salta de 46,52 a 52,55. La vibración sola casi no mueve la aguja una
vez normalizada; los dos canales juntos aportan más que cualquiera por separado
(sin vibración: 51,15 / 51,72).

---

## 2 · `limpio` — la reorganización y los tres bugs

Misma corrida, reordenada para que se pueda leer y auditar. **Tres bugs reales
corregidos**, cada uno señalado en su lugar con una celda de aviso:

1. la comparación mal etiquetada de `RAW_VIB`;
2. un `SyntaxError` en la celda de fusión (dos celdas casi idénticas de copiar y
   pegar en Colab, la primera truncada);
3. el desajuste de columnas entre train y test en la matriz combinada — por este
   la sección 7 **nunca había llegado a correr**.

Además se unificó la cañería: `evaluar`, `familia_de` y las cuatro variantes de
la función de benchmark estaban definidas entre tres y cuatro veces cada una;
ahora viven una sola vez en la sección 4. Los números son los mismos de la
corrida original; los textos que citaban valores de corridas anteriores se
reescribieron contra las salidas reales.

---

## 3 · La sección 10 — cinco palancas, medidas una por una

| § | Palanca | Resultado |
|---|---|---|
| 10.1 | ¿el estado es constante por sesión? | **Sí**: 7 ventanas exactas por sesión de los dos lados, 250 de 250 sesiones con una sola etiqueta. El n efectivo es 250, no 1750 |
| 10.2 | features de sesión y *pooling* de las 7 ventanas | **no mejora**: los errores de las 7 ventanas están correlacionados, comparten máquina y sesión. Promediar limpia ruido de segundo orden |
| 10.3 | de dónde sale el costo | reconstrucción del costo por ventana, descompuesto por familia real × acción elegida: dice qué atacar en vez de adivinar |
| 10.4 | **cabezas jerárquicas** | `P(F_k) = P(falla) · P(familia \| falla) · P(k \| familia)`. Con 85–170 ejemplos por clase, la cabeza de familia ve 200–900 y estima mucho mejor **justo lo que gobierna el 70 % del score** |
| 10.5 | **calibración por temperatura** | un solo parámetro sobre los logits, así que no puede sobreajustar y no cambia el orden de las clases: sólo qué tan afiladas quedan. `T < 1` afila (sube `cost` y `diag`), `T > 1` suaviza (sube `prob`) |
| 10.6 | features de señal orientadas a la familia | corriente v2 (fundamental estimada del espectro, bandas laterales de *slip*), vibración v2 (demodulación por Hilbert sobre la banda resonante) y coherencia vibración–corriente |
| 10.7 | `train_unlabeled` para la normalización | no como pseudo-etiquetado: para **estimar la mediana y el MAD por máquina con más ventanas**. La normalización por máquina es la palanca central del repo y se calibraba sólo con las etiquetadas |

También quedó arreglado un bug de las cabezas jerárquicas con XGBoost, que no
tolera etiquetas no contiguas cuando una familia no tiene todas sus clases
presentes en el fold.

---

## 4 · La matriz de 347 columnas

`3_limpio1.ipynb` llega hasta una matriz de 347 columnas: base + físicas + z +
NPZ (vibración/corriente/presión) + las v2 de 10.6 + z estimada con el pool de
`train_unlabeled`. El submit y su procedencia están en
[`notebooks/submits/`](notebooks/submits/).

**Cuidado al comparar sus números.** Se corrió con scikit-learn 1.9.0 en el
contenedor y no en Colab, así que los folds de `StratifiedGroupKFold` no coinciden
con los de la corrida original: el 58,25 de OOF **no es comparable** con el 55,35
medido en Colab ni con el 53,07 de nzp1. Es exactamente el tipo de trampa que el
resto del repo se ocupa de evitar, y por eso el JSON de procedencia lo dice.

## Cómo leer cualquier resultado de esta rama

Con `sd_folds` alrededor de 4,5, **cualquier diferencia menor a ~1 punto entre
configuraciones es ruido**. Dos reglas:

1. una mejora vale si aparece en **las dos semillas** de folds (42 y 2024), no
   sólo en una;
2. para decidir entre configuraciones parecidas, promediar el OOF sobre varias
   semillas de partición en vez de mirar una sola.

---

## Estructura

```
├── notebooks/
│   ├── 0_primer_intento.ipynb
│   ├── 1_nzp1.ipynb        <- la entrega de 50,6966
│   ├── 2_limpio.ipynb      <- reorganizado, 3 bugs corregidos, sección 10
│   ├── 3_limpio1.ipynb     <- hasta la matriz de 347 columnas
│   └── submits/            <- los CSV, con la procedencia en JSON
├── Machine_Health_NPZ_guia.pdf
└── ic_kit/                 <- kit de la cátedra (costs.py trae el scoring)
```

El trabajo vive en los notebooks: no hay módulos `.py` propios más allá del kit
de la cátedra.
