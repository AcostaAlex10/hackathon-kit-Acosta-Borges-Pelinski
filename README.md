# Machine Health — Hackathon IC415

**Equipo:** Acosta · Borges · Pelinski — **🏆 1.er puesto**

Mantenimiento predictivo sobre un banco de bombas: dado un registro de 2 segundos
de una máquina, decidir si está sana o cuál de 13 fallas tiene, y con eso elegir
la acción de mantenimiento más barata.

**Resultado final: 50,6966** en el servidor, con el notebook
[`notebooks/4_nzp1.ipynb`](notebooks/4_nzp1.ipynb).

---

## Por dónde empezar

| Si querés… | Andá a |
|---|---|
| entender el problema y la solución sin leer código | [`docs/Machine_Health_NPZ_guia.pdf`](docs/Machine_Health_NPZ_guia.pdf) |
| la versión para exponer | [`docs/Machine_Health_guia_exposicion.pdf`](docs/Machine_Health_guia_exposicion.pdf) |
| la consigna original | [`docs/Consigna_Hackathon.pdf`](docs/Consigna_Hackathon.pdf) |
| el notebook que ganó | [`notebooks/4_nzp1.ipynb`](notebooks/4_nzp1.ipynb) |
| ver cómo se llegó ahí | la tabla de abajo, en orden |

---

## La cadena de intentos

Cada notebook es un envío real al servidor. Están numerados en el orden en que
se hicieron, y cada uno parte del anterior.

| # | Notebook | Qué agrega | Cols | Score servidor |
|---|---|---|---|---|
| — | — | primer intento | — | 28 |
| — | — | descriptores crudos + z-score por máquina, ensamble | 92 | 43,42 |
| 1 | [`1_base4.ipynb`](notebooks/1_base4.ipynb) | **42 variables físicas** + z por máquina | 174 | **48,1150** |
| 2 | [`2_base4.5.ipynb`](notebooks/2_base4.5.ipynb) | protocolo de validación por cambio de régimen | 174 | — |
| 3 | [`3_v5.ipynb`](notebooks/3_v5.ipynb) | primera etapa de señales crudas, con compuerta de decisión | — | — |
| 4 | [`4_nzp1.ipynb`](notebooks/4_nzp1.ipynb) | **features espectrales de los NPZ** (vibración, corriente, presión) | 249 | **50,6966** ✅ |

Los CSV enviados están en [`submits/`](submits/), con el score en el nombre.

### Lo que midió cada matriz de features (OOF, métrica oficial)

`OOF` = *out-of-fold*: se parte el train en 5 folds con `StratifiedGroupKFold`
agrupando por `machine_id`, y cada fila se predice con un modelo que nunca la vio.

| Matriz de features | Cols | Mejor individual | Mejor ensamble |
|---|---|---|---|
| Piso: prevalencia constante (baseline de la cátedra) | — | 30,78 | — |
| `ana_z` — 46 originales + z por máquina | 92 | XGBoost 45,52 | LGB+RF+RegLog 47,33 |
| `fis_z` — + 42 variables físicas | 174 | CatBoost 50,86 | LGB+RF+RegLog 51,98 |
| + vibración NPZ normalizada | 201 | CatBoost 50,88 | — |
| + corriente NPZ | 227 | LightGBM 52,84 | — |
| + presión NPZ | 249 | XGBoost **53,07** | — |
| Techo: oráculo (etiquetas exactas) | — | 73,96 | — |

---

## Los siete hechos que gobiernan el problema

Están medidos, no estimados. Cualquiera que retome el repo debería leerlos antes
de proponer nada.

1. **No es multilabel.** `is_combo = 0` en las 1750 filas: es **multiclase de 14
   estados excluyentes** (sano + 13 fallas). Se modela con softmax, no con 13
   clasificadores binarios.
2. **La etiqueta es constante dentro de cada sesión** (250 de 250) y cada sesión
   tiene exactamente 7 ventanas. **El n efectivo es 250, no 1750.**
3. **Train y test no comparten ninguna máquina** (26 vs 22, intersección vacía)
   **y el régimen cambia**: `rpm_mean` 1742 → 2052, `delta_p_mean` 0,51 → 2,61.
   De acá sale la normalización por máquina, que es la palanca central del repo.
4. **El 70 % del score se juega en la FAMILIA**, no en la falla exacta. Confundir
   F03 con F04 no cambia `cost_score`: misma familia, misma acción.
5. **El techo real es 73,96, no 100.** Aun con predicción perfecta hay que pagar
   la inspección.
6. **El baseline de la cátedra no es pasivo:** elige INSPECT mecánica en las 1848
   ventanas. Para superarlo hay que acertar la familia más del 46 % de las veces.
7. **La capa de decisión de la métrica ya es óptima** (coincide en el 99,1 % con
   la acción óptima bajo nuestro posterior). No hay nada que ganar distorsionando
   probabilidades.

**La métrica:** 70 % `cost_score` (decisión de mantenimiento) + 20 % `prob_score`
(calibración) + 10 % `diag_score` (macro-F1).

---

## Las ramas

El repo tiene cuatro ramas y ninguna más. `main` es el registro de lo que se
probó y se envió; las otras tres son el cuaderno de trabajo de cada integrante.

| Rama | Qué hay |
|---|---|
| **`main`** | esta cadena de intentos, los submits, las guías y el material de la cátedra |
| **`Acosta`** | features físicas, EDA dirigido, fase NPZ con AR-Burg, auditoría de bugs |
| **`Borges`** | benchmark multimodelo, el protocolo de validación por régimen, módulo de señal cruda |
| **`Pelinski`** | el notebook nzp1 que ganó, la sección 10 de palancas, matriz de 347 columnas |

Las tres ramas personales tienen líneas de trabajo distintas, pero **el trabajo
fue de a tres en todas**: las decisiones se discutieron y se midieron en grupo, y
cada rama recibió aportes de los otros dos. La separación es de organización, no
de autoría.

---

## Estructura

```
├── notebooks/
│   ├── 1_base4.ipynb … 4_nzp1.ipynb   <- la cadena de intentos, en orden
│   └── BASE_tabular / BASE_vision / arranque_colab.ipynb   <- plantillas del kit
├── submits/                <- los CSV enviados, con el score en el nombre
├── docs/
│   ├── Machine_Health_NPZ_guia.pdf         <- guía de estudio del notebook final
│   ├── Machine_Health_guia_exposicion.pdf  <- guía para exponer
│   ├── Consigna_Hackathon.pdf
│   ├── GUIA_EQUIPO.md      <- qué hace cada pieza del kit
│   └── proceso/            <- handoffs entre sesiones, cronograma de la jornada
├── ic_kit/                 <- kit de la cátedra (costs.py trae el scoring)
├── contexto/               <- reglamento, memoria de la materia, desafíos previos
├── bootstrap.py            <- arranque + autotest
└── run_tabular.py, run_vision.py
```

Los datos del desafío **no van al repo** (`.gitignore`). Los notebooks los
descargan en su primera celda.

## Reproducir

```bash
git clone https://github.com/AcostaAlex10/hackathon-kit-Acosta-Borges-Pelinski.git
cd hackathon-kit-Acosta-Borges-Pelinski
python bootstrap.py --install     # termina en AUTOTEST OK si el kit quedó bien
```

Después, abrir `notebooks/4_nzp1.ipynb` y correrlo de arriba abajo: las primeras
celdas bajan los datos y las señales crudas.
