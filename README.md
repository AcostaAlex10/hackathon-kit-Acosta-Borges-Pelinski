# Rama `Borges` — Machine Health

Cuaderno de trabajo de **Borges** para el desafío Machine Health de la hackathon
IC415.

> **Sobre la autoría.** Las tres ramas personales (`Acosta`, `Borges`,
> `Pelinski`) organizan líneas de trabajo distintas, pero **el trabajo fue de a
> tres en todas**. Las hipótesis se discutieron en grupo, cada medición se
> revisó entre los tres y las tres ramas recibieron aportes de los otros dos. La
> separación es de organización, no de autoría.
>
> El resultado final del equipo (**50,6966**, 1.er puesto) y la cadena completa
> de intentos están en [`main`](../../tree/main).

---

## La línea de trabajo de esta rama

Esta rama es la del **método de validación**: qué juez usar para decidir si un
cambio sirve. Es el hilo que corrige el rumbo del equipo dos veces.

| # | Aporte | Dónde |
|---|---|---|
| 1 | **Estrategia**, en tres revisiones: lectura del desafío → techo tabular medido → *el estimador es el problema, no el modelo* | [`estrategia1_Borges.md`](estrategia1_Borges.md) |
| 2 | **Benchmark multimodelo**: el techo tabular está en ~48, y no lo mueve el algoritmo | [`notebooks/1_benchmark_multimodelo.ipynb`](notebooks/1_benchmark_multimodelo.ipynb) |
| 3 | **v3**: features físicas y validación por cambio de régimen | [`notebooks/2_machine_health_v3.ipynb`](notebooks/2_machine_health_v3.ipynb) |
| 4 | **v4**: el envío que bajó, y su diagnóstico | [`notebooks/3_machine_health_v4.ipynb`](notebooks/3_machine_health_v4.ipynb) |
| 5 | **Handoff de la fase de señales crudas**, con todo lo medido | [`TRASPASO_NPZ.md`](TRASPASO_NPZ.md) |
| 6 | **Guía de estudio para la exposición** | [`Machine_Health_guia_exposicion.pdf`](Machine_Health_guia_exposicion.pdf) |

---

## Lo principal: el juez tiene que ser el cambio de régimen

`StratifiedGroupKFold` por `machine_id` mide *"máquina nueva, mismo régimen"*.
El test es *"máquina nueva, **régimen nuevo**"* (`rpm_mean` 1742 → 2052). Decidir
con `GroupKFold` costó dos envíos.

El reemplazo: ordenar las 26 máquinas por `rpm_mean`, entrenar con las lentas y
validar con las rápidas. **Pero con un solo corte ese estimador engaña.** Medido
sobre cinco cortes (k = 11…15 máquinas lentas):

| Configuración | k=11 | k=12 | k=13 | k=14 | k=15 | media | sd |
|---|---|---|---|---|---|---|---|
| base4 tal cual | 42,98 | 43,70 | **45,90** | 43,48 | 43,06 | 43,83 | 1,07 |
| sin duplicados | 44,19 | 43,34 | 43,82 | 41,75 | 43,62 | 43,34 | 0,84 |
| sin redundancia | 43,13 | 42,65 | **46,55** | 44,79 | 44,10 | 44,24 | 1,37 |
| + secuencia | 44,60 | 44,10 | 44,71 | 43,47 | 43,19 | 44,01 | 0,60 |

El desvío del estimador es 1,0 y las diferencias entre configuraciones son 0,75.
El corte k=13 —el único que se usaba— es el más optimista de los cinco.

**Las cuatro reglas que salieron de acá:**

1. El juez es el **promedio de cinco cortes**, nunca uno solo, nunca `GroupKFold`.
2. Nada se acepta por menos de ~2 puntos, salvo que haya un argumento que no sea
   el score.
3. **Medir con el ensamble que se va a enviar**, no con un modelo suelto.
4. Reportar siempre media ± sd, no el número pelado.

---

## La regresión de v4, diagnosticada

v4 dio **47,3733** contra los **48,1150** de base4. Bajó. La causa, aislada
midiendo cada cambio por separado sobre los cinco cortes con el ensamble real:

| Configuración | final | cost | prob | diag |
|---|---|---|---|---|
| 174 columnas (base4) | **45,44** | 37,23 | 81,33 | 31,17 |
| 142 columnas (v4, sin redundancia) | **44,45** | 36,16 | 81,18 | 29,03 |
| 142 + sesión | 44,76 | 36,46 | 81,25 | 29,83 |
| 142 + sesión + OvR (= v4 enviado) | 45,21 | 36,69 | 81,30 | 32,64 |

**Quitar las 12 columnas redundantes cuesta un punto entero.** El error de método
fue medir ese cambio con LightGBM solo y aplicarlo a un ensamble con ExtraTrees y
RandomForest: con `max_features="sqrt"`, las copias correlacionadas **aumentan la
probabilidad de que una variable útil entre en cada split**. Para un ensamble de
bagging la redundancia no es ruido, es muestreo. La parsimonia que se buscaba era
una mejora para un modelo y un daño para otro.

Los otros dos cambios quedaron exonerados: el promedio por sesión es neutro
(+0,02) y la cobertura OvR es positiva (+0,40).

---

## El techo de las señales crudas, cuantificado

El equipo venía diciendo que el retorno de los NPZ era bajo porque sólo F03/F04/F05
son difíciles y ésas pesan en `diag_score`, que es el 10 %. **Medido, eso estaba
mal.** Simulando sobre el OOF:

| Escenario | final | cost | prob | diag |
|---|---|---|---|---|
| actual | 52,07 | 43,97 | 83,49 | 45,92 |
| + intra-familia perfecta (rodamientos) | **56,47** | 43,97 | 92,30 | 72,33 |
| + familia perfecta | **64,84** | 59,03 | 87,12 | 60,98 |

Resolver los rodamientos vale **+4,4 puntos**, no ~1,7: el razonamiento viejo sólo
contaba `diag_score` y olvidaba que concentrar la masa en la falla correcta
**también mejora el Brier** (`prob_score` 83,5 → 92,3, y eso pesa 20 %).

Acertar la familia vale aún más (**+12,8**). Esta medición es la que justificó
entrar a la fase NPZ, que terminó dando los 50,6966 finales.

**Qué buscar en la señal** (lo que la tabla no tiene: ya trae rms, kurtosis,
crest, 1x y 2x por canal, así que recalcular eso desde el NPZ no agrega nada):

- **espectro de envolvente** — donde viven BPFO/BPFI/BSF; es lo único que separa
  F03/F04/F05 entre sí;
- **kurtosis espectral / banda resonante** — dónde buscar esa envolvente;
- **bandas laterales alrededor de 1x en la corriente** (MCSA) — barras rotóricas;
- **espectro de presión** — firma de cavitación, banda ancha de alta frecuencia.

---

## Lo descartado, y por qué

- *Afilado de probabilidades:* la capa de decisión de la métrica ya es óptima
  (99,1 % de coincidencia con la acción óptima bajo nuestro posterior).
- *13 binarios independientes como modelo principal:* el train no tiene combos y
  el softmax aprovecha la restricción. Sí se usan como rama de cobertura.
- *Normalización por `session_id`:* borra la falla, que es constante en la sesión.
- *Rango percentil por máquina:* 51,9 en `GroupKFold` y 41,8 bajo régimen.
- *Búsqueda de algoritmo y tuneo de hiperparámetros:* mueven menos que el ruido.
- *Componentes de secuencia y modelo jerárquico:* medidos, no pasan.
- *Decidir con un solo corte de régimen:* k=13 es el más optimista en la mitad de
  los casos.

---

## Estructura

```
├── estrategia1_Borges.md   <- el análisis completo, en tres revisiones
├── TRASPASO_NPZ.md         <- handoff de la fase de señales crudas
├── Machine_Health_guia_exposicion.pdf
├── notebooks/
│   ├── 1_benchmark_multimodelo.ipynb   <- el techo tabular está en ~48
│   ├── 2_machine_health_v3.ipynb       <- features físicas + validación por régimen
│   ├── 3_machine_health_v4.ipynb       <- el envío que bajó (47,3733)
│   └── submits/
└── ic_kit/                 <- kit de la cátedra (costs.py trae el scoring)
```

El notebook `base4` (el de 48,1150) vive en
[`main` como `notebooks/1_base4.ipynb`](../../blob/main/notebooks/1_base4.ipynb):
estaba duplicado acá por un merge y se quitó.

El trabajo vive en los notebooks: no hay módulos `.py` propios más allá del kit
de la cátedra.
