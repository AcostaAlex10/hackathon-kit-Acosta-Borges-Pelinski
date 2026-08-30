# Rama `Acosta` — Machine Health

Cuaderno de trabajo de **Alex Acosta** para el desafío Machine Health de la
hackathon IC415.

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

| # | Aporte | Dónde |
|---|---|---|
| 1 | **Estrategia 1**: lectura del caso, EDA inicial y disección del scorer | [`contexto/estrategia1.md`](contexto/estrategia1.md) |
| 2 | **Features físicas + validación por cambio de régimen** — la matriz de 174 columnas que dio 48,1150 | [`notebooks/solucion_machine_health.ipynb`](notebooks/solucion_machine_health.ipynb) |
| 3 | **EDA dirigido**: responde con mediciones el plan de trabajo propuesto | [`notebooks/respuesta_a_ana_eda_dirigido.md`](notebooks/respuesta_a_ana_eda_dirigido.md) |
| 4 | **Análisis de la tercera bala**: qué quedaba por probar y con qué prioridad | [`notebooks/respuesta_posible_bala_3.md`](notebooks/respuesta_posible_bala_3.md) |
| 5 | **Notebook único que reproduce la entrega** (requisito del art. 5 del reglamento) | [`notebooks/solucion_final.ipynb`](notebooks/solucion_final.ipynb) |
| 6 | **Fase NPZ a ciegas** + reponderación por familia, identificación output-only (AR-Burg) y agregado por sesión | [`notebooks/npz_fase.ipynb`](notebooks/npz_fase.ipynb) |
| 7 | **Handoff a la sesión siguiente** y lista de los 89 NPZ prioritarios (~93 MB) para validar la extracción sin bajar los 4032 | [`trabajo_delegado/`](trabajo_delegado/) |
| 8 | **Verificación: `lgb+rf+lr` era una trampa de validación** | [`resultados/ensambles_dos_jueces.txt`](resultados/ensambles_dos_jueces.txt) |

---

## El hallazgo principal: el juez equivocado ordena mal

El ensamble `LGB + RF + RegLog` era el mejor bajo `GroupKFold`. Medido con los
dos jueces sobre las mismas 174 columnas:

| Ensamble | GroupKFold (3 semillas) | 5 cortes de régimen |
|---|---|---|
| `lgb+et+rf` | 54,07 | **45,21 ± 1,67** |
| `lgb+rf+lr` | **54,34** | 42,88 ± 1,37 |
| `lr` sola | 51,02 | 37,96 ± 1,85 |

`lgb+rf+lr` gana por 0,27 con `GroupKFold` y **pierde 2,33 puntos bajo cambio de
régimen**. La regresión logística sola se desploma 13 puntos entre un juez y el
otro: aporta diversidad *dentro* de la distribución, pero extrapola linealmente a
un régimen que nunca vio (train 1742 rpm → test 2052 rpm), mientras que los
árboles saturan en el borde del rango conocido.

Y no sólo ordena mal: **comprime las diferencias**. Bajo `GroupKFold` los
ensambles se separan por 3 puntos; bajo cambio de régimen, por 7,3.

Fue la cuarta cosa medida que sube con `GroupKFold` y baja bajo shift, junto con
el rango percentil por máquina, la limpieza por redundancia y la selección Top-K
por información mutua. Por eso **no se usó**, pese a haber sido el mejor en el
notebook que lo eligió.

---

## Otras dos cosas que se midieron y se descartaron

- **Calibración**: medida y descartada. La capa de decisión de la métrica ya es
  óptima (coincide en el 99,1 % con la acción óptima bajo nuestro posterior), así
  que no había nada que ganar distorsionando probabilidades.
- **Cobertura OvR a 0,3** sobre `base4`: se midió el efecto sobre `cost_score` y
  no justificó el cambio.

También quedó una **auditoría del notebook NPZ** que encontró y corrigió dos bugs
reales (la comparación mal etiquetada de `RAW_VIB` y el desajuste de columnas
entre train y test).

---

## Estructura

```
├── notebooks/
│   ├── solucion_machine_health.ipynb   <- features físicas + validación por régimen
│   ├── solucion_final.ipynb            <- reproduce la entrega de punta a punta
│   ├── npz_fase.ipynb                  <- fase de señales crudas
│   ├── respuesta_a_ana_eda_dirigido.md
│   └── respuesta_posible_bala_3.md
├── resultados/ensambles_dos_jueces.txt <- la medición de los dos jueces
├── trabajo_delegado/                   <- handoff + lista de NPZ prioritarios
├── submits/                            <- envíos generados desde esta rama
├── contexto/estrategia1.md             <- análisis del caso y del scorer
└── ic_kit/                             <- kit de la cátedra (costs.py trae el scoring)
```

El trabajo vive en los notebooks: no hay módulos `.py` propios más allá del kit
de la cátedra.
