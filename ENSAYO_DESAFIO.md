# Ensayo — Desafío de práctica

> **Simulacro.** Este documento imita el formato de las consignas de IC415.
> Usalo como contexto de entrada para una sesión nueva de Claude Code, junto
> con `CONTEXTO_CLAUDE.md`, y cronometrate.
>
> **No leas la sección 7 hasta terminar.** Está sellada a propósito.

---

## INTELIGENCIA COMPUTACIONAL — IC415
### Consigna Desafío de Ensayo
**Entrega: Individual**

---

## 1. Contexto y objetivo

La distribuidora eléctrica provincial opera una flota de transformadores de
potencia repartidos en subestaciones de toda la provincia. Una falla no
anticipada en una de estas unidades deja sin servicio a miles de usuarios,
puede provocar un incendio y su reemplazo demora meses.

El mantenimiento hoy es **por calendario**: cada unidad se inspecciona cada
tantos meses, independientemente de su estado real. Eso significa que se
gastan recursos en equipos sanos mientras alguna unidad crítica espera turno.

El área de activos dispone de análisis de gases disueltos en aceite
(cromatografía), registros de carga y datos de operación. Quiere pasar a un
esquema de **mantenimiento basado en condición**: un modelo que estime el
estado de riesgo de cada unidad para priorizar las cuadrillas.

Se definen cuatro estados:

| Estado | Significado operativo |
|---|---|
| `Normal` | sin acción |
| `Vigilancia` | repetir cromatografía en 3 meses |
| `Mantenimiento urgente` | intervención programada dentro de 30 días |
| `Falla inminente` | sacar de servicio ya |

---

## 2. Datos disponibles

En `tests/ensayo_data/`:

- **`train_labeled.csv`** — unidades con historial y el campo `estado`.
  Uso libre para exploración, análisis y entrenamiento.
- **`train_unlabeled.csv`** — mismos atributos sin `estado`, de una
  distribuidora vecina. Opcional, para técnicas semi/no supervisadas.
- **`test_features.csv`** — unidades a clasificar. **Queda terminantemente
  prohibido usar este archivo para ajustar o validar modelos.**

Los datos se exportaron del sistema de gestión de activos e incluyen todos los
campos que almacena. Buena parte de los registros de cromatografía se cargaron
manualmente en planilla antes de la migración, así que **no se descarta la
existencia de errores u omisiones**.

Variables: `id_unidad`, `horas_servicio`, `temperatura_aceite`, `carga_pct`,
`gas_h2_ppm`, `gas_ch4_ppm`, `gas_c2h2_ppm`, `humedad_aceite_ppm`,
`rigidez_dielectrica_kv`, `vibracion_mm_s`, `marca`, `refrigeracion`,
`orden_trabajo`, `distribuidora`, `inspeccion_visual`.

---

## 3. Métrica de evaluación

No todos los errores cuestan lo mismo. Mandar una cuadrilla a un equipo sano
cuesta plata; dejar en servicio una unidad a punto de fallar cuesta un
transformador y una interrupción provincial.

Matriz de costos `C[real, predicho]`:

| Real \ Predicho | Normal | Vigilancia | Mant. urgente | Falla inminente |
|---|---|---|---|---|
| **Normal** | 0 | 1 | 3 | 6 |
| **Vigilancia** | 2 | 0 | 1 | 4 |
| **Mantenimiento urgente** | 5 | 3 | 0 | 2 |
| **Falla inminente** | 15 | 9 | 3 | 0 |

La métrica es el **costo promedio por unidad**:

```
costo = (1/N) · Σ C[real_i, predicho_i]
```

**Menor es mejor.** El mínimo posible es 0 y el máximo 15.

| Costo | Interpretación |
|---|---|
| ≤ 1,15 | excelente |
| 1,15 – 1,30 | bueno |
| 1,30 – 1,60 | aceptable |
| > 1,60 | insuficiente |

**Objetivo mínimo para aprobar: costo ≤ 1,60.**

Los umbrales no son inventados: salen de medir el desafío. La señal de este
dataset es genuinamente ruidosa, así que un costo de 1,1 es un muy buen
resultado, no uno mediocre.

> Atención: la dirección de esta métrica es **opuesta** a la del Desafío 1
> (donde el score era `1/(1+costo)` y mayor era mejor). Configurá
> `SubmitLog(lower_is_better=True)`.

---

## 4. Formato de envío

Archivo `submit.csv` con dos columnas:

```
id_unidad,estado
400000,Normal
400001,Falla inminente
400002,Vigilancia
...
```

Una fila por cada unidad de `test_features.csv`. El orden no importa: se
vincula por `id_unidad`.

Máximo **10 envíos**, mínimo **5 minutos** entre cada uno. Se toma como
definitivo **el último enviado**.

---

## 5. Cómo correr el ensayo

```bash
cd hackathon-kit
./.venv/Scripts/python.exe tests/make_ensayo.py
```

Arrancá una sesión nueva de Claude Code y pasale `CONTEXTO_CLAUDE.md` + este
documento. Pedile un `STRATEGY.md` antes de que escriba código.

Para medir tu resultado:

```bash
./.venv/Scripts/python.exe tests/score_ensayo.py work/submit.csv
```

---

## 6. Qué se evalúa del ensayo

No es sólo el costo final. Anotá también:

- [ ] ¿Detectaste las trampas **antes** de entrenar, o después de un envío malo?
- [ ] ¿Tu costo OOF predijo el costo real? (gap < 0,05 = arnés honesto)
- [ ] ¿Cuántos envíos simulados necesitaste?
- [ ] ¿Podés reproducir el resultado de cero?
- [ ] ¿Cuánto tardaste hasta el primer `submit.csv` válido?

Referencias medidas sobre este dataset (costo real en el test sellado):

| Enfoque | CV que ves | Costo real |
|---|---|---|
| modelo que usa `orden_trabajo` (la fuga) | **0,214** | **2,877** |
| predecir siempre `Mantenimiento urgente` | — | 1,900 |
| limpio + `argmax` | 1,345 | 1,904 |
| limpio + `bayes_decision` | 0,997 | **1,196** |
| + corrección de prior con EM | — | **1,119** |
| oráculo (conociendo el prior real del test) | — | 1,095 |

Leé esa tabla con atención: **el modelo con la fuga muestra el mejor CV de
todos y el peor resultado real.** Y `argmax` sobre un modelo limpio empata con
predecir siempre la misma clase. Todo el valor está en la capa de decisión.

---

## 7. SOBRE CERRADO — no abrir antes de terminar

<details>
<summary>Trampas plantadas en el dataset (click para revelar)</summary>

1. **Fuga:** `orden_trabajo` codifica el estado en el train
   (`estado*1000 + ruido`) pero está aleatorizada en el test. Un modelo que la
   use da CV casi perfecto y se desploma. → `traps.leakage_scan`, AUC sola 1.0.
2. **Prior shift puro:** el test está enriquecido en casos graves (14 % de
   `Falla inminente` contra 7 % en el train). Es un prior shift *legítimo*:
   `p(x|y)` no cambió, sólo cambiaron las proporciones, así que el EM de
   Saerens lo puede recuperar. → `traps.class_prior_shift` para detectarlo,
   `traps.validate_prior_em` para confirmar que el método sirve acá, y
   `traps.estimate_test_prior` para corregir. Vale 0,08 de costo (1,196 →
   1,119). Ojo: `fit_prior` **no** resuelve esta trampa, porque sólo aprende
   del OOF, que tiene el prior del train.
3. **Fuga por orden:** las filas del train vienen ordenadas por `estado`.
   → `traps.order_leak`.
4. **Ruido de etiquetas:** 5 % de los `estado` del train están mal cargados.
   → `traps.label_noise_scan`; usar `label_smoothing`.
5. **Mezcla de unidades:** 14 % de `temperatura_aceite` está en °F.
   → `cleaning.audit` (`MEZCLA_UNIDADES?`) + `cleaning.fix_units`.
6. **Número como texto:** `gas_c2h2_ppm` viene como `"0,911 ppm"`.
   → `auto_clean` lo convierte solo.
7. **Centinelas:** `rigidez_dielectrica_kv` tiene `-999` en el 6 % de las filas.
8. **Columna constante:** `distribuidora` siempre vale `EMSA`.
9. **Columna que existe sólo en el train:** `inspeccion_visual` está poblada en
   el train y **entera vacía en el test**. Un modelo que se apoye en ella pierde
   su mejor feature justo en la inferencia. → `cleaning.audit` la marca con
   `DRIFT_TEST` / faltante 100 %.
10. **Distractores legítimos:** `marca` con typos y mayúsculas, `humedad` con
    9 % de faltantes. Sucios pero informativos: limpiar, no descartar.

La trampa que más plata cuesta es la **1**: da el mejor CV de todos y el peor
resultado real. La que más difícil es de ver es la **2**, porque no produce
ningún síntoma en el train: sólo aparece si comparás la distribución predicha
sobre el test contra la del train.

Nota honesta sobre el EM de prior: sólo funciona con prior shift *puro*. Si el
desafío real cambia la definición de las clases entre train y test (por ejemplo
moviendo los umbrales que las generan), el EM no ayuda e incluso empeora. Por
eso `validate_prior_em` existe: comprobás antes de aplicar, no después.

</details>
