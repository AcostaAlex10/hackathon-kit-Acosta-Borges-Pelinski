# Playbook de 48 h — desafíos IC415

> Regla que ordena todo: **el que gana no es el que entrena el mejor modelo,
> es el que entiende la métrica primero y no desperdicia envíos.**

---

## 0. Las tres cosas que deciden el resultado

| # | Palanca | Ganancia típica | Costo en tiempo |
|---|---------|-----------------|-----------------|
| 1 | Decisión bayesiana sobre la matriz de costos (no `argmax`) | **grande** | 10 min |
| 2 | Validación honesta (OOF que correlaciona con el leaderboard) | evita perder | 30 min |
| 3 | Limpieza de los datos sucios que avisa la consigna | grande | 1–2 h |
| 4 | Ensamble de semillas/modelos | media | 1 h |
| 5 | Tuneo de hiperparámetros | **chica** | horas |

En el ensayo con datos sintéticos del kit, la palanca 1 sola bajó el costo de
`0.729` a `0.574` — de score **0.578 a 0.635**. Cero cambios en el modelo.

---

## 1. Hora 0–1: reconocimiento (NO entrenar todavía)

1. Leé la consigna dos veces y **copiá la matriz de costos a mano** en
   `ic_kit/costs.py`. Anotá el orden EXACTO de las clases.
2. `python run_tabular.py ... --cost d1` (o el que corresponda) con `--seeds 1
   --folds 3` para tener un número honesto rápido.
3. Mirá `work/audit.csv`. Buscá: `CONSTANTE`, `ID?`, `NUMERO_COMO_TEXTO`,
   `CENTINELA`, `MEZCLA_UNIDADES?`, `DRIFT_TEST`.
4. Respondé por escrito estas cuatro preguntas antes de seguir:
   - ¿mayor o menor es mejor?
   - ¿cuál es el error más caro y qué proporción de casos lo sufre?
   - ¿la distribución de clases del train se parece a la del test?
   - ¿hay alguna columna que sea una fuga (leak) disfrazada?

> **Trampa nº 1 y la más cara:** el orden de las clases.
> Si tus etiquetas son texto, pandas las ordena alfabéticamente
> (`Crítico, Muy urgente, No urgente, Urgente`) y tu matriz de costos está en
> orden clínico (`No urgente, Urgente, Muy urgente, Crítico`). El código corre
> perfecto y el score baja ~10 %. Por eso `prepare()` exige `class_order`.

---

## 2. Hora 1–4: baseline sólido + primer envío

- Baseline con `run_tabular.py` completo (5 folds, 3 semillas).
- **Primer envío temprano.** Sirve para calibrar el gap CV↔LB, que es
  información que después no podés comprar. Registralo:

```bash
python -c "from ic_kit.submit import SubmitLog; SubmitLog().record('work/submit.csv', cv=0.635, notes='baseline bayes')"
```

- Apenas veas el número del leaderboard, cargalo con `set_lb(1, 0.6295)`.
  Con 3 envíos ya tenés la correlación CV↔LB. **Si r < 0.5, tu CV está mal
  armado**: parás todo y arreglás el CV antes de tocar el modelo.

---

## 3. Hora 4–20: mejoras, de la más barata a la más cara

Orden estricto. No pasás al siguiente hasta agotar el anterior:

1. **Limpieza dirigida** con lo que marcó la auditoría. Unidades mezcladas
   (`cleaning.fix_units`), rangos imposibles (`ranges={'temperatura': (30,45)}`),
   categorías con typos.
2. **Feature engineering de dominio.** En triaje: shock index (FC/PAS), qSOFA,
   flags de fiebre/hipotermia, saturación < 92, lactato > 2, edad en bins.
   Cinco features de dominio > cincuenta features automáticas.
3. **Ensamble**: LightGBM + CatBoost + un modelo distinto (Random Forest o
   red neuronal). `tabular.blend_weights` busca los pesos que minimizan el
   costo, no el logloss.
4. **`fit_prior`** al final, siempre sobre OOF. Es gratis y suele dar la última
   décima.
5. Recién ahora, tuneo de hiperparámetros. Y con un límite de tiempo puesto.

En **visión**: backbone más grande > más épocas > TTA > más folds > tuneo.
Y antes que todo eso: revisá las 40 imágenes de
`work/etiquetas_sospechosas.csv`. La consigna dice "un alto porcentaje fue
validado por expertos" — el resto no lo fue.

---

## 4. Gestión de los 10 envíos

Presupuesto sugerido:

| Envíos | Para qué |
|--------|----------|
| 1 | baseline (calibrar el gap CV↔LB) |
| 2–3 | mejoras grandes ya validadas en OOF |
| 4–6 | variantes que en OOF empatan (el LB desempata) |
| 7–8 | ensamble final |
| 9 | reserva por si el 8 sale mal |
| 10 | **reenvío del mejor archivo** — cuenta el ÚLTIMO, no el mejor |

Reglas que te ahorran intentos:

- Si `diff_vs()` dice que cambia **< 1 %** de las filas, no gastes un envío.
- Si tu OOF mejora **menos que el desvío entre folds**, es ruido. No lo envíes.
- **El último envío del día debe ser tu mejor archivo, aunque ya lo hayas
  mandado.** Es el error más frecuente de la jornada: la gente prueba algo
  arriesgado al final y se queda con eso.
- Dejate 20 minutos de margen. Si el server se cae a último momento, perdiste.

---

## 5. Errores concretos que vi caer en estas competencias

| Error | Síntoma | Antídoto en el kit |
|-------|---------|--------------------|
| Orden de clases desalineado con la matriz | corre bien, score bajo | `prepare(class_order=...)` |
| Optimizar accuracy cuando la métrica es costo | CV linda, LB fea | `costs.bayes_decision` |
| Duplicados repartidos entre folds | CV altísimo, LB baja | `vision.duplicate_groups` + `StratifiedGroupKFold` |
| Imputar la media en datos sucios | pérdida silenciosa de señal | `auto_clean` deja NaN + flags `__isna` |
| Usar el test para elegir algo | te ilusionás y perdés | prohibido por consigna; `oof.npy` es tu único juez |
| Ajustar umbrales sobre el propio OOF muchas veces | sobreajuste al OOF | `fit_prior` con pocas rondas; verificá en un fold aparte |
| Perder el `submit.csv` que dio el mejor LB | irreparable | `SubmitLog` guarda md5 y notas |
| No poder replicar la solución para Moodle | nota más baja | `work/report.json` + semillas fijas |

---

## 6. Checklist de los últimos 60 minutos

- [ ] `submit.csv` valida: filas = filas del test, sin NaN, sin ids de más
- [ ] la distribución predicha no es absurda vs el train
- [ ] el archivo que voy a mandar es el de **mejor LB**, no el más nuevo
- [ ] notebook/script corre de cero y reproduce el archivo (semillas fijas)
- [ ] `report.json`, `audit.csv` y el log de envíos guardados para Moodle
- [ ] carpeta comprimida y subida a Moodle **antes** del último envío

---

## 7. Si el desafío no es ninguno de los dos tipos

El kit sigue sirviendo. Lo único específico de cada desafío es la matriz de
costos:

- **Regresión**: reemplazá la matriz por la función de pérdida y optimizá la
  predicción puntual que minimiza esa pérdida (para MAE es la mediana, no la
  media; para pérdidas asimétricas es un cuantil).
- **Ranking / detección**: la decisión sigue siendo un umbral. `fit_prior` se
  generaliza a buscar el umbral que minimiza el costo esperado en OOF.
- **Series temporales**: cambiá `StratifiedKFold` por `TimeSeriesSplit`. Todo
  lo demás queda igual. Nunca uses folds aleatorios con datos temporales.
