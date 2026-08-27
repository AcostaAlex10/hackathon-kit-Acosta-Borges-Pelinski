# Estrategia 1 — Hackathon IA "Machine Health" (traspaso entre sesiones)

> Documento de handoff. Una sesión nueva de Claude debe **leer esto entero**
> antes de tocar nada, y después `CONTEXTO_CLAUDE.md` + `PLAYBOOK_HACKATHON.md`.
> Todo lo aquí escrito ya está **deducido y medido** en esta sesión; no re-derivar.
> Rama de trabajo: `claude/hackaton-repo-context-jcsi7k`. Equipo: Acosta · Borges · Pelinski.

---

## 0. Estado y lo primero que hay que hacer al retomar

**Lo que ya se hizo:** análisis del caso, EDA completo, disección de la métrica
oficial (`scoring.py`) y un primer modelo local honesto (**final 40.2** con
LightGBM crudo). NO se escribió código de solución todavía (freno por tokens).

**BLOQUEO al retomar — leer:** los datos y el kit oficial llegaron como ZIP
subidos por el usuario y viven en `work/` (que está en `.gitignore`, así que
**no se versionan** y el contenedor es efímero). La sesión nueva **no los
tendrá**. Pedir al usuario que vuelva a subir:
- `datos_sinraw.zip`  → contiene `train_sup`, `test`, `train_unlabeled`,
  `raw_signal_index`, `data_dictionary.csv` (CSV y Parquet).
- `participant_kit.zip` → contiene **`scoring.py`** (métrica oficial, crítico),
  `baseline.py`, `baseline.ipynb`, `eda_intro.ipynb`, `convert_raw_to_csv.py`.

Reconstrucción del entorno (contenedor limpio):
```bash
pip install -q pandas numpy scikit-learn pyarrow scipy lightgbm
# descomprimir los dos zips en work/desafio/ como estaban:
#   work/desafio/datos_zip/datos_sinraw/*
#   work/desafio/kit_zip/participant_kit/*  (tiene scoring.py)
```
Hay un script de sondeo ya escrito: `work/desafio/probe.py` (reproduce el 40.2).
`work/` no se pushea; si se quiere conservar, copiarlo fuera de `work/`.

**Nota PDF en este entorno:** el Read de PDF necesita poppler (ausente). Para
leer la consigna usar pypdf tras `pip install --force-reinstall cffi` (el
binding de `cryptography` del sistema panica sin eso).

---

## 1. El caso en tres líneas

Mantenimiento predictivo de máquinas rotativas (motor + bomba). **Multi-label:**
por cada ventana temporal hay que estimar **13 probabilidades independientes**
(F01–F13), una por modo de falla. Salida = CSV `window_id,F01..F13` con probas
en [0,1]. Envíos múltiples, **cuenta el último**. Prohibido usar test para
entrenar/ajustar/etiquetar. Señales crudas NPZ **opcionales** (arrancar sin
ellas). Ranking en `http://192.168.25.239:8000/`.

Familias de falla (importan para la métrica):
- **Mecánica:** F01 desbalance, F02 desalineación, F03 rodamiento pista ext,
  F04 rodamiento pista int, F05 elemento rodante, F06 lubricación.
- **Estructural:** F07 rigidez/tornillería.
- **Eléctrica:** F08 desequilibrio tensión, F09 pérdida de fase, F10 barras
  rotóricas, F11 cortocircuito espiras.
- **Hidráulica:** F12 cavitación, F13 obstrucción/fuga.

Datos: `train_sup` (1750×77, 26 máquinas, 250 sesiones, con `label_F*` y
`severity_F*`), `test` (1848×49, 22 máquinas), `train_unlabeled` (434×49).
46 features numéricas float, patrón `<sensor>__<métrica>`. Ids: `window_id`
(único), `machine_id`, `session_id` — **presentes también en test**.

---

## 2. La métrica (`scoring.py`) — donde vive el 70%. LEER CON CUIDADO.

```
final = 0.70·cost_score + 0.20·prob_score + 0.10·diag_score      (mayor mejor, 0–100)
cost_score = 100·max(0, 1 − mean_cost/naive_cost)
prob_score = 100·max(0, 1 − brier/0.25)      # brier = media (p−y)²
diag_score = 100·macro_f1(umbral 0.5)
```

**cost_score (70%) es una capa de decisión sobre NUESTRAS probas.** Para cada
ventana, el scorer elige la acción de mínimo costo esperado (`choose_action`),
usando SOLO sumas de nuestras probas. Con `S = Σ_13 p_f` y
`m_g = Σ p_f en familia g`:

| Acción | Costo esperado que usa el scorer para decidir |
|---|---|
| MONITOR | `10·S` |
| DERATE | `7 + 3.5·S` |
| STOP | `12` |
| INSPECT_g (por familia) | `4 + 8·(S − m_g)` |

Costos **reales** que se cobran según la verdad (`_cost_of_action`), con
`missed = 15 si severidad>0.5, else 5`:
- Sana + MONITOR → 0. Sana + (INSPECT/DERATE/STOP) → 4/7/12.
- Faltosa + MONITOR → missed (5 ó 15). + STOP → 12. + DERATE → 7+0.35·missed.
- Faltosa + INSPECT familia correcta → `4 + (fracción de otras familias)·missed`
  (en train, mono-falla → **4 exacto**). + INSPECT familia incorrecta → `4+8 = 12`.
- `naive_cost` = costo de MONITOR siempre (referencia del denominador).

### Implicaciones estratégicas (esto es lo importante)
1. El scorer **solo hace INSPECT en una ventana faltosa si S > ~0.4**. Un modelo
   calibrado-para-Brier predice probas **bajas** (clases raras ~5–10%) → el
   scorer se queda en MONITOR → comemos el falso negativo de 5/15.
2. **Tensión real entre componentes:** prob_score (20%) premia probas bajas y
   honestas; cost_score (70%) premia **masa suficiente para gatillar INSPECT en
   la familia correcta**. Como el scorer es **determinista y totalmente
   conocido**, la jugada es **optimizar la transformación de probas post-hoc
   sobre el OOF contra el scorer real** (análogo a `bayes_decision`/`fit_prior`
   del kit, adaptado a este costo acción×resultado). ESTA ES LA PALANCA #1.
3. **Techo del scorer:** oráculo (prob=label) da cost_score = **62.8** (no 100):
   acertando todo igual se paga la inspección. cost_score realista ≈ 35–45.
4. prob_score arranca en ~75 casi gratis (fallas raras). diag_score exige
   empujar probas de positivos >0.5 en clases raras: caro y solo 10%.

Constantes de `scoring.py`: `false_positive=1, fn_mild=5, fn_severe=15,
misdiagnosis=8, inspect=4, derate=7, stop=12`. Import: agregar
`sys.path.insert(0,"kit_zip/participant_kit")` y `from scoring import
compute_score, choose_action, FAULT_IDS, FAMILIES`.

---

## 3. EDA — hallazgos que cambian el diseño (con evidencia)

| # | Hallazgo | Evidencia medida | Decisión |
|---|---|---|---|
| 1 | **Las 22 máquinas de test son 100% nuevas** | 0 de 22 `machine_id` de test en train; 0 sesiones compartidas; 0 window_id compartidos | **Validación = `GroupKFold` por `machine_id`, siempre.** Nunca usar machine_id/session_id como feature. |
| 2 | **Covariate shift severo, de régimen** | AUC adversarial train-vs-test = **0.982**. Drivers: `voltage_*__rms`, `rpm_mean` (test 2052±432 vs train 1742±465), `flow_mean`, `delta_p_mean`, `V_rms_mean` | El test corre a **más rpm/carga**. → FE de **normalización por régimen** (ratios y features relativas a rpm/carga) es la palanca #2. |
| 3 | **Train es mono-falla puro** | `is_combo=0` en todo train_sup; nº fallas/ventana ∈ {0 (217=12.4%), 1 (1533=87.6%)} | Cada ventana faltosa tiene 1 sola familia → en OOF, INSPECT correcto cuesta 4 fijo. (Test podría traer combos; predecir 13 independientes igual.) |
| 4 | **57.5% de las fallas son severas** (sev>0.5) | severidad media 0.571 donde label=1 | Mayoría de FN cuestan **$15**. naive_cost alto → mucho margen para la capa de decisión. |
| 5 | **Datos limpios** | 0 nulos en features; sin centinelas (-999); sin números-como-texto | El playbook "datos sucios" del kit **no aplica**. Únicos detalles: 3 cols `temp_*__nan_frac` constantes (tirar); `*__nan_frac` sirve como proxy de calidad de señal. |
| 6 | Prevalencias 4.8%–9.6% | F01/F02 0.096 … F09/F10 0.048 | Multilabel desbalanceado. Jamás accuracy. |
| 7 | Ventanas/máquina: test uniforme 84; train 49–84 (media 67) | groupby machine_id | — |

---

## 4. Primer número local (infra ya validada)

`work/desafio/probe.py`: GroupKFold(5) por máquina, LightGBM binario por falla,
**sin tuning, sin calibración, sin features nuevas**. Scorer oficial sobre el OOF:

| Modelo | final | cost | prob | diag | brier | f1 |
|---|---|---|---|---|---|---|
| P0 prevalencia (baseline) | 30.8 | 22.6 | 75.0 | 0.0 | 0.062 | 0.00 |
| **LightGBM crudo** | **40.2** | 30.3 | 78.9 | 32.1 | 0.053 | 0.32 |
| LightGBM ^0.35 (temperatura) | 41.3 | 31.6 | 78.2 | 35.5 | 0.054 | 0.36 |
| Oráculo (prob=label, techo scorer) | 74.0 | 62.8 | 100 | 100 | 0.000 | 1.00 |

Referencia consigna: baseline 25–35 · buen modelo 45–55 · excelente >55.
Ya estamos en 40 con lo mínimo. **Cuello de botella: cost_score (30 vs techo 63).**
El sharpening ingenuo (×g lineal, ^T temperatura) da poco; el `^0.35` sube f1.
La ganancia real está en optimizar la transformación contra el scorer (§2.2).

Params LightGBM usados: `objective=binary, n_estimators=300, lr=0.03,
num_leaves=31, subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
seed=42`. Features = las 46 (excluir ids, label_*, severity_*, is_combo, is_normal).

---

## 5. Plan de construcción (por rentabilidad — NO empezado)

1. **Capa de decisión optimizada contra el scorer** (palanca #1, 70%).
   Módulo nuevo `ic_kit/machine_health.py`: wrapper multilabel + buscador, sobre
   OOF, de la transformación de probas (por-familia / temperatura / umbral) que
   maximiza `0.7·cost + 0.2·prob + 0.1·diag` con `compute_score`. Aplicar la
   misma transformación al test. Es el `bayes_decision` de este desafío.
2. **Calibración** por falla (isotónica/Platt) dentro de cada fold. Sube
   prob_score y da masa honesta a la capa de decisión. Es hueco del kit.
3. **Feature engineering de normalización** (palanca #2): atacar el shift de
   régimen. Ratios, features relativas a rpm/carga, armónicos ya son relativos
   (1x/2x), z-score por sesión, `nan_frac` como calidad. Reduce el gap a
   máquinas nuevas (validar con la AUC adversarial y con el OOF GroupKFold).
4. **Ensamble** LightGBM + HistGBM/ExtraTrees; blend sobre OOF optimizando el
   scorer (no logloss).
5. Recién ahí, hiperparámetros con límite de tiempo.
6. Si sobra tiempo y el techo tabular se estanca: features de señales crudas NPZ
   (FFT/Welch, bandas de rodamiento) vía `convert_raw_to_csv.py` /
   `raw_signal_index`. **Opcional y último.**

**Reutilizar del repo:** `traps.adversarial_validation` (ya confirmó shift),
patrón OOF de `tabular.py` (envolver a multilabel), `bitacora.py` para el
probatorio, `cleaning.audit` (rápido, aunque los datos están limpios).
**Construir nuevo:** wrapper multilabel + capa de decisión contra el scorer +
calibración. El `submit.py` del repo se reemplaza por el
`validate_prediction_package` oficial de `scoring.py` para el contrato de entrega.

---

## 6. Reglas que no se negocian (recordatorio del kit + de la consigna)

- Nunca entrenar/validar/ajustar con el test; nunca etiquetar test a mano.
- CV **siempre GroupKFold por machine_id** (test = máquinas nuevas).
- Decidir contra el scorer real (`compute_score`), nunca por accuracy/logloss.
- `random_state=42`. Reportar resultados como son (si el CV baja, se dice).
- Estilo probatorio: español, sin emojis, sin citar la consigna, conciso; se
  califica el análisis y la justificación, no el código. Ir anotando hipótesis /
  hallazgos / decisiones / descartes **durante** la jornada (bitácora).
- El último envío debe ser el mejor archivo.
