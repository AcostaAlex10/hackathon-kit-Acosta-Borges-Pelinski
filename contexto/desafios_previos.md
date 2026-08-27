# Desafíos previos de IC415 (2026)

Los dos desafíos que la cátedra ya tomó, con el mismo modus operandi que tendrá
la hackathon: dataset dado, problema en contexto, ~2 días, métrica de costo
asimétrico, servidor de ranking (`http://192.168.25.239:8000/`), 10 envíos,
cuenta el último. Sirven de referencia del formato y como datos de práctica.
Sus matrices están codificadas en `ic_kit/costs.py` como `d1` y `d2`.

Los notebooks base originales de la cátedra están en
`contexto/notebooks_catedra/`. Ya fueron destilados en `notebooks/BASE_*.ipynb`.

---

## Desafío 1 — Triaje en Admisiones Hospitalarias

Estimar el nivel de urgencia clínica al ingreso. **4 clases**, con codificación
que usa el servidor:

- 0: No urgente · 1: Urgente · 2: Muy urgente · 3: Crítico

**Métrica:** `score = 1 / (1 + costo_promedio)`, **mayor es mejor** (máx 1,0).
Mínimo para aprobar: 0,7. Matriz de costos `C[real, predicho]`:

| Real \ Predicho | No urgente | Urgente | Muy urgente | Crítico |
|---|---|---|---|---|
| No urgente | 0 | 1 | 2 | 4 |
| Urgente | 2 | 0 | 1 | 3 |
| Muy urgente | 4 | 2 | 0 | 2 |
| Crítico | **10** | 6 | 2 | 0 |

Lo caro es clasificar como "No urgente" a un "Crítico" (costo 10). Datos
tabulares (~29 columnas: signos vitales, síntomas, datos del paciente, contexto
operativo), cargados en parte a mano, con errores y omisiones. Archivos:
`train_labeled.csv`, `train_unlabeled.csv`, `test_features.csv`. Submit:
`id_paciente,nivel_urgencia`. Baseline de cátedra (árbol de decisión): 0,6662.

En `costs.py`: `COST_D1`, `LABELS_D1`.

---

## Desafío 2 — Especies Nativas del Bosque Atlántico

Visión por computadora: identificar **10 especies** (5 Animalia, 5 Plantae) de
fauna y flora a partir de imágenes en condiciones naturales.

| Clase | Etiqueta | Reino |
|---|---|---|
| Pecari_tajacu | 0 | Animalia |
| Nasua_nasua | 1 | Animalia |
| Harpia_harpyja | 2 | Animalia |
| Caiman_latirostris | 3 | Animalia |
| Phyllomedusa_distincta | 4 | Animalia |
| Aspidosperma_polyneuron | 5 | Plantae |
| Philodendron_bipinnatifidum | 6 | Plantae |
| Dicksonia_sellowiana | 7 | Plantae |
| Aechmea_distichantha | 8 | Plantae |
| Ilex_paraguariensis | 9 | Plantae |

**Métrica:** costo promedio por imagen, **menor es mejor** (mín 0, máx 5).
Objetivo para aprobar: 0,7. Penalización:

- Correcta: 0
- Incorrecta, mismo reino: 2
- Incorrecta, distinto reino: 5

Favorece modelos que, al fallar, lo hagan con menor distancia taxonómica.
Carpetas `train/` (subdirectorios por clase) y `only_submit/`. Submit:
`Image,Predict` con el entero 0–9. Set desbalanceado (de 90 a 724 imágenes por
clase), con variabilidad alta de iluminación, fondo y calidad.

En `costs.py`: `COST_D2`, `KINGDOM_D2` (= `grouped_cost_matrix` con
same_group=2, diff_group=5).

---

## Qué tienen en común (lo que se repite en la hackathon)

- Dataset dado + problema en contexto + métrica de **costo asimétrico**.
- **10 envíos, 5 min entre cada uno, cuenta el último.**
- Datos sucios "cargados a mano, no se descarta la existencia de errores".
- Un `train_unlabeled` opcional para semi/no supervisado.
- Prohibido usar el test para ajustar o validar.

La hackathon puede ser tabular o de imágenes, o algo distinto. El kit cubre las
dos primeras; para otra cosa, lo único que cambia es la matriz de costos.
