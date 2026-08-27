# hackathon-kit
**Equipo:** Acosta · Borges · Pelinski
```bash
git clone https://github.com/AcostaAlex10/hackathon-kit-Acosta-Borges-Pelinski.git
cd hackathon-kit-Acosta-Borges-Pelinski
python bootstrap.py --install
```

Si termina con `AUTOTEST OK`, el kit funciona en esa máquina. Si LightGBM no
instala, degrada solo a scikit-learn y sigue andando.

**Empezá por [GUIA_EQUIPO.md](GUIA_EQUIPO.md).**

---

## Estructura

```
├── GUIA_EQUIPO.md           <- LEER PRIMERO. Qué hace cada cosa y cómo se usa.
├── PLAYBOOK_HACKATHON.md    <- cronograma real de la jornada, roles, envíos
├── CONTEXTO_CLAUDE.md       <- para pasarle a una sesión de Claude Code
├── TRASPASO.md              <- estado del proyecto y handoff entre sesiones
├── bootstrap.py             <- arranque + autotest en una máquina desconocida
├── requirements.txt
├── run_tabular.py           <- baseline tabular completo, de CSV a submit.csv
├── run_vision.py            <- baseline de imágenes (Colab con GPU)
├── notebooks/
│   ├── BASE_tabular.ipynb   <- plantilla de arranque para desafío con CSV
│   ├── BASE_vision.ipynb    <- plantilla para imágenes (Keras, stack de cátedra)
│   └── arranque_colab.ipynb <- esqueleto mínimo para armar a mano
├── ic_kit/
│   ├── costs.py             matrices de costo + decisión de mínimo costo
│   ├── cleaning.py          auditoría y limpieza de datos sucios
│   ├── eda.py               EDA prearmado orientado a la métrica
│   ├── traps.py             detector de trampas del dataset
│   ├── tabular.py           OOF, blend, fit_prior, pseudo-labeling
│   ├── vision.py            TTA, duplicados, etiquetas ruidosas
│   ├── checkpoints.py       reanudación tras caída de Colab
│   ├── bitacora.py          registro de hipótesis/hallazgos/decisiones
│   ├── submit.py            validador + gestor de los 10 envíos
│   └── probatorio.py        probatorio como notebook autocontenido
└── contexto/                material fuente: reglamento, memoria, desafíos previos
```

`work/`, `.venv/` y `data/` quedan fuera del repo (`.gitignore`). El dataset del
desafío **nunca** va al repo.

## Uso en 30 segundos

```bash
python run_tabular.py --train data/train.csv --test data/test.csv \
    --target nivel_urgencia --id id_paciente --cost d1 --out work/submit.csv
```

En una corrida: auditoría de columnas sucias, limpieza, dos modelos, blend,
decisión óptima según costo, informe de dónde se va el costo, `submit.csv` y su
validación.

Para la jornada real, arrancá desde `notebooks/BASE_tabular.ipynb` (o
`BASE_vision.ipynb`): traen el recorrido completo y sólo se toca la celda de
configuración.

## Lo que hace distinto a este kit

**Decisión sensible al costo.** Con matriz de costos asimétrica, `argmax p(y|x)`
no es óptimo. Lo óptimo es `argmin_j Σ_i p(i|x)·C[i,j]`
(`costs.bayes_decision`). 

**Detección de trampas.** `traps.run_all()` busca, antes de entrenar, las
trampas que castigan a las soluciones automáticas: fugas, covariate shift, prior
shift, orden de filas, atajos de fondo en imágenes, ruido de etiquetas.

**Gestión de envíos y probatorio.** `SubmitLog` lleva el presupuesto de 10
envíos y la correlación validación↔ranking; `probatorio.generar_notebook()`
arma el entregable del art.

## Adaptar a un desafío nuevo

Lo único específico es la matriz de costos. En `ic_kit/costs.py`, o como CSV:

```python
MI_COSTO = np.array([[0, 1, 5],
                     [2, 0, 1],
                     [8, 3, 0]], dtype=float)   # C[real, predicho]
```

y `--cost ruta.csv` más `--classes "a;b;c"` **en el mismo orden que las filas de
la matriz**.
