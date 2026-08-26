# hackathon-kit

Herramientas para competencias de ML con **métrica de costo asimétrico**,
presupuesto limitado de envíos y poco tiempo. Hackathon IA — FI-UNaM.

**Equipo:** Acosta · Borges · Pelinski

```bash
git clone <URL> hackathon-kit && cd hackathon-kit
python bootstrap.py --install
```

Si termina con `AUTOTEST OK`, el kit funciona en esa máquina.
**Empezá por [GUIA_EQUIPO.md](GUIA_EQUIPO.md).**

---

Probado de punta a punta sobre un desafío sintético que imita el Desafío 1
(triaje) y sobre un ensayo con diez trampas plantadas.

```
hackathon-kit/
├── GUIA_EQUIPO.md           <- LEER PRIMERO. Qué hace cada cosa y cómo se usa.
├── PLAYBOOK_HACKATHON.md    <- cronograma real de la Hackathon FI-UNaM (~10,5 h)
├── PLAYBOOK.md              <- estrategia de 48 h (desafíos de cátedra)
├── bootstrap.py             <- arranque + autotest en una máquina desconocida
├── run_tabular.py           <- baseline tabular completo (Desafío 1)
├── run_vision.py            <- baseline de imágenes (Desafío 2), para Colab
├── ic_kit/
│   ├── costs.py             matrices de costo + decisión bayesiana
│   ├── cleaning.py          auditoría y limpieza de datos sucios
│   ├── tabular.py           OOF, blend, fit_prior, pseudo-labeling
│   ├── vision.py            timm, TTA, duplicados, etiquetas ruidosas
│   ├── traps.py             detector de trampas del dataset
│   ├── probatorio.py        probatorio como notebook autocontenido
│   └── submit.py            validador + gestor de los 10 envíos
├── CONTEXTO_CLAUDE.md       <- para arrancar una sesión nueva de Claude Code
├── ENSAYO_DESAFIO.md        <- simulacro completo con trampas, para practicar
├── notebooks/arranque_colab.ipynb <- entrada para sesión en la nube
├── requirements.txt
├── tests/make_fake_challenge.py   <- desafío falso para ensayar
└── .venv/                   entorno con numpy/pandas/sklearn/lightgbm
```

## Uso en 30 segundos

```bash
cd hackathon-kit
./.venv/Scripts/python.exe run_tabular.py \
    --train data/train_labeled.csv --test data/test_features.csv \
    --target nivel_urgencia --id id_paciente --cost d1 --out work/submit.csv
```

Eso te da, en una corrida: auditoría de columnas sucias, limpieza, dos modelos,
blend, decisión óptima según costo, informe de dónde se te va el costo,
`submit.csv` y su validación.

## Ensayo general (recomendado antes de la hackaton)

```bash
./.venv/Scripts/python.exe tests/make_fake_challenge.py
./.venv/Scripts/python.exe run_tabular.py --train tests/fake_data/train_labeled.csv \
    --test tests/fake_data/test_features.csv --target nivel_urgencia \
    --id id_paciente --cost d1 --out work/submit_fake.csv
```

`tests/fake_data/_solucion_oculta.csv` te deja medir tu score real y comprobar
que tu OOF no miente. En el ensayo el OOF dio `0.574` y el costo real `0.589`
— gap de 0.015, o sea el arnés de validación es honesto.

## Lo que hace distinto a este kit

**Decisión bayesiana.** Con matriz de costos asimétrica, `argmax p(y|x)` no es
óptimo. Lo óptimo es `argmin_j Σ_i p(i|x)·C[i,j]`, que es
`costs.bayes_decision(proba, C)`. En el ensayo: costo 0.729 → 0.574 sin tocar
el modelo.

**Auditoría de datos sucios.** Las dos consignas avisan que hay errores de
carga. `cleaning.audit()` detecta números guardados como texto (`"1,25 mmol/L"`),
centinelas (`-999`), mezcla de unidades (temperatura en °C y °F en la misma
columna), typos en categóricas y drift train↔test.

**Gestión de envíos.** `SubmitLog` lleva el presupuesto (10 máx., 5 min entre
envíos), avisa si el archivo es idéntico a uno anterior, y calcula la
correlación CV↔leaderboard — que es lo que te dice si podés confiar en tu
validación.

## Colab (Desafío tipo 2, con GPU)

```python
!pip -q install timm
!python run_vision.py --train data/train --submit data/only_submit \
    --cost d2 --model convnext_tiny.fb_in22k --size 320 --epochs 8 --folds 3
```

Runtime → Cambiar tipo de entorno → GPU. Con T4, `convnext_tiny` a 320 px y
~2000 imágenes tarda unos 6-8 min por fold.

## Adaptar a un desafío nuevo

Lo único específico es la matriz de costos. En `ic_kit/costs.py`:

```python
MI_COSTO = np.array([[0, 1, 5],
                     [2, 0, 1],
                     [8, 3, 0]], dtype=float)   # C[real, predicho]
```

y pasás `--cost ruta/a/matriz.csv` (sin encabezados) más
`--classes "clase_a;clase_b;clase_c"` **en el mismo orden que las filas de la
matriz**.
