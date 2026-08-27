# Traspaso de sesión — hackathon-kit

Documento de handoff para **continuar en una sesión nueva** (por ejemplo en la
nube). Una sesión nueva **no tiene la memoria local** de la máquina donde se
armó esto, así que todo el contexto necesario está acá y en `contexto/`.

Si sos una sesión de Claude Code retomando el trabajo: leé esto entero, después
`CONTEXTO_CLAUDE.md` (tu rol operativo) y `PLAYBOOK_HACKATHON.md` (la jornada).

---

## 1. Qué es este proyecto

Kit de herramientas para la **Hackathon IA de la Facultad de Ingeniería (UNaM,
Oberá)** y, de rebote, para los desafíos de la materia **Inteligencia
Computacional (IC415)**. Lo usa el equipo **Acosta · Borges · Pelinski**.

El patrón de estas competencias: dataset dado, problema en contexto, **métrica
de matriz de costos asimétrica** (equivocarse de una manera cuesta más que de
otra), presupuesto de **10 envíos** (5 min entre cada uno, cuenta el último), y
un **probatorio** de 30 min al cierre que justifica las decisiones.

## 2. El evento (de `contexto/reglamento_hackathon.md`)

- **~10,5 h de reloj, ~9 h efectivas**, en dos medias jornadas (jue 08–12 y
  14–18; vie 08–10:30). Presencial, aulas de informática.
- **Grupal**, los grupos se arman ese día.
- **LLMs explícitamente permitidos** (art. 10 nombra a Claude). Prohibido
  compartir entre grupos y la asistencia externa de personas (art. 7).
- Se premia al mejor grupo que supere al modelo base.
- El PDF dice "jueves 28/08" (jueves en 2025, viernes en 2026): **confirmar el
  cronograma con la cátedra.**

## 3. Estado del repositorio

- Repo privado en GitHub:
  `https://github.com/AcostaAlex10/hackathon-kit-Acosta-Borges-Pelinski.git`
- Rama estable: **`herramientas`** (default). Ramas personales: `Acosta`,
  `Borges`, `Pelinski`.
- Regla: nadie commitea directo a `herramientas` durante la competencia; se
  mergea lo probado. Los datos del desafío no van al repo.
- El kit está **probado**: `bootstrap.py` corre un autotest de punta a punta, y
  `BASE_tabular.ipynb` se ejecutó entero (17 celdas, cero errores) contra datos
  sintéticos.

## 4. Qué está construido

Mapa completo en `GUIA_EQUIPO.md`. En una línea cada uno:

- `ic_kit/costs.py` — decisión de mínimo costo esperado (`bayes_decision`), la
  palanca más grande. Matrices `d1` (triaje, 4 clases) y `d2` (especies, 10).
- `ic_kit/cleaning.py` — auditoría de datos sucios y limpieza conservadora.
- `ic_kit/eda.py` — EDA orientado a la métrica, con figuras numeradas.
- `ic_kit/traps.py` — detector de trampas del dataset (correr antes de entrenar).
- `ic_kit/tabular.py` — OOF, ensamble, `fit_prior`, pseudo-labeling.
- `ic_kit/vision.py` — TTA, duplicados por hash, etiquetas ruidosas (Keras en
  los notebooks, timm disponible en el módulo).
- `ic_kit/checkpoints.py` — reanudación tras caída de Colab (escritura atómica).
- `ic_kit/bitacora.py` — registro de hipótesis/hallazgos/decisiones/descartes.
- `ic_kit/submit.py` — validación del CSV y gestión de los 10 envíos.
- `ic_kit/probatorio.py` — arma el probatorio como notebook autocontenido.
- `notebooks/BASE_tabular.ipynb`, `BASE_vision.ipynb` — plantillas de arranque.
- `bootstrap.py` — arranque + autotest en una máquina desconocida.

## 5. Hallazgos de la sesión que conviene no re-derivar

- **La decisión sensible al costo** (no `argmax`) es la mejora individual más
  grande. Medido en datos sintéticos: costo 0.729 → 0.574; y en otro escenario
  1.904 → 1.196.
- **Trampa más cara y silenciosa: el orden de las clases.** Si las etiquetas son
  texto, pandas las ordena alfabéticamente y la matriz de costos queda
  desalineada; el código corre sin error y el puntaje baja ~10 %. Por eso
  `tabular.prepare()` exige `class_order` explícito. Los notebooks base tienen
  `assert` que frenan si no coincide. **Los notebooks de la cátedra declaran el
  orden de clases** — respetarlo tal cual.
- **Una fuga** (columna que predice perfecto en train y está rota en test) da la
  mejor validación cruzada y el peor ranking. Medido: CV 0.214 vs costo real
  2.877. Detectar con `traps.leakage_scan` / `adversarial_validation`.
- **Corrección de prior (EM de Saerens)** sólo sirve para *prior shift puro*;
  con un clasificador débil colapsa las clases raras y empeora. Nunca aplicarla
  a ciegas: `traps.validate_prior_em` lo comprueba antes (devuelve
  SIRVE/MARGINAL/NO SIRVE sobre 20 escenarios).
- **Datos sucios típicos** de estas consignas: números como texto ("36,5 °C"),
  centinelas -999, unidades mezcladas (°C y °F en la misma columna), typos en
  categóricas, columnas pobladas en train y vacías en test. `cleaning.audit`
  los marca.

## 6. Estilo (de `contexto/memoria_ic415.md`)

Se califica el **análisis y la justificación, no el código**. Notebooks en
español, sin emojis, sin citar las preguntas de la consigna, sin explicar
conceptos básicos, sin conclusiones en formato pregunta-respuesta,
`random_state = 42`. Stack Keras/TF (no torch). Todo corre en Colab free tier.

## 7. Cómo continuar en la nube

1. La sesión nueva clona el repo (rama `herramientas`).
2. `python bootstrap.py --install` → tiene que dar `AUTOTEST OK`.
3. Cuando llegue la consigna real: abrir `notebooks/BASE_tabular.ipynb` o
   `BASE_vision.ipynb`, llenar **sólo** la celda de configuración (target, id,
   `CLASES` en el orden de la consigna, matriz de costos), y seguir el recorrido.
4. Seguir `PLAYBOOK_HACKATHON.md` para el cronograma, los roles y los envíos.

## 8. Pendientes / cabos sueltos

- **Push del último trabajo:** verificar que `herramientas` en GitHub esté al
  día (`git log origin/herramientas`). Si esta limpieza se hizo local, falta
  pushear.
- **Colaboradores:** agregar a Borges y Pelinski al repo (Settings →
  Collaborators) para que puedan clonar el privado.
- **Default branch en GitHub:** dejar `herramientas` como default (Settings).
- **`BASE_vision.ipynb` no se ejecutó de punta a punta** (no hay GPU local): dar
  una corrida en Colab con un dataset chico antes del día.
- **Datos del desafío:** se comparten aparte del repo (van en `.gitignore`).

## 9. Material fuente en `contexto/`

- `reglamento_hackathon.md` — el reglamento y cronograma oficial, fiel al PDF.
- `memoria_ic415.md` — quiénes somos, cómo se califica, convenciones de estilo.
- `desafios_previos.md` — los dos desafíos ya tomados (d1 triaje, d2 especies),
  con sus matrices de costo y formato.
- `notebooks_catedra/` — los dos notebooks base originales de la cátedra, de los
  que salieron `notebooks/BASE_*.ipynb`.
