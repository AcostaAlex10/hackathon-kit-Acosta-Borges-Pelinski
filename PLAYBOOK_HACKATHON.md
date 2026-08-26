# Playbook de la Hackathon IA — UNaM FI

> Reemplaza al `PLAYBOOK.md` de 48 h. **Este evento no dura dos días.**
> Salió del reglamento y cronograma oficial.

---

## 0. El dato que cambia todo

| | 48 h (lo que suponíamos) | La hackathon real |
|---|---|---|
| Tiempo de trabajo | ~48 h | **~10,5 h en dos medias jornadas** |
| Modalidad | en casa | **presencial, aulas de informática, 3er piso** |
| Formato | individual | **grupal, y los grupos se arman ese día** |
| Entregable extra | — | **probatorio, 30 min después del cierre** |

Cronograma según el reglamento:

| Día | Horario | Bloque |
|---|---|---|
| Jueves | 08:00 – 12:00 | sesión de trabajo (4 h) |
| Jueves | 12:00 – 14:00 | almuerzo |
| Jueves | 14:00 – 18:00 | sesión de trabajo (4 h) |
| Viernes | 08:00 – 10:30 | sesión de trabajo (2,5 h) |
| Viernes | 10:30 – 12:00 | puesta en común |

Descontando la presentación de la consigna, el armado de grupos y los 30 min
del probatorio, quedan **unas 9 h efectivas**. Eso obliga a un orden distinto:
no hay tiempo para explorar cómodo y después apurar la entrega. **El baseline
tiene que estar enviado antes del almuerzo del jueves.**

> Verificar la fecha: el PDF dice "jueves 28/08", que fue jueves en 2025. En
> 2026 el 28/08 cae viernes. Confirmá el cronograma vigente con la cátedra.

---

## 1. Antes de ir (los días previos)

- [ ] Correr `python bootstrap.py` en la máquina que vas a usar. Si es una de
      la facultad, llevá el kit en pendrive y corré `python bootstrap.py --install`.
- [ ] Hacer el simulacro completo de `ENSAYO_DESAFIO.md` **cronometrado a 3 h**.
      No para ganarlo: para que el flujo te salga sin pensar.
- [ ] Tener una cuenta de Colab lista y probada, por si la máquina del aula no
      deja instalar nada.
- [ ] Llevar el kit en pendrive **y** en el drive. La red del aula puede fallar.

---

## 2. Jueves 08:00 – 12:00 — de cero a un envío

| Reloj | Qué |
|---|---|
| 08:00 | consigna, armado de grupos, reparto de roles |
| 08:20 | `bootstrap.py` en las tres máquinas, en paralelo |
| 08:30 | transcribir la matriz de costos **a mano** y fijar el orden de clases |
| 08:45 | `cleaning.audit()` + `traps.run_all()` |
| 09:30 | `run_tabular.py` completo |
| 10:15 | **primer envío** y registro en `SubmitLog` |
| 10:30 | cargar el puntaje del leaderboard; comparar con el OOF |
| 10:45 – 12:00 | atacar lo que marcó la auditoría |

Las cuatro preguntas que hay que responder por escrito antes de entrenar:

1. ¿Mayor o menor es mejor?
2. ¿Cuál es el error más caro y qué fracción de casos lo sufre?
3. ¿La distribución de clases del train se parece a la del test?
4. ¿Hay alguna columna que sea una fuga disfrazada?

---

## 3. Reparto de roles (grupo de 3)

Los grupos se forman ese día, así que esto se propone en los primeros diez
minutos. Sin roles, tres personas editan el mismo notebook y se pisan.

**Datos.** Corre la auditoría y los detectores de trampas. Es dueño del
preprocesamiento y del feature engineering de dominio. Entrega un CSV limpio y
una lista de hallazgos con evidencia.

**Modelo.** Es dueño del esquema de validación y del OOF. No toca la limpieza.
Su regla: ninguna mejora se acepta si no supera el desvío entre folds.

**Entrega.** El más importante y el que siempre falta. Es dueño de:
- el presupuesto de los 10 envíos y del `SubmitLog`
- validar cada `submit.csv` antes de subirlo
- **ir escribiendo el probatorio desde la primera hora**, no al final
- el reloj: avisa a los 30 min del cierre

Regla de oro: **quien maneja los envíos no es quien entrena.** El que entrena
siempre quiere mandar "una más".

---

## 4. Jueves 14:00 – 18:00 — mejoras

Orden estricto, de lo más barato a lo más caro. No pasás al siguiente hasta
agotar el anterior:

1. Limpieza dirigida por la auditoría (unidades, rangos imposibles, typos).
2. Feature engineering de dominio. Cinco variables pensadas valen más que
   cincuenta automáticas.
3. Ensamble de modelos distintos, con `blend_weights` (optimiza el costo, no
   el logloss).
4. `fit_prior`, verificado con `fit_prior_honest`.
5. Recién ahora hiperparámetros, y con límite de tiempo puesto.

A las 17:30, envío de cierre del día. Ese archivo queda guardado y etiquetado:
es tu red de seguridad si el viernes sale mal.

---

## 5. Viernes 08:00 – 10:00 — cerrar, no abrir

**No se empieza nada nuevo el viernes.** Dos horas y media alcanzan para
terminar lo del jueves, no para una idea fresca.

| Reloj | Qué |
|---|---|
| 08:00 | último ensamble con lo que ya funciona |
| 08:45 | envío |
| 09:15 | **reenvío del mejor archivo** — cuenta el último, no el mejor |
| 09:30 | `generar_notebook()` y completar los huecos de criterio |
| 09:50 | probatorio entregado |
| 10:00 | preparar qué contar en la puesta en común |

---

## 6. El probatorio (art. 5)

Treinta minutos, **un único archivo**, y evalúa las decisiones, no el código.
Por eso el generador produce un notebook autocontenido: los resultados van
embebidos en base64, así que corre solo, sin `work/` al lado.

```python
from ic_kit.probatorio import generar_notebook
generar_notebook(
    grupo="Grupo 8", integrantes=["Acosta", "Bareiro", "Borges"],
    desafio="<nombre>", metrica="<dirección de la métrica>",
    oof="work/oof.npy", y="work/y.npy", C="d1",
    trampas=[{"hallazgo": "...", "evidencia": "...", "decision": "..."}],
    hipotesis=["..."], decisiones=["..."], descartado=["..."])
```

Lo que el generador escribe solo: matriz de costos, tabla de auditoría,
justificación del preprocesamiento, esquema de validación, la derivación de la
regla de decisión, el desglose de dónde viene el costo con su figura, el
historial de envíos con la correlación validación–leaderboard, y la nota de
reproducibilidad.

Lo que tenés que poner vos, que es lo que se califica: las **hipótesis**, los
**hallazgos con su evidencia**, las **decisiones** y sobre todo lo
**descartado y por qué**. Ir anotando eso durante la jornada, en un `.txt`
suelto, es tarea del rol Entrega desde la hora cero.

Estilo, según la convención de la materia: español, sin emojis, sin citar las
preguntas de la consigna, sin explicar conceptos básicos, sin conclusiones en
formato pregunta-respuesta, `random_state=42`.

---

## 7. Presupuesto de los 10 envíos

| Envíos | Cuándo | Para qué |
|---|---|---|
| 1 | jue 10:15 | baseline: calibrar el gap validación↔leaderboard |
| 2–3 | jue tarde | mejoras grandes ya validadas en OOF |
| 4–6 | jue tarde | variantes que empatan en OOF; el leaderboard desempata |
| 7 | jue 17:30 | cierre del día, red de seguridad |
| 8–9 | vie mañana | ensamble final |
| 10 | vie 09:15 | **reenvío del mejor archivo** |

- Si `diff_vs()` dice que cambia menos del 1 % de las filas, no gastes el envío.
- Si la mejora en OOF es menor que el desvío entre folds, es ruido.
- El último envío debe ser el mejor archivo, aunque ya lo hayas mandado.

---

## 8. Sobre el reglamento

**Los LLMs están explícitamente permitidos** (art. 10: "Utilizar LLMs como
ChatGPT, Claude, DeepSeek, Llama o equivalentes, está más que permitido").
Usar Claude Code durante la jornada está dentro de las reglas.

**Sobre traer este kit.** El art. 7 prohíbe compartir información *entre
grupos* y recibir *asistencia externa*; el art. 10 permite material
bibliográfico, internet y "cualquier otra herramienta" que no viole las
limitaciones. Utilitarios propios preparados antes son análogos a llevar tus
propias librerías, y el probatorio los declara explícitamente. Aun así,
**mencionalo a los organizadores al inicio de la jornada**: es una línea que
conviene tener aclarada antes y no discutida después. Si te piden no usarlo, el
playbook y el orden de trabajo siguen valiendo igual — que es donde está la
mayor parte de la ventaja.

Lo que no se hace, sin importar cuánto ayude: usar el conjunto de test para
ajustar o validar, sondear el leaderboard para reconstruir las etiquetas, o
pasarle algo a otro grupo.

---

## 9. Checklist de los últimos 30 minutos

- [ ] el archivo enviado es el de mejor puntaje, no el más reciente
- [ ] `submit.csv` validado: filas = filas del test, sin NaN, sin ids de más
- [ ] la distribución predicha no es absurda contra la del train
- [ ] probatorio generado, huecos de criterio completados, ejecutado sin errores
- [ ] el notebook abre en una máquina limpia (probalo)
- [ ] quién cuenta qué en la puesta en común
