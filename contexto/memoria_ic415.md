# Memoria de contexto — IC415 y forma de trabajo del equipo

Destilado de las dos memorias de proyecto que Alex mantiene para la materia
(`memoria_proyecto_practica_IC415.md` y `MEMORIA_PROYECTO_IC415.md`). Recoge
sólo lo operativamente relevante para la hackathon; los detalles de trabajos
prácticos pasados quedan afuera.

## Quiénes

- **Alex Acosta**, estudiante de la UNaM, Facultad de Ingeniería (Oberá).
- Materia: **Inteligencia Computacional (IC415, 2026)**. Docentes: Krujoski,
  Skrauba, Pryszczuk — los mismos que organizan la hackathon y los desafíos.
- Trabajo grupal en la materia como Grupo 8. Para la hackathon el equipo es
  **Acosta, Borges, Pelinski**.

## Cómo se califica (clave)

**Se evalúan los análisis, procedimientos, justificaciones y conclusiones —
no el código en sí.** Un modelo bueno con un informe pobre pierde nota. Esto
aplica directo al probatorio de la hackathon.

## Convenciones de notebook (estrictas en la materia)

- **Español**, **sin emojis** en ninguna salida.
- **No citar las preguntas de la consigna**: las respuestas se embeben en el
  flujo analítico como afirmaciones demostradas, nunca como pares
  pregunta-respuesta.
- No explicar "por qué hacemos X" de conceptos básicos.
- Markdown conciso en el notebook; la interpretación detallada va al informe.
- **`random_state = 42`** en todos lados (el kit ya usa 42).
- Figuras autoguardadas con nomenclatura sistemática + celda de descarga ZIP
  (el módulo `eda.py` del kit ya hace esto: `fig01_...png` y `.zip()`).
- Autocontenido y ejecutable secuencialmente, con resultados idénticos.

## Recorrido metodológico esperado (no se saltea)

EDA → armado de pipeline → análisis de varios modelos → elección de métricas →
lo pertinente. Además de programar, se espera **explicar según la teoría**,
plantear hipótesis y justificar por qué la solución elegida es la mejor. El
foco está en la calidad del recorrido, no en la extensión.

## Principios propios que funcionan

- **Validar antes de escribir:** smoke-test empírico de la lógica central
  (fitness, métricas, comportamiento de clases) *antes* de generar el notebook.
- **Reporte honesto por sobre fórmulas prolijas:** si el experimento contradice
  la hipótesis, se corrige el texto, no los datos.
- **Transparencia de dos métodos:** si un solo enfoque no cumple todos los
  requisitos, se declaran ambos con su justificación en vez de ocultar el
  trade-off.

## Entorno y stack de la materia

- **Google Colab (free tier):** ~12 GB RAM, ~100 GB disco. Se diseña el
  notebook teniéndolo en cuenta (batch sizes, checkpoints) sin aclararlo en el
  texto.
- **Deep learning: Keras/TensorFlow 3.x** (por eso `BASE_vision.ipynb` es Keras
  y no torch). ML clásico: scikit-learn. Evolutivo: DEAP. NLP:
  sentence-transformers. Notebooks: nbformat/nbclient. Reportes: reportlab.
- Datos desde Drive vía `gdown`/`wget`, montaje adaptativo de Drive.

## Antecedente relevante

Alex ya participó del **1er Desafío 2026** (triaje hospitalario, métrica de
costo personalizada) llegando a score ~0,77, y del **2do** (biodiversidad de la
Selva Atlántica, costo por reino). Esos son los desafíos codificados como
`d1`/`d2` en `ic_kit/costs.py`. Ver `desafios_previos.md`.
