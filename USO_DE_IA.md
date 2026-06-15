# Documentación del Uso de Inteligencia Artificial (IA) en el Proyecto

---

## 1. Herramientas de IA Utilizadas y Tareas Específicas

Para el desarrollo de este proyecto se utilizaron los modelos de lenguaje **Gemini (Google)** y **Claude (Anthropic)**:

* **Adaptación de Código:** Ajustar la lógica de la red neuronal artificial (arquitectura secuencial en TensorFlow/Keras).
* **Depuración de Errores:** Resolución de problemas de dimensionalidad en las funciones de evaluación.
* **Estructuración y Traducción del EDA:** Optimización del orden lógico de las celdas del Notebook de Análisis Exploratorio de Datos (EDA) y traducción de las variables del dataset original (*German Credit Data*) del inglés al español.

---

## 2. Ejemplos de Prompts Representativos Utilizados

A continuación, se presentan los prompts exactos e instrucciones compartidas con la IA para guiar el desarrollo de los entregables:

> **Prompt de Arquitectura:**
> *"Adaptame mi codigo, ocupo que me de predicciones [se adjuntó el bloque de código base de referencia y el script propio inicial]"*
> * **Objetivo:** Lograr que la estructura secuencial de la ANN mapeara correctamente las 4 categorías académicas de salida en el modelo multiclase y asegurar que las funciones de decodificación (`np.argmax`) operaran correctamente.

> **Prompt de Depuración (Debugging):**
> *"ValueError: Number of classes, 2, does not match size of target_names, 4. Try specifying the labels parameter [se adjuntó el traceback completo de Python]"*
> * **Objetivo:** Identificar por qué `scikit-learn` no podía imprimir el reporte de clasificación analítico (`classification_report`) y corregirlo forzando los índices mediante el parámetro explícito `labels=[0, 1, 2, 3]`.

> **Prompt de Limpieza Estadística:**
> *"Ayudame a darle un mejor orden lógico a los gráficos y pasame los nombres de las variables de inglés a español."*
> * **Objetivo:** Garantizar que la visualización de las distribuciones fuera clara y ordenada

> **Prompt de Modularización:**
> *"Creame la estructura basica del preprocesamiento para mi proyecto / creame la estructura basica del data_prep.py [se adjuntó la descripción del problema del proyecto]"*

---

## 3. Modificaciones Críticas al Código y Análisis Generado

**Control de Épocas y Parada Temprana:** Los prompts sugerían configuraciones estáticas de entrenamiento cortas (v.g., 20 épocas). Se decidió expandir el rango máximo a **200 épocas**, y que se detuviera solo con **Early Stopping**.

---

## 4. Reflexión sobre el Aprendizaje y el Apoyo de la IA

El uso de la Inteligencia Artificial en este proyecto no actuó como un sustituto del pensamiento crítico ni del esfuerzo académico, sino como un **tutor personalizado 24/7**. 

