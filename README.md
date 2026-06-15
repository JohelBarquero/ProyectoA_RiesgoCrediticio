# Proyecto A: Sistema de Predicción de Riesgo Crediticio
### Equipo
Johel Josue Barquero Carvajal

Jefferson Rafael Granados Rodrigues
##  Descripción del Proyecto
Desarrollar un sistema inteligente que prediga el riesgo crediticio de clientes bancarios para apoyar decisiones de aprobación de préstamos. El sistema debe clasificar clientes en categorías de riesgo y proporcionar predicciones binarias de aprobación/rechazo. 
## Dataset
Fuente: German Credit Data (UCI Machine Learning Repository)
URL: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Registros: 1,000 clientes
Variables: 20 (demográficas, financieras, historial crediticio)

## Ejecutar Notebooks
jupyter notebook notebooks/
Ejecutar en orden:

EDA.ipynb - Análisis exploratorio
Preprocesamiento.ipynb - Limpieza y preparación
ANN_Modelo1.ipynb - Modelo binario
ANN_Modelo2.ipynb - Modelo multiclase
Comparacion.ipynb - Evaluación

## Estructura del Proyecto

## Modelos Implementados
Modelo 1: Clasificación Binaria
Objetivo: Predecir aprobación de crédito (Bueno/Malo)
Modelo 2: Clasificación Multiclase
Objetivo: Clasificar nivel de riesgo (Bajo/Medio/Alto/Crítico)

## Librerias utilizadas
Python 3.x
TensorFlow/Keras: Redes neuronales
Pandas/NumPy: Manipulación de datos
Matplotlib/Seaborn: Visualización
Scikit-learn: Preprocesamiento y métricas
Streamlit: Frontend web

