# Proyecto de Predicción de Diabetes - Machine Learning

Este proyecto aplica modelos de aprendizaje supervisado para predecir la presencia de diabetes en pacientes utilizando el dataset de Pima Indians.

## 📊 Comparativa de Modelos

Durante el desarrollo se probaron tres arquitecturas diferentes para evaluar cuál generalizaba mejor con los datos:

| Modelo | Accuracy (Prueba) | Observaciones |
| :--- | :--- | :--- |
| **Decision Tree** | ~72.00% | Modelo base, propenso al sobreajuste. |
| **Random Forest** | **79.22%** | **Mejor modelo.** Muy estable y robusto. |
| **Gradient Boosting** | 74.68% | Implementado con Sklearn debido a restricciones técnicas del entorno. |

> **Nota técnica:** Se intentó la implementación de XGBoost, pero debido a la falta de dependencias de sistema (`libgomp1`) en el entorno de Codespaces, se optó por el `GradientBoostingClassifier` de Scikit-Learn, obteniendo resultados consistentes.

## 🛠️ Tecnologías Usadas
- **Python 3.12**
- **Pandas & Numpy** (Limpieza de datos)
- **Scikit-Learn** (Modelado y GridSearchCV)
- **Pickle** (Para exportar el modelo final)

## 📁 Estructura del Proyecto
- `src/app.py`: Script principal con el flujo completo.
- `src/models/`: Contiene el archivo `.sav` del mejor modelo (Random Forest).
- `Exploring_Diabetes.ipynb`: Notebook con el análisis exploratorio y pruebas.

## 📈 Conclusión
El modelo de **Random Forest** fue el ganador para este dataset específico, logrando el equilibrio ideal entre precisión y capacidad de generalización.