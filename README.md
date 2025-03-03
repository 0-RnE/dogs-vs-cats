# 🐱🐶 Dogs vs Cats Classification

Este proyecto implementa un clasificador de imágenes para el dataset "Dogs vs Cats" utilizando dos enfoques:
1. **Naive Bayes (NB):** Se utiliza sobre imágenes aplanadas tras preprocesamiento.
2. **Redes Neuronales Convolucionales (CNN):** Se exploran diversas arquitecturas y se evalúan mediante validación cruzada estratificada (10-fold).

## Características del Proyecto

- **Preprocesamiento de Imágenes:**  
  - Lectura de imágenes desde archivos locales.
  - Redimensionamiento a 128x128 píxeles y normalización de los valores.
  - Extracción de etiquetas a partir del nombre de archivo (por ejemplo, "cat10" o "dog1").

- **Validación Cruzada:**  
  Se utiliza stratified 10-fold cross-validation para evaluar el desempeño de ambos modelos, usando las métricas:
  - AUC
  - Precisión
  - Recall
  - F1-score

- **Resultados del Entrenamiento:**  
  - **Naive Bayes:**  
    - AUC: 0.5700  
    - Precisión: 0.5401  
    - Recall: 0.6182  
    - F1-score: 0.5759  
  - **CNN:**  
    - AUC: 0.9180  
    - Precisión: 0.9120  
    - Recall: 0.9172  
    - F1-score: 0.9127
