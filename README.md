# Clasificación Supervisada (Dogs vs Cats)

## Descripción

Este proyecto tiene como objetivo aplicar técnicas de **clasificación supervisada** para analizar y clasificar imágenes del dataset "Dogs vs Cats". Se implementaron dos enfoques principales:

- **Naive Bayes:**  
  Se procesaron las imágenes aplanándolas después del preprocesamiento y se aplicó validación cruzada estratificada (10-fold). Aunque sus resultados fueron modestos (AUC ≈ 0.57), esta técnica sirvió como base para comparar métodos más avanzados.

- **Redes Neuronales Convolucionales (CNN):**  
  Se diseñó e implementó una arquitectura de deep learning que permite capturar características complejas de las imágenes. Gracias a la validación cruzada estratificada, la CNN obtuvo excelentes resultados:
  - **AUC:** ≈ 0.918  
  - **Precisión:** ≈ 0.912  
  - **Recall:** ≈ 0.917  
  - **F1-score:** ≈ 0.913  

El proyecto utiliza archivos locales para la carga de datos, extrayendo las etiquetas directamente de los nombres de los archivos (por ejemplo, "cat10" o "dog1").

## Metodología

1. **Carga y Preprocesamiento:**  
   - Se cargan las imágenes desde directorios locales y se redimensionan a 128x128 píxeles.  
   - Se normalizan los valores de píxel y se extraen las etiquetas a partir del nombre del archivo.  
   - El dataset se convierte a arrays de NumPy para facilitar el uso en el clasificador Naive Bayes.

2. **Entrenamiento y Evaluación de Modelos:**  
   - **Naive Bayes:**  
     Se aplicó validación cruzada estratificada (10-fold) sobre las imágenes aplanadas, obteniendo las métricas correspondientes.
   - **Redes Neuronales Convolucionales (CNN):**  
     Se implementó una arquitectura de CNN compuesta por capas de convolución, pooling y densas, entrenada mediante validación cruzada para evaluar su desempeño.

3. **Comparación de Resultados:**  
   Se compararon las métricas de ambos modelos para identificar el enfoque con mejor desempeño.

## Resultados

### Naive Bayes
- **AUC:** 0.57  
- **Precisión:** 0.54  
- **Recall:** 0.62  
- **F1-score:** 0.58  

### CNN
- **AUC:** 0.918  
- **Precisión:** 0.912  
- **Recall:** 0.917  
- **F1-score:** 0.913  

Estos resultados evidencian que, aunque Naive Bayes sirve como referencia, las CNN ofrecen un desempeño significativamente superior en la clasificación de imágenes.

## Visualizaciones

El repositorio incluye muestras de imágenes de entrenamiento con sus respectivas etiquetas, permitiendo observar de forma visual cómo se ha etiquetado cada imagen.

## Conclusiones

Este proyecto demuestra que aplicar técnicas de clasificación supervisada con un preprocesamiento adecuado puede lograr resultados sobresalientes en la clasificación de imágenes. La comparación entre Naive Bayes y CNN resalta la capacidad de los modelos de deep learning para capturar patrones complejos, lo que representa un paso importante hacia implementaciones empresariales y análisis de mercado.
