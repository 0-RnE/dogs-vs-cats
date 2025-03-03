# DataSet descargado de: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models

# Ruta de la carpeta de entrenamiento y prueba
data_dir = "directorio/del/dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
image_size = (128, 128)


# Función para leer la imagen y extraer la etiqueta desde el nombre del archivo
def process_path(file_path):
    # Leer y decodificar la imagen
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    # Extraer el nombre del archivo y determinar la etiqueta
    # Se asume que el nombre empieza con "cat" o "dog"
    file_name = tf.strings.split(file_path, os.sep)[-1]
    # Si el nombre contiene "cat", se le asigna etiqueta 0; si "dog", etiqueta 1.
    is_cat = tf.strings.regex_full_match(file_name, "cat.*")
    label = tf.cond(is_cat, lambda: 0, lambda: 1)
    return image, label


# Crear el dataset de entrenamiento a partir de los archivos
train_files = tf.data.Dataset.list_files(os.path.join(train_dir, "*"))
train_ds = train_files.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Calcular el número total de imágenes de entrenamiento
train_count = tf.data.experimental.cardinality(train_ds).numpy()
# Dividir en entrenamiento (80%) y validación (20%)
data_train = train_ds.take(int(0.8 * train_count))
data_val = train_ds.skip(int(0.8 * train_count))

# Crear el dataset de prueba
test_files = tf.data.Dataset.list_files(os.path.join(test_dir, "*"))
test_ds = test_files.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)


# Función para convertir un tf.data.Dataset a arrays de NumPy (aplanando las imágenes para NB)
def dataset_to_numpy(dataset, max_elements=1000):
    images = []
    labels = []
    for img, lbl in dataset.take(max_elements):
        images.append(tf.reshape(img, (-1,)).numpy())
        labels.append(lbl.numpy())
    return np.array(images), np.array(labels)


print("Convirtiendo datos de entrenamiento y validación a formato numpy...")
train_images, train_labels = dataset_to_numpy(data_train)
val_images, val_labels = dataset_to_numpy(data_val)
test_images, test_labels = dataset_to_numpy(test_ds)

# Mostrar algunas imágenes de entrenamiento con sus etiquetas
print("Mostrando imágenes de entrenamiento con sus etiquetas...")
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i].reshape(128, 128, 3))
    plt.title(f"Etiqueta: {'Perro' if train_labels[i] == 1 else 'Gato'}")
    plt.axis('off')
plt.show()

# --------------------------
# Implementación de Naive Bayes
# --------------------------
print("Implementando Naive Bayes...")
gnb = GaussianNB()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

nb_auc_scores, nb_precision_scores, nb_recall_scores, nb_f1_scores = [], [], [], []
fold = 1
for train_index, test_index in skf.split(train_images, train_labels):
    print(f"Naive Bayes - Procesando fold {fold}/10...")
    X_train, X_test = train_images[train_index], train_images[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    nb_auc_scores.append(roc_auc_score(y_test, y_pred))
    nb_precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
    nb_recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
    nb_f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    fold += 1

print("Naive Bayes resultados:")
print(f"AUC: {np.mean(nb_auc_scores):.4f}")
print(f"Precisión: {np.mean(nb_precision_scores):.4f}")
print(f"Recall: {np.mean(nb_recall_scores):.4f}")
print(f"F1-score: {np.mean(nb_f1_scores):.4f}")

# --------------------------
# Implementación de CNN
# --------------------------
print("Implementando CNN...")
cnn_model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_auc_scores, cnn_precision_scores, cnn_recall_scores, cnn_f1_scores = [], [], [], []
fold = 1
for train_index, test_index in skf.split(train_images, train_labels):
    print(f"CNN - Procesando fold {fold}/10...")
    X_train = train_images[train_index].reshape(-1, 128, 128, 3)
    X_test = train_images[test_index].reshape(-1, 128, 128, 3)
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    cnn_model.fit(X_train, y_train, epochs=5, verbose=0)
    y_pred = (cnn_model.predict(X_test) > 0.5).astype(int)
    cnn_auc_scores.append(roc_auc_score(y_test, y_pred))
    cnn_precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
    cnn_recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
    cnn_f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    fold += 1

print("CNN resultados:")
print(f"AUC: {np.mean(cnn_auc_scores):.4f}")
print(f"Precisión: {np.mean(cnn_precision_scores):.4f}")
print(f"Recall: {np.mean(cnn_recall_scores):.4f}")
print(f"F1-score: {np.mean(cnn_f1_scores):.4f}")


# ----- Punto Extra: Deep CNN -----
def create_deep_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


deep_cnn_model = create_deep_cnn_model()
