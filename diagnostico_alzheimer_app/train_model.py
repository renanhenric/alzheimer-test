import os
import numpy as np
import cv2 as cv
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurações
image_size = (64, 64)
data_dir = "./dataset/"  # ajuste conforme sua pasta local
dataset_classes = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]

# Função para carregar imagens
def load_images(data_dir, categories, image_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_COLOR)
                img_array = cv.resize(img_array, image_size)
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar imagem {img}: {e}")
    return np.array(data), np.array(labels)

# Carregar dados
images, labels = load_images(data_dir, dataset_classes, image_size)
images = images / 255.0
labels = to_categorical(labels, num_classes=len(dataset_classes))

# Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(dataset_classes), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          validation_data=(X_test, y_test),
          epochs=10)

# Avaliar
loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia final: {acc:.4f}")

# Salvar o modelo
model.save("model.h5")
