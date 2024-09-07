import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from PIL import Image

# Redução da resolução de imagem para 64x64
image_size = (64, 64)

# Função para carregar imagens e labels usando Pillow
def load_images(data_dir, categories, image_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img = Image.open(img_path)
                img = img.resize(image_size)
                img_array = np.array(img)
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar imagem {img}: {e}")
    return np.array(data), np.array(labels)

# Caminho dos dados e categorias
data_dir = "/kaggle/input/alzheimer-mri-dataset/Dataset/"
categories = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]

# Carregar imagens e labels
images, labels = load_images(data_dir, categories, image_size)

# Normalização das imagens
images = images / 255.0

# One-hot encoding dos labels
labels = to_categorical(labels, num_classes=len(categories))

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Aumento de dados (data augmentation)
datagen = ImageDataGenerator(
    rotation_range=15,  # Reduzido para acelerar o treinamento
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Função para carregar o modelo
@st.cache(allow_output_mutation=True)
def load_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Resolução menor
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(64, activation='relu'),  # Camada oculta menor
        Dropout(0.3),  # Dropout menor para evitar overfitting
        Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Carregar o modelo
model = load_model()

# Treinar o modelo com aumento de dados
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),  # Batch size maior
                    validation_data=(X_test, y_test), epochs=10)  # Menos épocas

# Avaliação do modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Função para processar a imagem e fazer predição
def process_image(uploaded_file):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((64, 64))  # Redimensiona para o tamanho adequado
        img = np.array(img)
        img = np.expand_dims(img, axis=0)  # Adiciona uma dimensão para compatibilidade com o modelo
        img = img / 255.0  # Normaliza a imagem
        
        # Faz a predição usando o modelo treinado
        prediction = model.predict(img)
        confidence = np.max(prediction)
        pred_label = np.argmax(prediction, axis=1)
        
        if pred_label == 0:
            return "Mild Demented", confidence
        elif pred_label == 1:
            return "Moderate Demented", confidence
        elif pred_label == 2:
            return "Non Demented", confidence
        else:
            return "Very Mild Demented", confidence

# Interface gráfica com Streamlit
st.title("Diagnóstico de Alzheimer com RNA")

# Carregar arquivo de imagem
uploaded_file = st.file_uploader("Carregar uma imagem", type=["jpg", "png", "jpeg"])

# Histórico de diagnósticos
if 'diagnosticos' not in st.session_state:
    st.session_state['diagnosticos'] = []

# Processamento da imagem carregada
if uploaded_file is not None:
    with st.spinner("Processando a imagem..."):
        diagnosis, confidence = process_image(uploaded_file)
        st.session_state['diagnosticos'].append({'imagem': uploaded_file, 'diagnóstico': diagnosis, 'confiança': confidence})

    # Mostrar a imagem carregada e o diagnóstico
    st.image(uploaded_file, caption='Imagem carregada', use_column_width=True)
    st.write(f"Diagnóstico: {diagnosis}")
    st.write(f"Confiança do modelo: {confidence:.2f}")

# Mostrar o histórico de diagnósticos
st.write("Histórico de diagnósticos:")
for diag in st.session_state['diagnosticos']:
    st.image(diag['imagem'], width=100)
    st.write(f"Diagnóstico: {diag['diagnóstico']} - Confiança: {diag['confiança']:.2f}")
