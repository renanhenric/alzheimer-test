import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()
categories = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Função para processar imagem
def process_image(uploaded_file):
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        pred_label = np.argmax(prediction)

        return categories[pred_label], confidence
    return None, None

# Interface Streamlit
st.set_page_config(page_title="Diagnóstico de Alzheimer", layout="centered")
st.title("Diagnóstico de Alzheimer com CNN")

uploaded_file = st.file_uploader("Envie uma imagem de ressonância", type=["jpg", "jpeg", "png"])

if 'diagnosticos' not in st.session_state:
    st.session_state['diagnosticos'] = []

if uploaded_file is not None:
    with st.spinner("Processando imagem..."):
        diagnosis, confidence = process_image(uploaded_file)
        if diagnosis:
            st.session_state['diagnosticos'].append({
                'imagem': uploaded_file,
                'diagnostico': diagnosis,
                'confianca': confidence
            })

    st.image(uploaded_file, caption='Imagem enviada', use_column_width=True)
    st.success(f"Diagnóstico: {diagnosis} (Confiança: {confidence:.2f})")

# Histórico
st.subheader("Histórico de diagnósticos")
for diag in st.session_state['diagnosticos']:
    st.image(diag['imagem'], width=100)
    st.write(f"{diag['diagnostico']} - Confiança: {diag['confianca']:.2f}")
