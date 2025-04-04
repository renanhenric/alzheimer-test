# 🧠 Alzheimer Diagnosis App

Aplicação em **Streamlit** que utiliza **Redes Neurais Convolucionais (CNNs)** para diagnosticar o estágio do Alzheimer com base em imagens de **ressonância magnética (MRI)**.

## 📌 Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Streamlit
- PIL (Pillow)
- Scikit-learn


## 📌 Funcionalidades

- Classificação automática de imagens em quatro categorias:
  - 🟢 Non Demented
  - 🟡 Very Mild Demented
  - 🟠 Mild Demented
  - 🔴 Moderate Demented
- Interface intuitiva para upload de imagem e exibição do diagnóstico.
- Histórico dos diagnósticos anteriores dentro da interface.

## 🚀 Como rodar localmente

1. Clone o repositório:
```bash
git clone https://github.com/seunome/alzheimer-diagnosis-app.git
cd alzheimer-diagnosis-app
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Treine o modelo
:
```bash
python train_model.py
```

5. Execute a aplicação
:
```bash
streamlit run app.py
```

🧠 Dataset
O modelo foi treinado usando o dataset disponível no Kaggle:
[Alzheimer MRI Dataset](https://www.kaggle.com/datasets)

Certifique-se de baixar os dados e ajustar o caminho em app.py para apontar para a pasta correta.

---
```bash
## ✅ `requirements.txt`

```txt
streamlit==1.32.2
tensorflow==2.15.0
opencv-python==4.9.0.80
pandas==2.2.1
numpy==1.26.4
Pillow==10.2.0
scikit-learn==1.4.1
