import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.applications.resnet50 import preprocess_input
import plotly.express as px

# model yükle
model = tf.keras.models.load_model("efficent_net224B0.h5")

# Etiketler
waste_labels = {0: 'Fibres', 1: 'Nanowires', 2: 'Particles', 3: 'Powder'}

# uygulama yükle
st.title("SEM Atık Tahmin Uygulaması")
st.write("Lütfen bir SEM görüntüsü yükleyin.")

# giriş yap
uploaded_image = st.file_uploader("SEM Görüntüsünü Yükleyin", type=["jpg", "png", "jpeg"])

# resim işleme

if uploaded_image is not None:
    # Görüntüyü modelin girdi boyutuna yeniden boyutlandırın
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    

    # tahmin
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Sonuç
    st.image(uploaded_image, caption='Yüklenen Görüntü', use_column_width=True)
    st.write(f"Tahmin Edilen Sınıf: {waste_labels[predicted_class]}")

    # görselleştirme
    st.write("Tahmin İhtimalleri:")
    labels = list(waste_labels.values())
    probabilities = prediction[0] * 100  # İhtimalleri yüzde olarak hesapla

    # Çubuk grafik
    fig_bar = px.bar(x=labels, y=probabilities, labels={'x': 'Sınıf', 'y': 'Yüzde (%)'},
                     title="Tahmin İhtimalleri (Çubuk Grafik)")
    st.plotly_chart(fig_bar)

    # Pasta grafiği
    fig_pie = px.pie(values=probabilities, names=labels, title="Tahmin İhtimalleri (Pasta Grafiği)")
    st.plotly_chart(fig_pie)