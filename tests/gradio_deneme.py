import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


model = load_model("efficent_net224B0.h5")


waste_labels = {0: 'Fibres', 1: 'Nanowires', 2: 'Particles', 3: 'Powder'}

def classify_image(pil_image):
    
    img = image.img_to_array(pil_image)
    
 
    img = tf.image.resize(img, (224, 224))
    
    
    img = np.expand_dims(img, axis=0)
    
    
    img = preprocess_input(img)
    

    prediction = model.predict(img)
    
    
    predicted_class = np.argmax(prediction)
    predicted_class_name = waste_labels[predicted_class]
    confidence = prediction[0, np.argmax(prediction)]

    
    class_names = list(waste_labels.values())
    probabilities = prediction[0]

    return class_names, probabilities

print(classify_image("B:\Dosya\Kodlar\Sem-Images-Classification\data\data100\Fibres\L9_0a95f6c416d2bd7c7675d27602ca1b4b.jpg"))