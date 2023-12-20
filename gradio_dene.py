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

   
    plt.bar(class_names, probabilities, color='blue')
    plt.xlabel('Waste Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.savefig('prediction_plot.png')

    
    output_text = f"Predicted Class: {predicted_class_name}, Confidence: {confidence:.4f}\n"
    for class_name, prob in zip(class_names, probabilities):
        output_text += f"{class_name}: {prob:.4f}\n"

    return output_text, 'prediction_plot.png'


iface = gr.Interface(fn=classify_image, inputs="image", outputs=["text", "image"],live=True)


iface.launch()
