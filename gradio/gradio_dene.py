import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
model = load_model("B:\Dosya\Kodlar\Sem-Images-Classification\efficent_net224B0.h5")

# Define the classes
waste_labels = {0: 'Fibres', 1: 'Nanowires', 2: 'Particles', 3: 'Powder'}

# Define the Gradio interface
def classify_image(pil_image):
    # Convert PIL.Image to Numpy array
    img = image.img_to_array(pil_image)
    
    # Resize to the model's expected input size
    img = tf.image.resize(img, (224, 224))
    
    # Expand dimensions to create a batch size of 1
    img = np.expand_dims(img, axis=0)
    
    # Preprocess the input for the EfficientNet model
    img = preprocess_input(img)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(prediction)
    predicted_class_name = waste_labels[predicted_class]
    confidence = prediction[0, np.argmax(prediction)]

    # Get class names and probabilities for all classes
    class_names = list(waste_labels.values())
    probabilities = prediction[0]

    # Create Plotly bar chart
    fig = go.Figure(data=[go.Bar(x=class_names, y=probabilities)])
    fig.update_layout(title='Prediction Probabilities', xaxis_title='Waste Classes', yaxis_title='Probability')

    # Save Plotly chart as an HTML file
    fig.write_html('prediction_plot.html')

    # Prepare output text with all class probabilities
    output_text = f"Predicted Class: {predicted_class_name}, Confidence: {confidence:.4f}\n"
    for class_name, prob in zip(class_names, probabilities):
        output_text += f"{class_name}: {prob:.4f}\n"

    return output_text, 'prediction_plot.html'

# Create the Gradio interface
iface = gr.Interface(fn=classify_image, inputs="image", outputs=["text", "html"])

# Launch the Gradio interface
iface.launch()
