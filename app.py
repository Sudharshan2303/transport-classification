import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load trained model
model = load_model('military_vs_transport_model.h5')

st.title("Military vs Civilian Vehicle Classifier")
st.write("Upload an image of a vehicle for classification")

# Image upload and prediction
def predict(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    return "military" if confidence < 0.5 else "civilian", abs(confidence-0.5)*2

uploaded_file = st.file_uploader("Choose vehicle image:", 
                               type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    label, conf = predict(image)
    st.success(f"Prediction: {label.upper()} (confidence: {conf:.2%})")
