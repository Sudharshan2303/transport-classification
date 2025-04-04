import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Military vs. Transport Classifier",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def load_my_model():
    return load_model("military_vs_transport_model.h5")

# Load model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# App UI
st.title("🚀 Military vs. Transport Vehicle Classifier")
st.write("Upload an image of a vehicle to classify it as Military or Other Transport")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a vehicle"
)

# Prediction function
def predict_image(image):
    try:
        image = image.resize((224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        result = model.predict(image)
        confidence = result[0][0]
        
        if confidence < 0.5:
            return "🚔 Military Vehicle", confidence
        else:
            return "🚗 Other Transport", confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Display results
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            prediction, confidence = predict_image(image)
        
        if "Error" in prediction:
            st.error(prediction)
        else:
            st.success(f"**Prediction:** {prediction}")
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Show confidence bar
            conf_percent = confidence if "Other" in prediction else 1-confidence
            st.progress(conf_percent)
