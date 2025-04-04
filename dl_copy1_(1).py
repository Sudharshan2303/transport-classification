import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

@st.cache_resource
def load_my_model():
    return load_model("military_vs_transport_model.h5")

model = load_my_model()

st.title("🚀 Military vs. Transport Vehicle Classifier")
st.write("Upload an image of a vehicle to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_image(image):
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
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

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction, confidence = predict_image(image)
    
    st.success(f"**Prediction:** {prediction}")
    st.metric("Confidence", f"{confidence:.2%}")
    
    # Show confidence visualization
    conf_percent = confidence if "Other" in prediction else 1-confidence
    st.progress(float(conf_percent))
