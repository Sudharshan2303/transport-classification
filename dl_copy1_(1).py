# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    "C:/Users/HP/Downloads/archive/dataset/train",  # Replace with your training data path
    target_size=(224, 224),  # Resize images to match MobileNet input
    batch_size=32,
    class_mode='binary'  # Binary classification
)

test_set = test_datagen.flow_from_directory(
    "C:/Users/HP/Downloads/archive/dataset/test",  # Replace with your test data path
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Display Class Indices
print("Class Indices:", train_set.class_indices)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Add a global average pooling layer
    Dense(512, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_set,
    validation_data=test_set,
    epochs=10,  # Train for 10 epochs (adjust as needed)
    steps_per_epoch=train_set.samples // train_set.batch_size,
    validation_steps=test_set.samples // test_set.batch_size
)

model.evaluate(test_set)

# Evaluate the Model
loss, accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("military_vs_transport_model.h5")

def predict_image(image_path, model):
    # Load and preprocess the image
    test_image = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    test_image = img_to_array(test_image) / 255.0  # Normalize pixel values
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Make prediction
    result = model.predict(test_image)
    confidence = result[0][0]
    if confidence < 0.5:
        prediction = "military"
    else:
        prediction = "other transport"
    return prediction, confidence

image_path = "C:/Users/HP/Downloads/military.webp"  # Replace with your image path
prediction, confidence = predict_image(image_path, model)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

image_path = "C:/Users/HP/Downloads/v.webp"  # Replace with your image path
prediction, confidence = predict_image(image_path, model)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("military_vs_transport_model.h5")

# Set the title and description
st.title("🚀 Military vs. Transport Vehicle Classifier")
st.write("Upload an image, and the AI model will classify it as either **Military** or **Other Transport**.")

# File uploader for user image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    result = model.predict(image)
    confidence = result[0][0]

    if confidence < 0.5:
        return "🚔 Military Vehicle", confidence
    else:
        return "🚗 Other Transport", confidence

# Display uploaded image and predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    prediction, confidence = predict_image(image)

    # Show results
    st.markdown(f"### **Prediction: {prediction}**")
    st.markdown(f"### **Confidence: {confidence:.2f}**")
