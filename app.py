import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

# Define the class labels (these should match the folder names in the dataset)
class_labels = ['happy', 'anger', 'pain', 'disgust', 'fear', 'sad']

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_emotion(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Predict the emotion
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

# Streamlit App Interface
st.title("Emotion Recognition")

st.write("""
Upload an image or take a picture using your camera to predict the emotion displayed in the image.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Capture image from camera
camera_image = st.camera_input("Take a picture")

if uploaded_file or camera_image:
    if uploaded_file:
        img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
    elif camera_image:
        img_path = os.path.join(UPLOAD_FOLDER, "camera_image.jpg")
        with open(img_path, 'wb') as f:
            f.write(camera_image.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    
    # Show progress bar
    progress_bar = st.progress(0)
    
    for percent_complete in range(1, 101):
        progress_bar.progress(percent_complete)
        # Simulate processing time
        time.sleep(0.01)

    # Predict emotion
    predicted_emotion = predict_emotion(img_path)
    st.success(f'The predicted emotion is: {predicted_emotion}')
