import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# 1. Download model from Google Drive if not present
# -----------------------------
FILE_ID = "YOUR_FILE_ID_HERE"  # 👈 paste your Drive File ID here
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "stroke_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(URL, MODEL_PATH, quiet=False)

# -----------------------------
# 2. Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🧠 Brain Stroke Detection App")
st.write("Upload a CT/MRI image to check for stroke")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (resize to 224x224 or your model’s input size)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Show result
    st.subheader("🔍 Prediction Result:")
    if prediction > 0.5:
        st.error(f"⚠ Stroke Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Stroke Detected (Confidence: {1-prediction:.2f})")