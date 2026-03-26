import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Blood Group Detection", layout="centered")

MODEL_PATH = "model.onnx"

# Load ONNX model safely
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Load labels
try:
    with open("labels.txt") as f:
        class_labels = [line.strip() for line in f.readlines()]
except:
    st.error("labels.txt file missing")
    st.stop()

st.title("Blood Group Detection System")

# User inputs
name = st.text_input("Patient Name")
age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg","png","bmp"])

if st.button("Predict"):

    if uploaded_file is None or name.strip() == "":
        st.warning("Please fill all details and upload image")

    else:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Resize (change if your model uses different size)
            img = img.resize((128, 128))

            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = session.run(None, {input_name: img_array})[0][0]

            best_idx = int(np.argmax(prediction))
            confidence = float(prediction[best_idx])

            st.success(f"Blood Group: {class_labels[best_idx]}")
            st.write(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
