import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import gdown
import os

# IMPORTANT: use tensorflow instead of tflite_runtime
from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(page_title="Blood Group Detection", layout="centered")

MODEL_PATH = "model.tflite"
MODEL_URL = "PASTE_YOUR_TFLITE_LINK_HERE"   # must be direct download link

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully")
    except:
        st.error("Model download failed. Check your link.")
        st.stop()

# Load model safely
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

            # Preprocess
            img = img.resize((128,128))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])

            st.success(f"Blood Group: {class_labels[best_idx]}")
            st.write(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
