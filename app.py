import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Blood Group Detection", layout="centered")

MODEL_PATH = "model.tflite"

# 🔥 Try tflite-runtime first, fallback if needed
try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
except:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Interpreter loading failed: {e}")
        st.stop()

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Auto shape
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Labels
try:
    with open("labels.txt") as f:
        class_labels = [line.strip() for line in f.readlines()]
except:
    st.error("labels.txt file missing")
    st.stop()

st.title("Blood Group Detection System")

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

            img = img.resize((width, height))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            best_idx = np.argmax(prediction)
            confidence = float(prediction[best_idx])

            st.success(f"Blood Group: {class_labels[best_idx]}")
            st.write(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
