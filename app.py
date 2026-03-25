import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import gdown
import os
import tflite_runtime.interpreter as tflite

# Page config
st.set_page_config(page_title="Blood Group Detection", layout="centered")

# ================= MODEL =================
MODEL_PATH = "model.tflite"

# 🔥 IMPORTANT: PUT YOUR REAL LINK HERE
MODEL_URL = "https://drive.google.com/uc?id=PASTE_YOUR_ID_HERE"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================= LABELS =================
with open("labels.txt") as f:
    class_labels = [line.strip() for line in f.readlines()]

# ================= UI =================
st.markdown("<h1 style='text-align: center;'>🧬 Blood Group Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Patient Name")

with col2:
    age = st.number_input("🎂 Age", min_value=1, max_value=120)

gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

st.markdown("---")

uploaded_file = st.file_uploader("🖐 Upload Fingerprint Image", type=["jpg", "png", "bmp"])

# ================= BUTTON =================
if st.button("🔍 Predict Blood Group"):

    if uploaded_file is None or name == "":
        st.warning("Please fill all details and upload image")

    else:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image")

        img_array = np.array(img)

        # ===== QUALITY CHECK =====
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        with col2:
            st.subheader("🔍 Image Quality")
            st.write(f"Brightness: {brightness:.2f}")
            st.write(f"Blur Score: {blur:.2f}")

            if brightness < 50:
                st.warning("⚠️ Image too dark")
            if blur < 100:
                st.warning("⚠️ Image blurry")

        # ===== PREPROCESS =====
        img = img.resize((128, 128))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ===== PREDICTION =====
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        best_idx = np.argmax(prediction)
        confidence = prediction[best_idx]

        st.markdown("---")

        # ===== RESULT =====
        st.markdown("### 🧾 Result")

        st.success(f"🩸 Blood Group: {class_labels[best_idx]}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        st.write(f"👤 Name: {name}")
        st.write(f"🎂 Age: {age}")
        st.write(f"⚧ Gender: {gender}")

        if confidence < 0.6:
            st.warning("⚠️ Model is not confident")

        # ===== REPORT =====
        report = f"""Patient Name: {name}
Age: {age}
Gender: {gender}

Predicted Blood Group: {class_labels[best_idx]}
Confidence: {confidence*100:.2f}%
"""

        st.download_button(
            label="📥 Download TXT Report",
            data=report,
            file_name="blood_group_report.txt",
            mime="text/plain"
        )

        # ===== PDF =====
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in report.split("\n"):
            pdf.cell(200, 10, txt=line, ln=True)

        pdf_output = pdf.output(dest='S').encode('latin1')

        st.download_button(
            label="📑 Download PDF Report",
            data=pdf_output,
            file_name="blood_group_report.pdf",
            mime="application/pdf"
        )

