# 🩸 Blood Group Detection using Fingerprint

This project is a Machine Learning-based web application that predicts a person's blood group using fingerprint images. The system uses a trained Convolutional Neural Network (CNN) model and provides predictions through an interactive web interface built with Streamlit.

---

##  Live Demo

 Try the app here:  
https://blood-group-detection-fingerprint-ioxnjjb5dgfmf8jjjgbm3i.streamlit.app/

---

##  Objective

The main objective of this project is to explore whether fingerprint patterns can be used to predict blood groups using machine learning techniques. It demonstrates how deep learning models can be integrated into a real-time web application.

---

##  Methodology / Implementation

### 1. Data Collection  
Fingerprint image dataset was used where each image is labeled with its corresponding blood group.

### 2. Data Preprocessing  
- Images resized to 128 × 128 pixels  
- Converted to RGB format  
- Pixel values normalized (0–1 range)  
- Dataset split into training and testing  

### 3. Model Development  
A Convolutional Neural Network (CNN) was built using TensorFlow/Keras with:
- Convolution layers for feature extraction  
- Pooling layers for dimensionality reduction  
- Fully connected layers for classification  

### 4. Model Training  
- Loss function: Categorical Crossentropy  
- Optimizer: Adam  
- Metric: Accuracy  

### 5. Model Conversion  
- Model saved as: blood_group_model.h5  
- Converted to: model.onnx using tf2onnx  
- ONNX used because it supports deployment on modern environments  

### 6. Deployment  
- Web app built using Streamlit  
- Model loaded using ONNX Runtime  
- User uploads fingerprint image  
- Model predicts blood group with confidence score  

---

##  Technologies Used

- Python  
- TensorFlow / Keras  
- ONNX Runtime  
- Streamlit  
- NumPy  
- PIL (Python Imaging Library)  

---

##  Project Structure

blood-group-detection-fingerprint/  
│  
├── app.py  
├── model.onnx  
├── labels.txt  
├── requirements.txt  
├── BloodGroup_Fingerprint_AI.ipynb  
├── test_images/  
└── README.md  

---

##  Notebook

The complete model training process is available in:  
BloodGroup_Fingerprint_AI.ipynb  

It includes:
- Data preprocessing  
- Model building  
- Training and evaluation  
- Model saving  

---

##  Installation

Clone the repository:  
git clone https://github.com/Biswasrita/blood-group-detection-fingerprint.git  
cd blood-group-detection-fingerprint  

Install dependencies:  
pip install -r requirements.txt  

Run the application:  
streamlit run app.py  

---

##  Usage

1. Open the web app  
2. Enter patient details  
3. Upload fingerprint image  
4. Click Predict  
5. View predicted blood group and confidence  

---

##  Test Images

Sample fingerprint images are available in the test_images/ folder.

---

##  Author

Biswasrita Hazra  
