# 🛸 Aerial Object Classification & Detection

## Bird vs Drone — Deep Learning Binary Classifier

### 📌 Project Overview

This project develops and deploys a deep learning-based image classification system capable of distinguishing between aerial images of birds and drones.

With the increasing presence of drones near airports, restricted airspaces, and wildlife conservation zones, automated aerial object identification has become an important component of modern surveillance and safety systems.

The project covers the complete machine learning workflow including data preprocessing, model development, transfer learning, evaluation, and deployment through an interactive Streamlit web application.

---

## 🎯 Objectives

* Build a binary image classification model (Bird vs Drone)
* Compare a Custom CNN baseline with Transfer Learning approaches
* Evaluate model performance using standard classification metrics
* Deploy the best-performing model as a user-friendly web application

---

## 📊 Model Performance

| Model                         | Test Accuracy |
| ----------------------------- | ------------- |
| Custom CNN                    | 56.28%        |
| Fine-Tuned CNN                | 96.74%        |
| MobileNetV2 Transfer Learning | **99.07%** ✅  |

MobileNetV2 achieved the highest accuracy and was selected for deployment.

---

## 🗂️ Project Structure

aerial-object-classification/

├── app.py

├── aerial_classification.ipynb

├── model_comparison_report.csv

├── requirements.txt

├── models/

│ ├── best_model.h5

│ └── best_model_info.txt

└── README.md

---

## 🧠 Methodology

### 1. Data Preparation

* Collected aerial images containing birds and drones
* Resized images to 224 × 224 pixels
* Applied preprocessing compatible with MobileNetV2
* Split dataset into training and testing subsets

### 2. Custom CNN Baseline

A Convolutional Neural Network was developed from scratch using:

* Conv2D Layers
* MaxPooling Layers
* Dense Layers
* Dropout Regularization

Performance: 56.28% Accuracy

---

### 3. Transfer Learning using MobileNetV2

To improve performance, MobileNetV2 pre-trained on ImageNet was utilized.

Key Steps:

* Loaded ImageNet weights
* Added custom classification layers
* Fine-tuned upper layers
* Optimized using Adam optimizer

Performance: 99.07% Accuracy

---

### 4. Deployment

The best-performing model was deployed using Streamlit.

Features:

* Upload aerial images
* Real-time Bird/Drone classification
* Confidence score visualization
* Lightweight and easy-to-use interface

---

## 🚀 Installation

### Clone Repository

git clone https://github.com/YOUR_USERNAME/aerial-object-classification.git

cd aerial-object-classification

### Install Dependencies

pip install -r requirements.txt

### Run Application

streamlit run app.py

Open:

http://localhost:8501

---

## 🛠️ Technology Stack

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Deep Learning        | TensorFlow, Keras   |
| Transfer Learning    | MobileNetV2         |
| Data Processing      | NumPy, Pandas       |
| Visualization        | Matplotlib, Seaborn |
| Evaluation           | Scikit-learn        |
| Deployment           | Streamlit           |

---

## 📈 Future Enhancements

* Multi-class aerial object classification
* Real-time video stream processing
* YOLOv8 object detection integration
* Edge deployment on drones and surveillance devices
* Cloud deployment using AWS or Azure

---

