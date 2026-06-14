# 🛸 Aerial Object Classification & Detection
### Bird vs Drone — Deep Learning Binary Classifier

---

## 📌 Project Overview

This project builds and deploys a deep learning model to classify aerial images as either **Bird** or **Drone**. With drones increasingly appearing near airports, restricted zones, and wildlife areas, automated detection systems are critical for safety and surveillance.

The pipeline covers data preparation, model training (Custom CNN + Transfer Learning), evaluation, and deployment via a Streamlit web app.

---

## 🎯 Objectives

- Build a binary image classifier (Bird vs Drone)
- Compare a Custom CNN baseline against Transfer Learning (MobileNetV2)
- Deploy the best model as an interactive web application

---

## 📊 Model Results

| Model | Test Accuracy |
|---|---|
| Custom CNN | 56.28% |
| Transfer Learning (Fine-tuned) | 96.74% |
| **Transfer Learning MobileNetV2** | **99.07% ✅** |

> MobileNetV2 was selected as the best model and deployed in the Streamlit app.

---

## 🗂️ Project Structure

```
aerial-object-classification/
├── app.py                        # Streamlit deployment app
├── aerial_classification.ipynb   # Full training notebook
├── model_comparison_report.csv   # Model accuracy results
├── requirements.txt              # Python dependencies
├── models/
│   ├── best_model.h5             # Saved best model (MobileNetV2)
│   └── best_model_info.txt       # Preprocessing mode info
└── README.md
```

---

## 🧠 Approach

### 1. Data Preparation
- Aerial images of birds and drones collected and organized
- Images resized to **224×224** pixels
- Split into training and test sets

### 2. Custom CNN (Baseline)
- A CNN built from scratch with Conv2D, MaxPooling, Dense layers
- Achieved ~56% accuracy — limited by small dataset size

### 3. Transfer Learning — MobileNetV2
- Pre-trained MobileNetV2 (ImageNet weights) used as a feature extractor
- Custom classification head added on top
- Fine-tuned top layers for domain-specific learning
- Achieved **99.07% test accuracy**

### 4. Deployment
- Best model served via **Streamlit**
- Users upload an aerial image → model predicts Bird or Drone with confidence score

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aerial-object-classification.git
cd aerial-object-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser and upload any aerial image.

> ⚠️ Make sure `models/best_model.h5` exists. Run the notebook first to train and save the model.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Transfer Learning | MobileNetV2 (ImageNet) |
| Data Processing | NumPy, Pandas, Pillow |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Deployment | Streamlit |

---

## 📊 Project Presentation

🔗 [View Project PPT on Google Drive](https://drive.google.com/file/d/1Gj8DIDbznU06rD27nYAFg_90BRFDrXZx/view?usp=sharing)

---
