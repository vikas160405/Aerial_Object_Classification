import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Aerial Object Classification",
    page_icon="🛸",
    layout="centered",
)

CLASS_NAMES = ["Bird", "Drone"]
IMG_SIZE = (224, 224)

MODEL_PATH = os.path.join("models", "best_model.h5")
MODEL_INFO_PATH = os.path.join("models", "best_model_info.txt")


@st.cache_resource
def load_classifier():
    model = tf.keras.models.load_model(MODEL_PATH)

    preprocess_mode = "rescale_1_255"
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            for line in f.read().splitlines():
                if line.startswith("preprocess:"):
                    preprocess_mode = line.split(":")[1].strip()
    return model, preprocess_mode


def preprocess_image(img: Image.Image, preprocess_mode: str):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if preprocess_mode == "mobilenet_v2":
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


st.title("🛸 Aerial Object Classification")
st.markdown(
    "Upload an aerial image and the model will classify it as **Bird** or **Drone**, "
    "with a confidence score."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        model, preprocess_mode = load_classifier()
        x = preprocess_image(image, preprocess_mode)
        prob = float(model.predict(x, verbose=0)[0][0])

        pred_idx = int(prob >= 0.5)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = prob if pred_idx == 1 else 1 - prob

        st.subheader("Classification Result")
        st.success(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))

    except Exception as e:
        st.error(f"Could not load classification model: {e}")
        st.info("Make sure 'models/best_model.h5' exists (run train.py first).")

else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.caption(" Aerial Object Classification & Detection ")
