import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_cropper import st_cropper


st.set_page_config(
    page_title="Aerial Object Classification",
    page_icon="🛸",
    layout="centered",
)

CLASS_NAMES = ["Bird", "Drone"]
IMG_SIZE = (224, 224)

MODEL_WEIGHTS_PATH = os.path.join("models", "best_model_v2.weights.h5")
MODEL_INFO_PATH = os.path.join("models", "best_model_info.txt")


def build_model():
    """Same architecture used during training: MobileNetV2 base (frozen) +
    a small classification head. Building the architecture in code and
    loading only the weights avoids Keras-version-specific config
    deserialization issues that can occur with full model.save() files."""
    from tensorflow.keras import layers, models as keras_models
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base_model.trainable = False
    model = keras_models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.build((None, 224, 224, 3))
    return model


@st.cache_resource
def load_classifier():
    model = build_model()
    model.load_weights(MODEL_WEIGHTS_PATH)

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
    "Upload an aerial image, then drag the box below to select the bird or "
    "drone you want to classify. This helps a lot when the subject is small "
    "or far away in the original photo."
)
st.caption(
    "⚠️ Known limitation: this model only recognizes Bird or Drone — it has no "
    "'neither' option. Selecting a region that isn't actually a bird or drone "
    "(a boat, buoy, building, etc.) will still be forced into one of the two "
    "classes, since the model has never seen negative/background-only examples."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("**Drag the box to select the bird/drone, then resize as needed:**")
    cropped_image = st_cropper(
        image,
        realtime_update=True,
        box_color="#00FF00",
        aspect_ratio=None,
        return_type="image",
    )

    st.markdown("**Selected region (this is what gets classified):**")
    st.image(cropped_image, width=300)

    try:
        model, preprocess_mode = load_classifier()

        x = preprocess_image(cropped_image, preprocess_mode)
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
        st.info("Make sure 'models/best_model_v2.weights.h5' exists.")

else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.caption(" Aerial Object Classification & Detection ")