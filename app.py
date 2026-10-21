import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload a traffic sign image to predict its category.")

MODEL_PATH = Path(__file__).parent / "model" / "traffic_sign_mobilenetv2.keras"
CLASS_NAMES = ["Cautionary", "Informatory", "Mandatory"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Check that the model folder contains traffic_sign_mobilenetv2.keras.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)

        resized = image.resize((160, 160))
        image_array = np.array(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Predict Traffic Sign", type="primary"):
            with st.spinner("Analyzing image..."):
                predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.success(f"Predicted Category: {predicted_class}")
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.subheader("Class-wise Scores")
            for name, score in zip(CLASS_NAMES, predictions):
                st.write(f"{name}: {score * 100:.2f}%")
                st.progress(float(np.clip(score, 0, 1)))

    except Exception as e:
        st.error(f"Unable to process this image: {e}")