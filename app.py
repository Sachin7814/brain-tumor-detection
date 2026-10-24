
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils import normalize
import os

# -------------------- CONFIG --------------------
MODEL_PATH = "BrainTumor10EpochsCategorical.h5"
INPUT_SIZE = 64

st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------- WEBSITE DESIGN --------------------
st.title("🧠 NeuroScan AI")
st.subheader("Brain Tumor Detection System")
st.write(
    "Upload a brain MRI image to get an AI model prediction."
)

st.info(
    "This tool is for educational purposes only. "
    "It is not a substitute for professional medical diagnosis."
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_brain_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_brain_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- PREDICTION --------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Tumor", type="primary"):
            with st.spinner("Analyzing image..."):
                # Resize to match model input
                resized_image = image.resize(
                    (INPUT_SIZE, INPUT_SIZE)
                )

                image_array = np.array(resized_image)
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                # Match the preprocessing used in your original code
                image_array = normalize(
                    image_array, axis=1
                )

                prediction = model.predict(
                    image_array, verbose=0
                )

                probabilities = np.asarray(
                    prediction[0]
                ).flatten()

                if len(probabilities) != 2:
                    st.error(
                        "This app expects a two-class model. "
                        "Check the model output shape."
                    )
                    st.stop()

                no_tumor = float(probabilities[0])
                tumor = float(probabilities[1])

                st.subheader("Prediction Result")

                if tumor > no_tumor:
                    st.error("Tumor Detected")
                else:
                    st.success("No Tumor Detected")

                st.write(
                    f"Tumor probability: {tumor * 100:.2f}%"
                )
                st.write(
                    f"No tumor probability: "
                    f"{no_tumor * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, tumor))
                )

                st.caption(
                    "Probabilities are model outputs, "
                    "not confirmed diagnostic confidence."
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "NeuroScan AI | Educational project | "
    "Consult a qualified medical professional for diagnosis."
)
