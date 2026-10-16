"""Streamlit app: upload a weather photo and get the predicted condition.

    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st

from preprocess import decode_image

MODEL_PATH = os.path.join("models", "weather_cnn.keras")
CLASSES_PATH = os.path.join("models", "class_names.json")


@st.cache_resource
def load_model():
    # Cached: loading the model takes seconds, and Streamlit reruns this whole
    # script on every interaction. Imported here so the page can still say
    # "model missing" quickly without TensorFlow's import cost.
    from tensorflow.keras.models import load_model as keras_load_model

    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return keras_load_model(MODEL_PATH), class_names


st.set_page_config(page_title="Climate Vision", page_icon="🌦️")
st.title("Climate Vision")
st.caption("CNN-based weather condition detection: Cloudy, Rain, Shine or Sunrise.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    st.error(
        "No trained model found. Run `python train.py`, or download "
        "`weather_cnn.keras` from the GitHub release into `models/`."
    )
    st.stop()

model, class_names = load_model()

uploaded = st.file_uploader("Upload a weather photo", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

image = decode_image(uploaded.getvalue())
if image is None:
    st.error("That file couldn't be read as an image. Try a JPEG or PNG photo.")
    st.stop()

st.image(uploaded, caption="Uploaded photo", use_container_width=True)

# The model scales pixels itself, so it takes the raw 0-255 array.
probabilities = model.predict(image[np.newaxis].astype("float32"), verbose=0)[0]
best = int(np.argmax(probabilities))

st.success(f"Predicted weather: **{class_names[best]}** ({probabilities[best]:.0%} confidence)")

st.subheader("All classes")
for index in np.argsort(probabilities)[::-1]:
    st.progress(float(probabilities[index]), text=f"{class_names[index]}: {probabilities[index]:.1%}")
