import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩻",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_xray_model.keras")

model = load_model()

class_names = [
    "Lung Opacity",
    "No Lung Opacity / Not Normal",
    "Normal"
]

st.title("Chest X-ray Classification")

st.write(
    "Upload a chest X-ray image to obtain a model-based classification prediction."
)

st.info(
    "This application is intended for project demonstration and educational purposes. "
    "It is not a medical diagnostic tool."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width=400
    )

    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    probability = predictions[predicted_index]

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.write(
        f"Prediction Probability: {probability:.2%}"
    )

    st.subheader("Class Probabilities")

    for class_name, prob in zip(class_names, predictions):

        st.write(
            f"**{class_name}:** {prob:.2%}"
        )

        st.progress(float(prob))
