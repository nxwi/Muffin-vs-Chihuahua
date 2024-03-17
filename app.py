# pip3 install -r requirements.txt
# streamlit run app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    """Preprocesses an image for model prediction."""
    image = Image.open(image)
    image = image.resize((120, 120))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    return image_array.reshape(1, 120, 120, 3)  # Add batch dimension

def predict_breed(image_file):
    """Performs image preprocessing, prediction, and confidence score calculation."""
    preprocessed_image = preprocess_image(image_file)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class] * 100

    if predicted_class == 0:
        breed = "Chihuahua"
    else:
        breed = "Muffin"

    return breed, confidence_score

st.set_page_config(
    page_title="Muffin vs Chihuahua Classification",
    page_icon="💡",
    # layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': '',
    #     'Report a bug': '',
    #     'About': ''
    # }
    )

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">Muffin vs Chihuahua</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.image('image.jpg', width=700)
st.subheader("**Is it a Muffin or a Chihuahua?** Upload an image to find out!")

# Sidebar
st.sidebar.markdown('# Made by: [Naswih](https://github.com/nxwi)')
st.sidebar.markdown('# Repo link: [Muffin vs Chihuahua](https://github.com/nxwi/Muffin-vs-Chihuahua-Image-Classification)')

uploaded_file = st.file_uploader("Choose an image:", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_breed, confidence = predict_breed(uploaded_file)
    st.success(f"Predicted Breed: {predicted_breed} (Confidence: {confidence:.2f}%)")
