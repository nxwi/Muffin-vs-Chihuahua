# pip3 install -r requirements.txt
# streamlit run app.py


# import streamlit as st
# import pickle


# def main():
#     st.set_page_config(
#     page_title="Muffin vs Chihuahua Classification",
#     page_icon="💡",
#     # layout="wide",
#     initial_sidebar_state="expanded",
#     # menu_items={
#     #     'Get Help': 'https://www.extremelycoolapp.com/help',
#     #     'Report a bug': "https://www.extremelycoolapp.com/bug",
#     #     'About': "# This is a header. This is an *extremely* cool app!"
#     # }
#     )

#     gradient_text_html = """
#     <style>
#     .gradient-text {
#         font-weight: bold;
#         background: -webkit-linear-gradient(left, red, orange);
#         background: linear-gradient(to right, red, orange);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         display: inline;
#         font-size: 3em;
#     }
#     </style>
#     <div class="gradient-text">Muffin vs Chihuahua</div>
#     """

#     st.markdown(gradient_text_html, unsafe_allow_html=True)
#     st.image('image.jpg', width=700)
#     st.write('In the section below, I will take you through the task of Insurance Prediction with Machine Learning using Python. For the task of Insurance prediction with machine learning, I have collected a dataset from Kaggle about the previous customers of a travel insurance company. Here our task is to train a machine learning model to predict whether an individual will purchase the insurance policy from the company or not.')

#     # Sidebar
#     st.sidebar.markdown('# Made by: [Naswih](https://github.com/nxwi)')
#     st.sidebar.markdown('# Git link: [Docsummarizer](https://github.com/e-johnstonn/docsummarizer)')
#     st.sidebar.markdown("""<small>It's always good practice to verify that a website is safe before giving it your API key. 
#                         This site is open source, so you can check the code yourself, or run the streamlit app locally.</small>""", unsafe_allow_html=True)


#     file = st.file_uploader('Upload An Image')

#     age = st.number_input('Age', step=1, placeholder='Type here')
#     option = st.selectbox('Employment Type', ('Government Sector', 'Private Sector/Self Employed'))
#     employmentType = 0 if option == 'Government Sector' else 1
#     annualIncome = st.number_input('Annual Income', step=1, placeholder='Type here')
#     familyMembers = st.number_input('Family Members', step=1, placeholder='Type here')
#     graduateOrNot = st.checkbox('Graduated')
#     chronicDiseases = st.checkbox('Chronic Diseases')
#     frequentFlyer = st.checkbox('Frequent Flyer')
#     everTravelledAbroad = st.checkbox('Ever Travelled Abroad')

#     features = [age, employmentType, graduateOrNot, annualIncome, familyMembers, chronicDiseases, frequentFlyer,
#                 everTravelledAbroad]

#     model = pickle.load(open('model.sav', 'rb'))  # rb-->read binarey

#     btn = st.button('PREDICT')  # FOR PREDICTION BUTTON

#     if btn:
#         prediction = model.predict([features])
#         if prediction == 0:
#             st.write('## Will Buy INSURANCE')  # writ-->instead of print in st
#         else:
#             st.write('## Will Not Buy Insurance')


# main()


import streamlit as st
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from PIL import Image
import pickle

# Load your pre-trained TensorFlow model (replace with your specific model loading code)
model = pickle.load(open('model.sav', 'rb'))

def predict_breed(image_file):
    """Performs image preprocessing, prediction, and confidence score calculation."""
    image = Image.open(image_file)
    image = image.resize((224, 224))  # Assuming your model expects 224x224 input
    # image_array = tf.keras.preprocessing.image.img_to_array(image)
    # image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    # image_batch = tf.expand_dims(image_array, axis=0)  # Add batch dimension
    # predictions = model.predict(image_batch)
    predictions = model.predict(resize(image,(120,120,3)).reshape(1,120,120,3)).argmax(axis=1).item()
    predicted_class = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class] * 100

    if predicted_class == 0:
        breed = "Muffin"
    else:
        breed = "Chihuahua"

    return breed, confidence_score

st.title("Muffin vs. Chihuahua Breed Predictor")
st.subheader("Using your pre-trained TensorFlow model")

uploaded_file = st.file_uploader("Choose an image of a dog:", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Breed"):
        predicted_breed, confidence = predict_breed(uploaded_file)
        st.success(f"Predicted Breed: {predicted_breed} (Confidence: {confidence:.2f}%)")

st.write("**Note:**")
st.write("- Ensure your pre-trained model is trained to classify Muffin and Chihuahua breeds.")
st.write("- The model's accuracy depends on the quality of your training data and the chosen model architecture.")

