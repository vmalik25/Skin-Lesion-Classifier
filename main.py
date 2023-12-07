from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import tensorflow as tf
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True)

model = load_model("model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image):
    # resizing the image to be at least 224x224 and then cropping from the center
    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Expand dimensions to create a batch size of 1
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

def predict_skin_disease(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('Skin Lesion Classifier')

st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5; /* Light gray background */
            color: #333333; /* Dark text color */
            font-family: Arial, sans-serif; /* Use a common sans-serif font */
        }
        .st-eb {
            background-color: #3498db; /* Blue background for Streamlit elements */
            color: #ffffff; /* White text for Streamlit elements */
        }
        .stSelectbox {
            color: #333333; /* Dark text color for dropdown */
        }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

uploaded_file = col1.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if col2.button("Classify"):
    prediction = predict_skin_disease(uploaded_file)
    col2.write("### Confidence Scores:")
    for i, class_name in enumerate(class_names):
        confidence = prediction[0][i]
        col2.write(f"{class_name}: {confidence*100:.2f}%")


