import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = 'teeth_resnet50_model.keras'
CLASS_NAMES_PATH = 'class_names.txt'

st.set_page_config(page_title="Teeth Disease Classifier", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model()

st.title("Teeth Disease Classifier")
st.write(
    "Upload an image of an oral/dental condition, and the model "
    "will predict the "
    "most likely condition among 7 categories."
)

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

  
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)[0]

    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    st.subheader("Prediction")
    st.success(f"**{predicted_class}** ({confidence:.1f}% confidence)")
