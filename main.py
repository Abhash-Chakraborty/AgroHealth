import streamlit as st
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile

# Function to predict the class of uploaded images using a pre-trained TensorFlow model
def model_prediction(test_images):
    model = tf.keras.models.load_model('trained_model.h5')
    predictions = []
    for test_image in test_images:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        pred = model.predict(input_arr)
        result_index = np.argmax(pred)
        predictions.append(result_index)
    return predictions

# Define tabs for the application
Home, App, About = st.tabs(["🏠Home", "⚙️App", "ℹ️About"])

# Home tab content
with Home:
    st.header("Welcome to the Plant Disease Detection App")
    st.write("This application leverages a TensorFlow model to identify plant diseases from images.")
    st.image("OIP.jpeg", use_column_width=True)
    st.markdown("""
    ### How It Works
    1. **Upload Images**: Navigate to the **Disease Detection** tab and upload images of plants.
    2. **Analyze**: The app processes the images using a pre-trained machine learning model.
    3. **Results**: View the predicted disease class for each uploaded image.

    ### Features
    - Supports multiple image uploads.
    - Provides quick and accurate predictions.
    - User-friendly interface for seamless interaction.

    ### About the Model
    The model is trained on a diverse dataset of plant images to recognize various diseases with high accuracy. It uses state-of-the-art deep learning techniques to ensure reliable results.

    ### Get Started
    Go to the **Disease Detection** tab to upload your images and begin the analysis!
    """)

# App tab content
with App:
    st.header("Disease Detection")
    st.write("Upload plant images to detect potential diseases.")
    uploaded_files = st.file_uploader("Select images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        if st.toggle("Display Images"):
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_file = NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_files.append(temp_file.name)
                result_indices = model_prediction(temp_files)
                classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Apple___rust', 'Apple___scab', 'Blueberry___healthy', 'Cassava___bacterial_blight', 'Cassava___brown_streak_disease', 'Cassava___green_mottle', 'Cassava___healthy', 'Cassava___mosaic_disease', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry___healthy', 'Cherry___powdery_mildew', 'Chili___healthy', 'Chili___leaf curl', 'Chili___leaf spot', 'Chili___whitefly', 'Chili___yellowish', 'Coffee___cercospora_leaf_spot', 'Coffee___healthy', 'Coffee___red_spider_mite', 'Coffee___rust', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn___common_rust', 'Corn___gray_leaf_spot', 'Corn___healthy', 'Corn___northern_leaf_blight', 'Cucumber___diseased', 'Cucumber___healthy', 'Gauva___diseased', 'Gauva___healthy', 'Grape___black_measles', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Jamun___diseased', 'Jamun___healthy', 'Lemon___diseased', 'Lemon___healthy', 'Mango___diseased', 'Mango___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Pepper_bell___bacterial_spot', 'Pepper_bell___healthy', 'Pomegranate___diseased', 'Pomegranate___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Rice___brown_spot', 'Rice___healthy', 'Rice___hispa', 'Rice___leaf_blast', 'Rice___neck_blast', 'Soybean___bacterial_blight', 'Soybean___caterpillar', 'Soybean___diabrotica_speciosa', 'Soybean___downy_mildew', 'Soybean___healthy', 'Soybean___mosaic_virus', 'Soybean___powdery_mildew', 'Soybean___rust', 'Soybean___southern_blight', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Strawberry____leaf_scorch', 'Sugarcane___bacterial_blight', 'Sugarcane___healthy', 'Sugarcane___red_rot', 'Sugarcane___red_stripe', 'Sugarcane___rust', 'Tea___algal_leaf', 'Tea___anthracnose', 'Tea___bird_eye_spot', 'Tea___brown_blight', 'Tea___healthy', 'Tea___red_leaf_spot', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___mosaic_virus', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___spider_mites_(two_spotted_spider_mite)', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___yellow_leaf_curl_virus', 'Wheat___brown_rust', 'Wheat___healthy', 'Wheat___septoria', 'Wheat___yellow_rust']
                results = [classes[idx] for idx in result_indices]
                for file, result in zip(uploaded_files, results):
                    st.success(f"Result for {file.name}: {result}")

# About tab content
with About:
    st.header("About This Application")
    st.write("This project is designed to assist in identifying plant diseases using machine learning.")
    st.write("The goal is to provide farmers and agricultural professionals with a quick and accurate tool for disease detection.")