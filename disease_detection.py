import streamlit as st
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile
from collections import Counter
import google.generativeai as genai

# Cache the models to load them only once
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model('trained_model1.keras')
    model2 = tf.keras.models.load_model('trained_model2.keras')
    model3 = tf.keras.models.load_model('trained_model3.keras')
    return model1, model2, model3

# Function to predict using ensemble of models with hard voting
def model_prediction(test_images):
    model1, model2, model3 = load_models()
    
    final_predictions = []
    individual_predictions = []
    confidence_scores = []
    
    for test_image in test_images:
        image1 = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr1 = tf.keras.preprocessing.image.img_to_array(image1)
        input_arr1 = np.array([input_arr1])
        
        image2 = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr2 = tf.keras.preprocessing.image.img_to_array(image2)
        input_arr2 = np.array([input_arr2])
        
        image3 = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr3 = tf.keras.preprocessing.image.img_to_array(image3)
        input_arr3 = np.array([input_arr3])
        
        pred1 = model1.predict(input_arr1)
        pred2 = model2.predict(input_arr2)
        pred3 = model3.predict(input_arr3)
        
        class1 = np.argmax(pred1, axis=1)[0]
        class2 = np.argmax(pred2, axis=1)[0]
        class3 = np.argmax(pred3, axis=1)[0]
        
        individual_predictions.append((class1, class2, class3))
        confidence_scores.append((np.max(pred1), np.max(pred2), np.max(pred3)))
        
        votes = [class1, class2, class3]
        vote_count = Counter(votes)
        final_prediction = vote_count.most_common(1)[0][0]
        
        final_predictions.append(final_prediction)
    
    return final_predictions, individual_predictions, confidence_scores

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
    response = model.generate_content(prompt)
    return response.text

st.header("Disease Detection")
st.write("Upload plant images to detect potential diseases using ensemble learning.")

uploaded_files = st.file_uploader("Select images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.toggle("Display Images"):
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing with ensemble models..."):
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_file = NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_files.append(temp_file.name)
            
            result_indices, individual_indices, confidence_scores = model_prediction(temp_files)
            
            classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Apple___rust', 'Apple___scab', 'Blueberry___healthy', 'Cassava___bacterial_blight', 'Cassava___brown_streak_disease', 'Cassava___green_mottle', 'Cassava___healthy', 'Cassava___mosaic_disease', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry___healthy', 'Cherry___powdery_mildew', 'Chili___healthy', 'Chili___leaf curl', 'Chili___leaf spot', 'Chili___whitefly', 'Chili___yellowish', 'Coffee___cercospora_leaf_spot', 'Coffee___healthy', 'Coffee___red_spider_mite', 'Coffee___rust', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn___common_rust', 'Corn___gray_leaf_spot', 'Corn___healthy', 'Corn___northern_leaf_blight', 'Cucumber___diseased', 'Cucumber___healthy', 'Gauva___diseased', 'Gauva___healthy', 'Grape___black_measles', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Jamun___diseased', 'Jamun___healthy', 'Lemon___diseased', 'Lemon___healthy', 'Mango___diseased', 'Mango___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Pepper_bell___bacterial_spot', 'Pepper_bell___healthy', 'Pomegranate___diseased', 'Pomegranate___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Rice___brown_spot', 'Rice___healthy', 'Rice___hispa', 'Rice___leaf_blast', 'Rice___neck_blast', 'Soybean___bacterial_blight', 'Soybean___caterpillar', 'Soybean___diabrotica_speciosa', 'Soybean___downy_mildew', 'Soybean___healthy', 'Soybean___mosaic_virus', 'Soybean___powdery_mildew', 'Soybean___rust', 'Soybean___southern_blight', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Strawberry____leaf_scorch', 'Sugarcane___bacterial_blight', 'Sugarcane___healthy', 'Sugarcane___red_rot', 'Sugarcane___red_stripe', 'Sugarcane___rust', 'Tea___algal_leaf', 'Tea___anthracnose', 'Tea___bird_eye_spot', 'Tea___brown_blight', 'Tea___healthy', 'Tea___red_leaf_spot', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___mosaic_virus', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___spider_mites_(two_spotted_spider_mite)', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___yellow_leaf_curl_virus', 'Wheat___brown_rust', 'Wheat___healthy', 'Wheat___septoria', 'Wheat___yellow_rust']
            
            for i, (file, result, individual, confidence) in enumerate(zip(uploaded_files, result_indices, individual_indices, confidence_scores)):
                st.subheader(f"Analysis for {file.name}")
                
                # Display final prediction
                st.success(f"Final prediction: {(classes[result].split('__')[0]).split('_')[0]} {classes[result].split('__')[1].replace('_', ' ').title()}")
                # Display individual model predictions
                st.subheader("Model Predictions")
                for j, (model_name, pred, conf) in enumerate([
                    ("Model 1 (128×128)", individual[0], confidence[0]),
                    ("Model 2 (224×224)", individual[1], confidence[1]),
                    ("Model 3 (128×128)", individual[2], confidence[2])
                ]):
                    st.metric(
                        label=model_name,
                        value=classes[pred],
                        delta=f"{conf*100:.1f}% confidence"
                    )
                        
                # Show voting details
                st.subheader("Voting Breakdown")
                for class_idx, count in Counter([individual[0], individual[1], individual[2]]).items():
                        st.write(f"- {classes[class_idx]}: {count} vote(s)")
                
                # Add a button to open the sidebar
                with st.sidebar:
                    st.header(f"Detailed Analysis for {file.name}")
                        
                    # Get Gemini API response
                    prompt = f"Analyze the following plant disease detection results:\n\nFinal prediction: {classes[result]}\n\nIndividual model predictions:\nModel 1: {classes[individual[0]]}\nModel 2: {classes[individual[1]]}\nModel 3: {classes[individual[2]]}\n\nProvide insights on the disease, potential causes, and recommended actions for the farmer.\nDon't include any thing that is not related to the analysis.\ndont include Model Discrepancy as it is not beneficial for the farmer.\n\n Tell the farmer, how to take care of the plant and what to do next, if there is a disease detected otherwise tell the farmer how to mantain the crop.\n\nDon't tell the end user what that some model gave this prediction and some model gave that prediction.\n\nThe final prediction is the most important one.\n\nThe models are trained on the following classes:\n{', '.join(classes)}\n\nThe final prediction is {classes[result]}.\n\nOnly provide the analysis and recommendations based on the final prediction."
                        
                    gemini_response = get_gemini_response(prompt)
                        
                    st.subheader("AI Analysis")
                    st.write(gemini_response)
                
                if i < len(uploaded_files) - 1:
                    st.divider()
