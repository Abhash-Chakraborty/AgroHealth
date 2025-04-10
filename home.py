import streamlit as st

st.header("Welcome to AgroHealth Analyzer")
st.write("This application leverages ensemble learning with multiple CNN models to identify plant diseases from images.")
st.image("hero.png", use_container_width=True)
st.markdown("""
### How It Works
1. **Upload Images**: Navigate to the **Disease Detection** page and upload images of plants.
2. **Analyze**: The app processes the images using an ensemble of pre-trained CNN models.
3. **Results**: View the predicted disease class for each uploaded image based on majority voting.

### Features
- Supports multiple image uploads.
- Uses ensemble learning with hard voting for improved accuracy.
- Provides quick and reliable predictions.
- Shows individual model predictions for transparency.
- User-friendly interface for seamless interaction.

### About the Ensemble Model
The system uses three different models trained on plant disease datasets:
- Two models trained on 128×128 images
- One model trained on 224×224 images

Hard voting is used to combine predictions, which helps reduce errors and improve overall accuracy.

### Get Started
Go to the **Disease Detection** page to upload your images and begin the analysis!
""")
