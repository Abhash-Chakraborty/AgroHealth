import streamlit as st

st.header("About AgroHealth Analyzer")
st.write("This project is designed to assist in identifying plant diseases using ensemble machine learning.")
st.write("The application uses three different CNN models and combines their predictions through hard voting to provide more accurate results.")

st.subheader("Technical Details")
st.markdown("""
- **Model Architecture**: Convolutional Neural Networks (CNNs)
- **Ensemble Method**: Hard Voting (Majority Vote)
- **Input Sizes**: Two models use 128×128 images, one model uses 224×224 images
- **Dataset**: Trained on a comprehensive dataset of plant disease images

### References
- Mohanty, S.P., Hughes, D.P. & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.
- Ferentinos, K.P. (2018). Deep learning models for plant disease detection and diagnosis. Computers and Electronics in Agriculture, 145, 311-318.
- Geetharamani, G., & Pandian, A. (2019). Identification of plant leaf diseases using a nine-layer deep convolutional neural network. Computers & Electrical Engineering, 76, 323-338.
""")

st.subheader("Project Goal")
st.write("The goal is to provide farmers and agricultural professionals with a reliable tool for early disease detection, helping to reduce crop losses and improve agricultural productivity.")

st.subheader("Future Improvements")
st.markdown("""
- Implement soft voting (weighted average of probabilities)
- Add disease treatment recommendations
- Develop a mobile application for field use
- Expand the dataset to include more plant species and disease types
- Implement explainable AI features to help users understand model decisions
""")
