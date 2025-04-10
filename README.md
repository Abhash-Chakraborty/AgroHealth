# AgroHealth Analyzer

## **Description**

**AgroHealth Analyzer** is an AI-powered solution designed to detect plant diseases using computer vision and ensemble machine learning models. By integrating IoT hardware, cloud storage, and a user-friendly frontend, this project empowers farmers with real-time diagnostics, actionable insights, and disease management recommendations. It bridges the gap between advanced technology and agricultural needs, contributing to sustainable farming practices and global food security.

---

## **Table of Contents**

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [AI Model Explanation](#ai-model-explanation)
- [Frontend Features](#frontend-features)
- [Challenges Faced](#challenges-faced)
- [Future Enhancements](#future-enhancements)
- [Credits](#credits)
- [License](#license)

---

## **Features**

- **Automated Disease Detection**: Ensemble AI model analyzes plant images to determine health status.
- **IoT Integration**: ESP32-CAM captures high-quality images for analysis.
- **Cloud Storage**: Images are uploaded to cloud storage for preprocessing and evaluation.
- **Explainable AI**: Provides detailed insights into diseases, precautions, solutions, and next steps using Gemini 2.0 Flash Thinking API.
- **Flexible Image Input**: Supports direct upload of single or multiple images from local devices for users without hardware modules.
- **User-Friendly Frontend**: Streamlit-based interface for seamless interaction.

---

## **Installation**

Follow these steps to set up the project:

1. Clone the repository:
```bash
git clone https://github.com/Abhash-Chakraborty/AgroHealth.git
cd AgroHealth
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run main.py
```

---

## **Usage**

### **For Hardware Users**
1. Set up the ESP32-CAM module programmed to capture images at 800 x 600 pixels *(if necessary)*.
2. Images are automatically uploaded to cloud storage for preprocessing.
3. Access the Streamlit frontend to view diagnostic results.

### **For Non-Hardware Users**
1. Directly upload single or multiple leaf images from your local device through the frontend.
2. The system processes the images and provides health analysis.

---

## **AI Model Explanation**

The AI model is an **ensemble model**, consisting of four components:

| Model Type       | Image Resolution | Classes Trained On | Purpose                                                                 |
|------------------|------------------|--------------------|------------------------------------------------------------------------|
| Sub-model 1      | 128 x 128        | 107                | Analyzes resized images for disease detection                          |
| Sub-model 2      | 224 x 224        | 107                | Analyzes higher resolution images for disease detection                          |
| Sub-model 3      | 128 x 128        | 107                | Analyzes resized images for disease detection                |
| Classifier Model | 128 x 128            | 2                | Crops leaf images into useful parts before sending them to sub-models |

- The sub-models use **hard voting** to finalize the health status of the plant.
- This architecture ensures high accuracy and robustness in disease detection while minimizing overfitting.

---

## **Frontend Features**

1. **Hardware Integration**: Automatically processes images captured by ESP32-CAM modules.
2. **Direct Image Upload**: Allows users without hardware modules to upload leaf images directly from their local devices.
3. **Explainable AI Insights**: Provides detailed explanations about detected diseases, precautions, solutions, and future steps using Gemini 2.0 Flash Thinking API.

---

## **Challenges Faced**

| Challenge                           | Solution                                                                 |
|-------------------------------------|-------------------------------------------------------------------------|
| ESP32 module did not support JPEG format | Captured images in RGB565 format and converted them to JPEG before uploading to cloud storage |
| AI model overfitting during training | Implemented hard voting techniques with CNNs                           |
| Camera port malfunction             | Replaced defective ESP32 module                                         |
| Images unreadable due to unsupported formats | Converted RGB565 images to JPEG format                                 |
| Low camera frame rate causing delays | Optimized camera settings                                              |
| Difficulty finding free deployment platforms | Selected GitHub for deployment and Streamlit.app for frontend          |
| Lack of preprocessing standardization | Developed a secondary preprocessing model                              |

---

## **Future Enhancements**

1. **Lightweight Hardware Modules**: Explore Raspberry Pi or Jetson Nano for field deployment.
2. **Improved AI Models**: Optimize algorithms for faster inference times and higher accuracy under varying conditions.
3. **Hyperspectral Imaging**: Detect pre-symptomatic disease indicators using advanced sensors.
4. **Mobile Application Development**: Create a mobile app for real-time diagnostics accessible to farmers in remote areas.
5. **Cloud-Based Data Analytics**: Implement analytics to store and analyze large-scale agricultural data for predictive insights.

---

## **License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

### Visuals



---

### Last Updated

This README was last updated on April 10, 2025.
