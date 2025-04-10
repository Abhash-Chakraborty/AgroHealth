import streamlit as st
import google.generativeai as genai

# Configure Gemini API
def configure_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Set up the navigation
def main():
    # Configure the page
    st.set_page_config(
        page_title="AgroHealth Analyzer",
        page_icon="🌿",
        layout="wide"
    )
    
    # Configure Gemini API
    configure_gemini()
    
    # Define pages
    home_page = st.Page("home.py", title="Home", icon="🏠")
    app_page = st.Page("disease_detection.py", title="Disease Detection", icon="⚙️")
    about_page = st.Page("about.py", title="About", icon="ℹ️")
    download_page = st.Page("download.py", title="Download", icon="⬇️")
    
    # Set up navigation
    page = st.navigation([home_page, app_page, about_page, download_page])
    
    # Run the selected page
    page.run()

if __name__ == "__main__":
    main()
