import os
import io
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

# Define constants
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'agrohealth-456405-4f4333b07084.json' 
FOLDER_ID = '1JYKlQtLOGUGm9PReSfJCtoqTf4VsoSkE'  
TEMP_PATH = './temp/'  
STORAGE_PATH = './storage/'  
MODEL_PATH = 'leaf_classifier_model.h5'  

def authenticate_google_drive():
    """Authenticate using a service account."""
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def list_files_in_folder(drive_service, folder_id):
    """List all files in a specific Google Drive folder."""
    query = f"'{folder_id}' in parents"
    response = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return response.get('files', [])

def download_file(drive_service, file_id, file_name, download_path):
    """Download a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    file_path = os.path.join(download_path, file_name)
    
    with io.BytesIO() as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download Progress: {int(status.progress() * 100)}%")
        
        # Write the downloaded content to a local file
        with open(file_path, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())
    
    print(f"File downloaded: {file_path}")
    return file_path

def crop_individual_leaves(image_path, output_dir, base_filename):
    """Crop individual leaves from bunch images."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return []
    
    # Create grayscale for processing without modifying original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    min_contour_area = 1000  # Adjust based on your images
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Crop and save individual leaves
    cropped_paths = []
    for i, contour in enumerate(valid_contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding around the bounding box
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        # Crop the image (keeping original BGR format)
        cropped = image[y_start:y_end, x_start:x_end]
        
        # Skip if cropped image is too small
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue
        
        # Make the cropped image square
        height, width, _ = cropped.shape
        max_dim = max(height, width)
        square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)  # Black background
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped
        
        # Resize to 224x224
        resized_image = cv2.resize(square_image, (224, 224))
        
        # Save cropped image directly to output directory
        filename = f"{base_filename}_leaf_{i}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_image)  # Save in original BGR format
        cropped_paths.append(output_path)
    
    return cropped_paths

def process_image(image_path, output_dir, model):
    """Process an image using the leaf classifier model."""
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return []
    
    # Create a copy for model input (needs RGB conversion for the model)
    img_for_model = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # For model input, resize to model's expected size
    img_size = (128, 128)  # Keep the model's expected input size
    img_resized = cv2.resize(img_for_model, img_size)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Classify as single leaf or bunch of leaves
    bunch_prediction = model.predict(img_array)[0][0]
    
    processed_paths = []
    
    if bunch_prediction < 0.5:  # Single leaf
        # Make the image square first (using original BGR image)
        height, width, _ = img.shape
        max_dim = max(height, width)
        square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)  # Black background
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_image[y_offset:y_offset + height, x_offset:x_offset + width] = img
        
        # Resize to 224x224
        output_image = cv2.resize(square_image, (224, 224))
        
        # Save single leaf image directly to output folder
        output_path = os.path.join(output_dir, f"{base_filename}.jpg")
        cv2.imwrite(output_path, output_image)  # Save in original BGR format
        processed_paths.append(output_path)
    else:  # Bunch of leaves
        # Crop individual leaves and save directly to output folder
        processed_paths = crop_individual_leaves(image_path, output_dir, base_filename)
    
    return processed_paths

def main():
    """Main function to download, process, and store leaf images."""
    # Authenticate and create the Drive service
    drive_service = authenticate_google_drive()

    # Ensure the temp and storage directories exist
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)

    # Load the leaf classifier model
    print("Loading leaf classifier model...")
    model = load_model(MODEL_PATH)

    # List all files in the specified folder
    print("Fetching files from Google Drive folder...")
    files = list_files_in_folder(drive_service, FOLDER_ID)

    if not files:
        print("No files found in the specified folder.")
        return

    try:
        # Download and process each file
        for file in files:
            file_id = file['id']
            file_name = file['name']
            
            # Skip non-image files
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skipping non-image file: {file_name}")
                continue
                
            print(f"Downloading {file_name}...")
            file_path = download_file(drive_service, file_id, file_name, TEMP_PATH)
            
            print(f"Processing {file_name}...")
            processed_paths = process_image(file_path, STORAGE_PATH, model)
            
            print(f"Processed {len(processed_paths)} images from {file_name}")
    
    finally:
        # Clean up: remove the temp directory
        print("Cleaning up temporary files...")
        if os.path.exists(TEMP_PATH):
            shutil.rmtree(TEMP_PATH)
            print(f"Removed temporary directory: {TEMP_PATH}")

    print("Processing complete! All images have been saved to:", STORAGE_PATH)

if __name__ == '__main__':
    main()
