import streamlit as st
import os
import subprocess

st.header("Backend Images")

# Define the storage path
storage_path = './storage/'

# Check if the storage folder exists and contains files
folder_exists = os.path.exists(storage_path)
files_in_folder = os.listdir(storage_path) if folder_exists else []

if folder_exists and files_in_folder:
    # Display files in a 3-column grid
    st.write("Processed Images")
    cols = st.columns(3)
    for i, file in enumerate(files_in_folder):
        with cols[i % 3]:
            file_path = os.path.join(storage_path, file)
            st.image(file_path, caption=file, use_container_width=True)
    
    # Add an "Update" button
    if st.button("Update", key="update_button"):
        with st.spinner("Updating processed images..."):
            try:
                # Run the server-side script
                result = subprocess.run(["python", "serversidescript.py"], 
                                        capture_output=True, 
                                        text=True, 
                                        check=True)
                
                # Display success message
                st.success("Images updated successfully!")
                
                # Display the output from the script
                with st.expander("View Update Output"):
                    st.code(result.stdout)
                
                # Refresh the page to show updated images
                st.rerun()
            
            except subprocess.CalledProcessError as e:
                # Display error message if the script fails
                st.error(f"Error updating images: {e}")
                st.code(e.stderr)
            except Exception as e:
                # Handle other exceptions
                st.error(f"An error occurred: {str(e)}")
else:
    # Display "Synchronize Images" button
    st.write("No processed images available. Click the button below to synchronize images.")
    if st.button("Synchronize Images", key="sync_button"):
        with st.spinner("Running server-side script to download processed images..."):
            try:
                # Run the server-side script
                result = subprocess.run(["python", "serversidescript.py"], 
                                        capture_output=True, 
                                        text=True, 
                                        check=True)
                
                # Display success message
                st.success("Server-side script executed successfully!")
                
                # Display the output from the script
                with st.expander("View Script Output"):
                    st.code(result.stdout)
                
                # Refresh the page to show newly processed images
                st.rerun()
            
            except subprocess.CalledProcessError as e:
                # Display error message if the script fails
                st.error(f"Error executing server-side script: {e}")
                st.code(e.stderr)
            except Exception as e:
                # Handle other exceptions
                st.error(f"An error occurred: {str(e)}")
