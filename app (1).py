import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load model
model = YOLO("best.pt")

# Title
st.title("🌳 Mango Tree Detection & Counting")
st.write("Upload multiple images to detect and count mango trees")

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# Process images
if uploaded_files:
    total_count = 0

    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file)

        # Show uploaded image
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        image.save(temp_file.name)

        # Predict
        results = model(temp_file.name)

        # Count trees
        count = len(results[0].boxes)
        total_count += count

        # Show result image
        result_img = results[0].plot()
        st.image(result_img, caption=f"Detected Trees: {count}", use_column_width=True)

    # Show total count
    st.success(f"🌴 Total Trees in All Images: {total_count}")