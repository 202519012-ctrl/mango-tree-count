app.py
import gdown
import os

MODEL_PATH = "best.pt"

# Download model from Drive if not present
if not os.path.exists(MODEL_PATH):
    st.info("⬇️ Downloading model...")

    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)

    st.success("✅ Model downloaded!")
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page settings
st.set_page_config(page_title="Mango Tree Counter", page_icon="🌳")

st.title("🌳 Mango Tree Detection & Counting")
st.write("Upload images to detect and count mango trees")

# ✅ Check if model exists
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'best.pt' not found. Please upload it to GitHub.")
    st.stop()

# ✅ Load model safely
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    st.info("⏳ Loading model...")
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Upload images
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Process images
if uploaded_files:
    total_count = 0

    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)

            st.image(image, caption=f"Uploaded: {uploaded_file.name}")

            # Save temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(temp_file.name)

            # Run prediction
            results = model(temp_file.name)

            count = len(results[0].boxes)
            total_count += count

            # Show result image
            result_img = results[0].plot()
            st.image(result_img, caption=f"Detected Trees: {count}")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    st.success(f"🌴 Total Trees in All Images: {total_count}")
