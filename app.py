import streamlit as st
import os
import subprocess

# Fix OpenCV issue
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

try:
    subprocess.run(["pip", "uninstall", "-y", "opencv-python"])
except:
    pass

from ultralytics import YOLO
from PIL import Image
import tempfile
import requests

st.set_page_config(page_title="Mango Tree Counter", page_icon="🌳")

st.title("🌳 Mango Tree Detection & Counting")

MODEL_PATH = "best.pt"

# Download model
if not os.path.exists(MODEL_PATH):
    st.info("⬇️ Downloading model...")

    file_id = "1Fok4Y8IIU"  # your file id
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(url)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    st.success("✅ Model downloaded!")

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.success("✅ Model loaded!")

# Upload images
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    total = 0

    for file in uploaded_files:
        img = Image.open(file)
        st.image(img, caption=file.name)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp.name)

        results = model(temp.name)

        count = len(results[0].boxes)
        total += count

        st.image(results[0].plot(), caption=f"Trees: {count}")

    st.success(f"🌴 Total Trees: {total}")
