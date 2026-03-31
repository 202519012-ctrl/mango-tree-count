import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="Mango Tree Counter", page_icon="🌳")

st.title("🌳 Mango Tree Detection & Counting")
st.write("Upload images to detect and count mango trees")

# ✅ Load model safely (prevents crash)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Processing
if uploaded_files:
    total_count = 0

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        st.image(image, caption=f"Uploaded: {uploaded_file.name}")

        # Save temp image
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        image.save(temp_file.name)

        try:
            results = model(temp_file.name)

            count = len(results[0].boxes)
            total_count += count

            result_img = results[0].plot()

            st.image(result_img, caption=f"Detected Trees: {count}")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    st.success(f"🌴 Total Trees in All Images: {total_count}")