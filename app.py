import streamlit as st
import torch
import numpy as np
import cv2
import os
import sys

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from src.models.unet import UNet
from src.utils.visualization import overlay_mask

# Set page config
st.set_page_config(page_title="NeuroSeg - Brain MRI segmentation", layout="wide")

st.title("NeuroSeg: Automated Brain MRI Segmentation")
st.markdown("""
This tool uses a U-Net architecture to segment brain tumors from MRI slices.
""")

# Load model
@st.cache_resource
def load_model(weights_path):
    model = UNet(n_channels=3, n_classes=1)
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            st.sidebar.success("Loaded weights from best_model.pth")
        except Exception as e:
            st.sidebar.error(f"Error loading weights: {e}")
    else:
        st.sidebar.warning("No pre-trained model found (best_model.pth). Using uninitialized weights.")
    model.eval()
    return model

weights_path = "best_model.pth"
model = load_model(weights_path)

# Sidebar
st.sidebar.header("Inputs")
uploaded_file = st.sidebar.file_uploader("Upload an MRI Slice (.tif, .jpg, .png)", type=["tif", "jpg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original MRI Slice")
        st.image(image_rgb, use_column_width=True)
        
    # Preprocess
    img_input = image_rgb.astype(np.float32)
    img_input = (img_input - np.mean(img_input)) / (np.std(img_input) + 1e-8)
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = torch.from_numpy(img_input).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(img_input)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).squeeze().cpu().numpy()
        
    with col2:
        st.subheader("Segmentation Overlay")
        overlay = overlay_mask(image_rgb, mask)
        st.image(overlay, use_column_width=True)
        
    # Quantitative Analysis
    st.divider()
    st.subheader("Quantitative Analysis")
    tumor_pixels = np.sum(mask)
    st.metric("Detected Tumor Surface Area", f"{tumor_pixels} pixels")
    
    if tumor_pixels > 0:
        st.success("Tumor detected in slice.")
    else:
        st.info("No significant tumor detected in slice.")
else:
    # Sample images if available
    st.info("Please upload an MRI slice or use a sample.")
    if os.path.exists("archive/kaggle_3m"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Sample Images")
        # Just list some samples
        samples = [
            "TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11.tif",
            "TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_10.tif"
        ]
        for s in samples:
            if st.sidebar.button(f"Load {s}"):
                path = os.path.join("archive/kaggle_3m", s)
                if os.path.exists(path):
                    image = cv2.imread(path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption=s)
                    # Implementation for sample prediction could be added here
