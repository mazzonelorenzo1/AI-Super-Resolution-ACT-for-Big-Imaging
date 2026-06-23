import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import gc

# --- CUSTOM MODEL IMPORTS ---
# Import all the models created during the project using aliases to avoid name collisions
try:
    from model import SuperResolutionModel as BaselineModel
    from gan_model import SRGANModel as SRGAN_VGG
    from gan_model_resnet import SRGANModel as SRGAN_ResNet
    from gan_model_attention import SRGANModel as SRGAN_Attention
    from gan_model_double import Stage1DenoisingModel as DecoupledStage1, SRGANModel as DecoupledStage2

    CUSTOM_MODELS_AVAILABLE = True
except ImportError as e:
    CUSTOM_MODELS_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)

# --- SOTA MODEL IMPORTS ---
try:
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Super Resolution Platform", page_icon="🔎", layout="wide")

# ==========================================
# 1. CHECKPOINT PATHS CONFIGURATION
# ==========================================
CKPT_PATHS = {
    "Baseline ESPCN": "C:/Users/mazzo/PycharmProjects/BigImaging/checkpoints/best-dlss-model-epoch=41-val_loss=0.0044.ckpt",
    "SRGAN + VGG19": "checkpoints/best2-VGG-gan-epoch=35-val_psnr=22.32.ckpt",
    "SRGAN + ResNet50": "checkpoints/best-resnet-VGG-gan-epoch=38-val_psnr=20.39.ckpt",
    "SRGAN + Attention": "checkpoints/best-attention-VGG-gan-epoch=41-val_psnr=20.73.ckpt",
    "SRGAN + Attention (Official Dataset)": "checkpoints/best-official-attention-VGG-gan-epoch=44-val_psnr=25.01.ckpt",
    "Decoupled Stage 1": "checkpoints/stage1/denoiser-epoch=epoch=13-val_psnr=val_psnr=27.54.ckpt",
    "Decoupled Stage 2": "checkpoints/stage2/srgan-attention-epoch=epoch=38-val_psnr=val_psnr=22.71.ckpt"
}


# ==========================================
# 2. CACHED MODEL LOADING FUNCTIONS
# ==========================================
# We use st.cache_resource to avoid reloading weights from disk every time,
# but we clear unused models from VRAM automatically when switching.

@st.cache_resource
def load_custom_model(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if model_name == "Baseline ESPCN":
            model = BaselineModel.load_from_checkpoint(CKPT_PATHS[model_name])
        elif model_name == "SRGAN + VGG19":
            model = SRGAN_VGG.load_from_checkpoint(CKPT_PATHS[model_name])
        elif model_name == "SRGAN + ResNet50":
            model = SRGAN_ResNet.load_from_checkpoint(CKPT_PATHS[model_name])
        elif model_name == "SRGAN + Attention":
            model = SRGAN_Attention.load_from_checkpoint(CKPT_PATHS[model_name])
        elif model_name == "SRGAN + Attention (Official Dataset)":
            model = SRGAN_Attention.load_from_checkpoint(CKPT_PATHS[model_name])
        else:
            return None, None

        model.eval().to(device)
        return model, device
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_decoupled_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model1 = DecoupledStage1.load_from_checkpoint(CKPT_PATHS["Decoupled Stage 1"])
        model2 = DecoupledStage2.load_from_checkpoint(CKPT_PATHS["Decoupled Stage 2"])
        model1.eval().to(device)
        model2.eval().to(device)
        return model1, model2, device
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_swinir_sota():
    processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    model_sota = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    return processor, model_sota


# ===================================
# 3. PROCESSING & TILING LOGIC
# ===================================
def apply_opencv_denoise(image_pil):
    """Applies Fast Non-Local Means Denoising using OpenCV before neural processing."""
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, 10, 10, 7, 21)
    image_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def process_with_tiling(input_tensor, model1, model2=None, tile_size=64, scale_factor=4):
    """
    Splits the image into tiles, processes them one by one, and stitches them back.
    Supports both Single-Stage (model2=None) and Two-Stage Decoupled processing.
    """
    _, _, h, w = input_tensor.shape
    output_tensor = torch.zeros((1, 3, h * scale_factor, w * scale_factor), device=input_tensor.device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = input_tensor[:, :, y:y_end, x:x_end]

            with torch.no_grad():
                if model2 is not None:
                    # Decoupled Mode: Denoise first, then Upscale
                    clean_tile = model1(tile)
                    out_tile = model2(clean_tile)
                else:
                    # Single Stage Mode
                    out_tile = model1(tile)

            out_y, out_y_end = y * scale_factor, y_end * scale_factor
            out_x, out_x_end = x * scale_factor, x_end * scale_factor

            output_tensor[:, :, out_y:out_y_end, out_x:out_x_end] = out_tile

    return output_tensor


def upscale_image_custom(image, model_name, apply_cv2_denoise=False):
    if apply_cv2_denoise:
        image = apply_opencv_denoise(image)

    transform = transforms.ToTensor()

    # Load appropriate model(s)
    if model_name == "Decoupled Pipeline (Two Stages)":
        model1, model2, device_or_err = load_decoupled_pipeline()
        if model1 is None:
            return None, f"Failed to load Decoupled Checkpoints: {device_or_err}"
        input_tensor = transform(image).unsqueeze(0).to(device_or_err)
        output_tensor = process_with_tiling(input_tensor, model1, model2, tile_size=64, scale_factor=4)
    else:
        model, device_or_err = load_custom_model(model_name)
        if model is None:
            return None, f"Failed to load Checkpoint for {model_name}: {device_or_err}"
        input_tensor = transform(image).unsqueeze(0).to(device_or_err)
        output_tensor = process_with_tiling(input_tensor, model, model2=None, tile_size=64, scale_factor=4)

    output_tensor = output_tensor.squeeze(0).cpu().permute(1, 2, 0)
    output_np = (output_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np), None


def upscale_image_sota(image, processor, model_sota, apply_cv2_denoise=False):
    if apply_cv2_denoise:
        image = apply_opencv_denoise(image)

    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sota(**inputs)
    output_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_tensor = np.moveaxis(output_tensor, source=0, destination=-1)
    output_np = (output_tensor * 255.0).round().astype(np.uint8)
    return Image.fromarray(output_np)


# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.title("AI Super Resolution Platform")
st.write("Compare the progression of our deep learning architectures for Image Restoration and 4x Upscaling.")

if not CUSTOM_MODELS_AVAILABLE:
    st.error(
        f"Missing local python model files! Ensure gan_model.py, dataset.py etc. are in the root directory. Error: {IMPORT_ERROR_MSG}")
    st.stop()

# MODEL SELECTION MENU
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select the AI Engine:",
    [
        "Baseline ESPCN",
        "SRGAN + VGG19",
        "SRGAN + ResNet50",
        "SRGAN + Attention",
        "SRGAN + Attention (Official Dataset)",
        "Decoupled Pipeline (Two Stages)",
        "SOTA: SwinIR Transformer x4"
    ]
)

apply_denoise = st.sidebar.checkbox("Apply OpenCV Pre-Denoising", value=False,
                                    help="Smooths out extreme noise before passing the image to the neural network.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Use the different models to evaluate how the architecture evolved over the course of the project. The Decoupled Pipeline represents the final proposed solution.")

# FILE UPLOADER
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # IMAGE PROCESSING SECTION
    if file_ext in ['jpg', 'jpeg', 'png']:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.write(f"**Original Resolution:** {original_image.width}x{original_image.height} pixels")

        if st.button("Apply Super Resolution", type="primary"):
            with st.spinner(
                    f"The neural network ({model_choice}) is processing the image. Tiling is active to save VRAM..."):

                # Cleanup VRAM before massive processing
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if "SOTA" in model_choice:
                    if TRANSFORMERS_AVAILABLE:
                        processor_sota, model_sota = load_swinir_sota()
                        upscaled_image = upscale_image_sota(original_image, processor_sota, model_sota, apply_denoise)
                        success = True
                    else:
                        st.error("Transformer library missing. Open terminal and type: pip install transformers")
                        success = False
                else:
                    upscaled_image, error_msg = upscale_image_custom(original_image, model_choice, apply_denoise)
                    if error_msg:
                        st.error(error_msg)
                        success = False
                    else:
                        success = True

                if success:
                    st.success("Upscaling completed successfully!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Input")
                        st.image(original_image, width='stretch')
                    with col2:
                        st.subheader(f"Reconstruction ({model_choice})")
                        st.image(upscaled_image, width='stretch')

else:
    st.info("Upload an image from the box above to start processing.")
