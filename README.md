#  🔎 AI Super Resolution - ACT for Big Imaging Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📑 Table of Contents
- [About the Project](#intro)
- [Repository Structure](#struct)
- [Installation](#install)
- [How to Run the Demo](#demo)
- [Results](#results)
- [Aknowledgments](#aknow)

<a id="intro"></a>
## 📖 About the Project
This project tackles the Image Super-Resolution and Restoration problem in real-world scenarios. The goal is to reconstruct a High-Resolution image starting from a heavily degraded Low-Resolution image, affected by sensor noise, blur, and JPEG compression artifacts.

While classic Generative Adversarial Networks (GANs) struggle with simultaneous denoising and upscaling (often resulting in gradient collapse, checkerboard artifacts, or severe color shifts), this project proposes a Decoupled Restoration Pipeline:

1. **Stage 1 (The Denoiser):** A ResNet-based CNN optimized purely on Mean Squared Error (MSE) to clean the image without hallucinating details.

2. **Stage 2 (The Enhancer):** An SRGAN powered by Self-Attention blocks and a ResNet50 Perceptual Loss, trained explicitly to inject high-frequency textures into the clean output of Stage 1.

The project also features a custom solution for the Domain Gap problem using Teacher Forcing and pixel-perfect geometric alignment.

For in-depth analysis of this project please refer to the [presentation](https://github.com/mazzonelorenzo1/AI-Super-Resolution-ACT-for-Big-Imaging/blob/main/Canva_Presentation).

<a id="struct"></a>
## 📂 Repository Structure
The codebase is organized modularly, separating data loading, model architectures, and training logic.

```text
📁 AI-Super-Resolution-ACT-for-Big-Imaging/
│
├── 📁 Dataset/                        
├── 📁 Code/                           # Stores all the code required to modify the dataset
│   ├── dataset                        # Standard dataset with synthetic albumentations degradation
│   └── dataset_double                 # Baseline dataset using official DIV2K Bicubic X4
│   └── dataset_official               # Decoupled dataset with Domain Alignment (Teacher Forcing)
├── DIV2K Repository                   # Contains the link to the official DIV2K Dataset Repository
│
├── 🧠 Models
├── 📁 Code/  
│   ├── baseline_model.py           # Baseline ESPCN Model (MSE Only)
│   ├── gan_model.py                # Single-Stage SRGAN + VGG19
│   ├── gan_model_resnet.py         # Single-Stage SRGAN + ResNet50 Perceptual Loss
│   ├── gan_model_attention.py      # Single-Stage SRGAN + Self-Attention
│   └── gan_model_double.py         # The Final Decoupled Pipeline (Stage 1 & Stage 2)
├── Models_Drive_Link               # Contains the Google Drive Link with all the trained models and respective logs    
│
├── 🚂 Train
│   ├── train.py                    # Training script for single-stage models
│   ├── train_double.py             # CLI Training script for the Two-Stage Pipeline
│
├── 🔎 Support_Functions
│   ├── check_data.py               # Visualizer for standard datasets
│   └── check_data_double.py        # Visualizer for the decoupled datasets
│   └── prep_stage2.py              # Offline inference script to prepare Stage 2 training data
│
└── 🖥️ Demo
    └── app_def.py                  # Interactive Streamlit Web App
```
