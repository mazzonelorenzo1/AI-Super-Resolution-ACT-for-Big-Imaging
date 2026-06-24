#  🔎 AI Super Resolution - ACT for Big Imaging Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 📑 Table of Contents
- [About the Project](#intro)
- [Repository Structure](#struct)
- [Installation](#install)
- [How to Run the Demo](#demo)
- [Quantitative Results](#results)
- [Aknowledgments](#aknow)

---

<a id="intro"></a>
## 📖 About the Project
This project tackles the Image Super-Resolution and Restoration problem in real-world scenarios. The goal is to reconstruct a High-Resolution image starting from a heavily degraded Low-Resolution image, affected by sensor noise, blur, and JPEG compression artifacts.

While classic Generative Adversarial Networks (GANs) struggle with simultaneous denoising and upscaling (often resulting in gradient collapse, checkerboard artifacts, or severe color shifts), this project proposes a Decoupled Restoration Pipeline:

1. **Stage 1 (The Denoiser):** A ResNet-based CNN optimized purely on Mean Squared Error (MSE) to clean the image without hallucinating details.

2. **Stage 2 (The Enhancer):** An SRGAN powered by Self-Attention blocks and a ResNet50 Perceptual Loss, trained explicitly to inject high-frequency textures into the clean output of Stage 1.

The project also features a custom solution for the Domain Gap problem using Teacher Forcing and pixel-perfect geometric alignment.

For in-depth analysis of this project please refer to the [presentation](https://github.com/mazzonelorenzo1/AI-Super-Resolution-ACT-for-Big-Imaging/blob/main/Canva_Presentation).

---

<a id="struct"></a>
## 📂 Repository Structure
The repository is organized modularly, separating data loading, model architectures, and training logic.

```text
📁 AI-Super-Resolution-ACT-for-Big-Imaging/
│
├── 📁 Dataset/                        
├── 📁 Code/                        
│   ├── dataset                     # Standard dataset with synthetic albumentations degradation
│   └── dataset_double              # Baseline dataset using official DIV2K Bicubic X4
│   └── dataset_official            # Decoupled dataset with Domain Alignment
├── DIV2K Repository                # Contains the link to the official DIV2K Dataset Repository
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

---

<a id="install"></a>
## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/mazzonelorenzo1/AI-Super-Resolution-ACT-for-Big-Imaging.git](https://github.com/mazzonelorenzo1/AI-Super-Resolution-ACT-for-Big-Imaging.git)
cd AI-Super-Resolution-ACT-for-Big-Imaging
```

2. **Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install the required dependencies:**
Ensure you have PyTorch installed with CUDA support if you intend to train the models.
```bash
pip install -r requirements.txt
```

---

<a id="demo"></a>
## 🚀 How to Run the Demo
1. **Data Preparation & Checking**
The scripts will automatically download the DIV2K dataset from the official ETH Zurich servers on the first run. To verify that the data augmentation and geometric cropping are working perfectly, run:
```bash
python check_data_double.py
```

2. **Training the Decoupled Pipeline**
The training is split into two phases.

**Step A: Train Stage 1 (The Denoiser)**
```bash
python train_double.py --stage 1
```

Wait for the training to finish. The best weights will be saved in checkpoints/stage1/.

**Step B: Prepare Data for Stage 2 (Domain Alignment)**
Before training the GAN, we must process all HR images through the flawed Stage 1 model to simulate the production environment.
Open *prep_stage2.py*, ensure *CKPT_STAGE1 points* to your newly trained Stage 1 weights, and run:
```bash
python prep_stage2.py
```

**Step C: Train Stage 2 (The Enhancer)**
Once the intermediate images are generated, you can train the GAN:
```bash
python train_double.py --stage 2
```

**Shortcut**:
Alternatively, you can already find all the trained models in [this Google Drive](https://drive.google.com/drive/folders/1jcDUcz_Un1R-vTTecjsqVsS8oow4UezQ?usp=drive_link). Download all the necessary models and put them in the correct path.

3. **Running the Streamlit UI**
For clarity and browsing convenience on GitHub, this repository organizes files into logical subfolders (`Models/`, `Datasets/`, `User Interface/`, etc.). However, to successfully run the Streamlit demo locally without Python `ImportError` issues, `app.py` needs to be in the same root directory as the model files it imports.

Before running the app, please rearrange your local working directory to match this flattened structure:

```text
📁 Working_Directory/
│
├── 📁 checkpoints/                 # Contains all the trained .ckpt files
│   ├── stage1/                     
│   └── stage2/
│   ├── #All the other models                
│
├── app.py                          # Moved from User Interface/
├── model.py                        # Moved from Models/
├── gan_model.py                    # Moved from Models/
├── gan_model_resnet.py             # Moved from Models/
├── gan_model_attention.py          # Moved from Models/
├── gan_model_double.py             # Moved from Models/
└── #And so on with ALL the rest of the scripts
```

To then test the models, compare different historical architectures, and upscale your own images or videos, launch the interactive Streamlit Application:

- Open app.py and ensure the paths inside the CKPT_PATHS dictionary point to your saved .ckpt files. You can find all the necessary models in [this Google Drive](https://drive.google.com/drive/folders/1jcDUcz_Un1R-vTTecjsqVsS8oow4UezQ?usp=drive_link).

- Run the app:
```bash
streamlit run app.py
```

- A browser window will open automatically. Upload an image, select your desired AI engine from the sidebar, and watch the inference happen with dynamic VRAM-safe tiling!

---

<a id="results"></a>
## 📈 Quantitative Results

Below is a summary of the evaluation metrics across the different architectures tested during the project's evolution.

| **Model Type** | **Loss Value** | **PSNR Value (dB)** | **Similarity Value (SSIM)** |
| :--- | :---: | :---: | :---: |
| Baseline ESPCN | 0.0044 | 23.73 | 0.74 |
| VGG Model | 0.0065 | 22.32 | 0.57 |
| VGG Model + ResNet | 0.0109 | 20.39 | 0.53 |
| VGG Model + Attention | 0.0095 | 20.73 | 0.54 |
| Decoupled VGG Model - Stage 1 | **0.0019** | **27.54** | **0.83** |
| Decoupled VGG Model - Stage 2 | 0.0059 | 22.71 | 0.59 |
| VGG Model + Attention (Official Dataset) | 0.0040 | 25.01 | 0.70 |

---

<a id="aknow"></a>
## 🏆 Acknowledgments

1. The DIV2K dataset used for training and validation is provided by the Computer Vision Laboratory, ETH Zurich.
2. State-of-the-art comparisons in the app are powered by HuggingFace Transformers (SwinIR).
