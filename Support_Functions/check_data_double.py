import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch

# Import the two-stage dataset
from dataset_double import DIV2KDataset


def show_batch_with_names(dataset, num_images=3):
    # Determine the titles based on the dataset mode!
    if dataset.mode == 'stage1_denoise':
        main_title = "Data Check: STAGE 1 (Denoiser - Denoising Only)"
        input_title = "Network Input (Noisy LR)"
        target_title = "Exact Target (Clean LR)"
    else:
        main_title = "Data Check: STAGE 2 (Artist - Super Resolution)"
        input_title = "Network Input (Clean but SMOOTH LR)"
        target_title = "Exact Target (Sharp 4K HR)"

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 12))
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # Pick random indices from the dataset
    random_indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(random_indices):
        # Extract the tensors and the filename
        input_tensor, target_tensor = dataset[idx]
        file_name = dataset.image_filenames[idx]

        # Prepare images for Matplotlib
        input_np = input_tensor.permute(1, 2, 0).numpy()
        target_np = target_tensor.permute(1, 2, 0).numpy()
        
        # Clip values to be strictly between 0 and 1 for correct plotting
        input_np = np.clip(input_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)

        # Left Column: Input
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title(f"{input_title}\nFile: {file_name}")
        axes[i, 0].axis("off")

        # Right Column: Target
        axes[i, 1].imshow(target_np)
        axes[i, 1].set_title(f"{target_title}\nFile: {file_name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ==========================================
    # PYTORCH HARDWARE CHECK
    # ==========================================
    print("\n" + "=" * 40)
    print("🖥️  PYTORCH HARDWARE CHECK")
    print("=" * 40)
    gpu_available = torch.cuda.is_available()
    print(f"CUDA Availability: {gpu_available}")
    if gpu_available:
        print(f"Detected GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    else:
        print("⚠️ NO GPU DETECTED! PyTorch will use the CPU (Very Slow).")
    print("=" * 40 + "\n")

    DATA_ROOT = "./data"
    hr_dir = os.path.join(DATA_ROOT, "DIV2K_train_HR")

    if not os.path.exists(hr_dir):
        print("⚠️ Dataset folder not found. Run dataset.py first to download the data.")
        exit()

    # --- STAGE 1 TEST ---
    print("Loading STAGE 1 pipeline (Denoising)...")
    dataset_stage1 = DIV2KDataset(hr_dir=hr_dir, patch_size=256, scale_factor=4, mode='stage1_denoise')
    print("Displaying window for Stage 1. (Close the window to proceed to Stage 2)")
    show_batch_with_names(dataset_stage1, num_images=3)

    # --- STAGE 2 TEST ---
    print("\nLoading STAGE 2 pipeline (Super Resolution)...")
    try:
        dataset_stage2 = DIV2KDataset(hr_dir=hr_dir, patch_size=256, scale_factor=4, mode='stage2_upscale')
        print("Displaying window for Stage 2. (Close the window to exit)")
        show_batch_with_names(dataset_stage2, num_images=3)
    except FileNotFoundError as e:
        print(f"\n❌ Error during Stage 2 initialization: {e}")
