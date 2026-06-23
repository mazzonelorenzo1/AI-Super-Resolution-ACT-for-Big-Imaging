import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Import dataset
from dataset import DIV2KDataset


def show_batch_with_names(dataset, num_images=4):
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 16))
    fig.suptitle("Data Check with Original Filenames", fontsize=16)

    # Pick 4 random indices from the dataset
    random_indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(random_indices):
        # Manually extract the tensors and the FILENAME
        lr_tensor, hr_tensor = dataset[idx]
        file_name = dataset.image_filenames[idx]  # <-- HERE IS THE FILENAME!

        # Prepare images for Matplotlib
        lr = lr_tensor.permute(1, 2, 0).numpy()
        hr = hr_tensor.permute(1, 2, 0).numpy()
        lr = np.clip(lr, 0, 1)
        hr = np.clip(hr, 0, 1)

        # Left Column: Input
        axes[i, 0].imshow(lr)
        axes[i, 0].set_title(f"Network Input (LR)\nFile: {file_name}")
        axes[i, 0].axis("off")

        # Right Column: Target
        axes[i, 1].imshow(hr)
        axes[i, 1].set_title(f"Exact Target (HR)\nFile: {file_name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading dataset...")
    DATA_ROOT = "./data/DIV2K_train_HR"
    
    if not os.path.exists(DATA_ROOT):
         print(f"Error: Folder not found ({DATA_ROOT}). Make sure you have downloaded the data.")
    else:
         dataset = DIV2KDataset(hr_dir=DATA_ROOT, patch_size=256, scale_factor=4)
         print(f"Dataset loaded. Total images: {len(dataset)}")
         print("Generating plot... (close the window to exit)")
         show_batch_with_names(dataset, num_images=4)
