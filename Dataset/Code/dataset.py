import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Official PyTorch tool to download and extract the data
from torchvision.datasets.utils import download_and_extract_archive

DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=256, scale_factor=4):
        self.hr_dir = hr_dir
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"Folder not found: {hr_dir}")

        # Load all valid image filenames
        self.image_filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        # 1. GEOMETRIC TRANSFORMATIONS (Crop and rotation only, HR Target remains perfect)
        self.geom_transform = A.Compose([
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

        # 2. DEGRADATIONS (Applied only to the small LR image)
        # Multi-Stage Synthetic Degradation
        self.degrade_transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Simulate out-of-focus camera
            A.GaussNoise(std_range=(0.02, 0.08), p=0.5),  # Simulate sensor noise
            A.ImageCompression(quality_range=(40, 80), p=0.5)  # Simulate web compression artifacts
        ])

    def __len__(self):
        # Returns the total number of images available in the dataset
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1. Load the original High-Resolution image
        img_path = os.path.join(self.hr_dir, self.image_filenames[idx])
        hr_image = cv2.imread(img_path)

        # OpenCV loads images in BGR format by default, we convert it to RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # 2. Apply random geometric transformations (Extracts a random 256x256 patch)
        hr_patch = self.geom_transform(image=hr_image)['image']

        # 3. Create the Low-Resolution equivalent
        lr_size = (self.patch_size // self.scale_factor, self.patch_size // self.scale_factor)
        lr_base = cv2.resize(hr_patch, lr_size, interpolation=cv2.INTER_AREA)

        # 4. Apply the synthetic degradation pipeline to the LR image only
        lr_degraded = self.degrade_transform(image=lr_base)['image']

        # 5. Convert Numpy arrays to PyTorch Tensors and normalize pixels between 0 and 1
        input_tensor = torch.from_numpy(lr_degraded).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor


# === DATASET TESTING SCRIPT ===
if __name__ == "__main__":
    # Setup data root folder
    DATA_ROOT = "./data"
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Final extracted folder path
    hr_dir = os.path.join(DATA_ROOT, "DIV2K_train_HR")

    # If it's not downloaded yet, download and extract with a progress bar
    if not os.path.exists(hr_dir):
        print(f"📥 Downloading DIV2K dataset from {DIV2K_URL}...")
        print("☕ This might take a few minutes (file is ~700MB)...")
        download_and_extract_archive(DIV2K_URL, download_root=DATA_ROOT)
        print("✅ Download and extraction complete!")
    else:
        print("✅ Dataset already exists locally, skipping download.")

    # Instantiate the Dataset and DataLoader
    train_dataset = DIV2KDataset(hr_dir=hr_dir, patch_size=256, scale_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    print(f"\n✅ Dataloader is ready! Number of HD images found: {len(train_dataset)}")

    # Test loading the first batch
    lr_batch, hr_batch = next(iter(train_loader))
    print(f"📦 LR Batch Shape (Input X): {lr_batch.shape} --> (Batch, Channels, Height, Width)")
    print(f"📦 HR Batch Shape (Target Y): {hr_batch.shape} --> (Batch, Channels, Height, Width)")
