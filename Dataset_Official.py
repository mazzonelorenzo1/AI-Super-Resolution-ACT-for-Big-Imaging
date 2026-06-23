import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive

# Official DIV2K links (HR and LR Bicubic X4)
HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
LR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"


class DIV2KOfficialDataset(Dataset):
    def __init__(self, data_dir="./data_official", patch_size=256, scale_factor=4):
        self.hr_dir = os.path.join(data_dir, "DIV2K_train_HR")
        # Pay attention to the internal path of the official zip structure:
        self.lr_dir = os.path.join(data_dir, "DIV2K_train_LR_bicubic", "X4")
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        # 1. Automatic download of BOTH packages if missing
        if not os.path.exists(self.hr_dir):
            print(f"📥 Downloading HR dataset...")
            download_and_extract_archive(HR_URL, download_root=data_dir)

        if not os.path.exists(self.lr_dir):
            print(f"📥 Downloading Official LR dataset (Bicubic x4)...")
            download_and_extract_archive(LR_URL, download_root=data_dir)

        # Retrieve original HR filenames
        self.hr_filenames = sorted([f for f in os.listdir(self.hr_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        # 2. Official Naming Convention Handling (e.g., 0001.png -> 0001x4.png)
        hr_name = self.hr_filenames[idx]
        name, ext = os.path.splitext(hr_name)
        lr_name = f"{name}x4{ext}"

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(self.lr_dir, lr_name)

        # Load images
        hr_image = cv2.imread(hr_path)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        lr_image = cv2.imread(lr_path)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # 3. SYNCHRONIZED GEOMETRIC CROP (Crucial for alignment!)
        lr_h, lr_w, _ = lr_image.shape
        lr_crop_size = self.patch_size // self.scale_factor  # Usually 64

        # Random anchor point on the small grid
        x_lr = random.randint(0, max(0, lr_w - lr_crop_size))
        y_lr = random.randint(0, max(0, lr_h - lr_crop_size))

        # 64x64 Crop from the LR image
        input_img = lr_image[y_lr:y_lr + lr_crop_size, x_lr:x_lr + lr_crop_size]

        # Multiply coordinates by 4 to get the corresponding 256x256 crop on the HR image
        x_hr = x_lr * self.scale_factor
        y_hr = y_lr * self.scale_factor
        target_img = hr_image[y_hr:y_hr + self.patch_size, x_hr:x_hr + self.patch_size]

        # 4. MANUAL DATA AUGMENTATION (Keeps crops perfectly aligned)
        if random.random() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()
        if random.random() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()

        # Normalize to [0, 1] PyTorch Tensors
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor


# QUICK TEST
if __name__ == "__main__":
    print("Testing official dataset loading...")
    dataset = DIV2KOfficialDataset()
    in_t, tg_t = dataset[0]
    print(f"✅ Success! Official LR Input: {in_t.shape}, HR Target: {tg_t.shape}")
