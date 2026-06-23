import os
import cv2
import torch
import albumentations as A
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from gan_model_double import Stage1DenoisingModel

# Stage 1 Checkpoint
CKPT_STAGE1 = "checkpoints/stage1/denoiser-epoch=epoch=13-val_psnr=val_psnr=27.54.ckpt"


def process_with_tiling(input_tensor, model, tile_size=64):
    """
    Breaks down a large tensor into smaller 'tiles',
    processes them one by one to avoid blowing up VRAM (Tiling),
    and stitches them back together.
    """
    _, _, h, w = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)

    # Scans the image like a grid
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            # Extracts the single tile
            tile = input_tensor[:, :, y:y_end, x:x_end]

            # Inference on the tile only
            with torch.no_grad():
                out_tile = model(tile)

            # Stitches the processed tile into the final tensor
            output_tensor[:, :, y:y_end, x:x_end] = out_tile

        return output_tensor


if __name__ == "__main__":
    print("🚀 Starting Offline Dataset generation for Stage 2...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = Stage1DenoisingModel.load_from_checkpoint(CKPT_STAGE1)
    model1.eval().to(device)

    hr_dir = "./data/DIV2K_train_HR"
    out_dir = "./data/DIV2K_train_LR_Stage1"
    os.makedirs(out_dir, exist_ok=True)

    stage1_degrade = A.Compose([
        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        A.ImageCompression(quality_range=(30, 70), p=1.0),
    ])

    transform = transforms.ToTensor()
    images = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))]

    # Process all images with a progress bar
    for img_name in tqdm(images, desc="Processing with Network 1"):
        hr_image = cv2.imread(os.path.join(hr_dir, img_name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        h, w, _ = hr_image.shape
        lr_size = (w // 4, h // 4)
        lr_base = cv2.resize(hr_image, lr_size, interpolation=cv2.INTER_AREA)

        lr_noisy = stage1_degrade(image=lr_base)['image']
        input_tensor = transform(lr_noisy).unsqueeze(0).to(device)

        # Use tiling to avoid OOM errors on 2K images
        out_tensor = process_with_tiling(input_tensor, model1, tile_size=64)

        out_np = out_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)

        out_image = Image.fromarray(out_np)
        out_image.save(os.path.join(out_dir, img_name))

    print("✅ Generation completed! You can now train Stage 2.")
