import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import the classes we created in the previous files
from dataset_official import DIV2KOfficialDataset
from gan_model_attention import SRGANModel

if __name__ == "__main__":
    print("⚙️ Data preparation in progress...")

    # 1. DATA SPLITTING (As required for the exam evaluation)
    DATA_ROOT = "./data_official/DIV2K_train_HR"

    # Load the entire dataset
    full_dataset = DIV2KOfficialDataset(data_dir="./data_official", patch_size=256, scale_factor=4)

    # Split the 800 images: 750 for Training, 50 for Validation
    train_size = 750
    val_size = len(full_dataset) - train_size

    # Use a manual seed for reproducibility (good data science practice!)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📊 Training Set Size: {len(train_dataset)} images")
    print(f"📊 Validation Set Size: {len(val_dataset)} images")

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. MODEL INITIALIZATION (Our Custom SRGAN)
    model = SRGANModel(lr=1e-4)

    # 3. CHECKPOINT CALLBACK (Deployment Step)
    # PyTorch Lightning will monitor the validation loss/metric and automatically save
    # ONLY the weights file (.ckpt) of the epoch with the highest quality.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',  # <--- We monitor the image quality metric!
        mode='max',          # <--- We want the PSNR to be as HIGH as possible
        dirpath='checkpoints/',
        filename='best-official-attention-VGG-gan-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=1,
        verbose=True
    )

    # 4. TRAINER CONFIGURATION
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Mixed precision to save VRAM on the GPU
        log_every_n_steps=10
    )

    # 5. START TRAINING
    print("\n🚀 Starting training on the GPU...")
    trainer.fit(model, train_loader, val_loader)
