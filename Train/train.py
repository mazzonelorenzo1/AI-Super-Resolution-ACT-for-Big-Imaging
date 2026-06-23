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

    # 1. Data Splitting
    DATA_ROOT = "./data_official/DIV2K_train_HR"

    # Load the entire dataset
    full_dataset = DIV2KOfficialDataset(data_dir="./data_official", patch_size=256, scale_factor=4)

    # Split the 800 images: 750 for Training, 50 for Validation
    train_size = 750
    val_size = len(full_dataset) - train_size

    # Use a manual seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📊 Training Set Size: {len(train_dataset)} images")
    print(f"📊 Validation Set Size: {len(val_dataset)} images")

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. Model Initialization
    model = SRGANModel(lr=1e-4)

    # 3. Checkpoint Callback
    # PyTorch Lightning will monitor the validation loss/metric and automatically save
    # only the weights file (.ckpt) of the epoch with the highest quality.
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',  
        mode='max',         
        dirpath='checkpoints/',
        filename='best-official-attention-VGG-gan-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=1,
        verbose=True
    )

    # 4. Trainer Configuration
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Mixed precision to save VRAM on the GPU
        log_every_n_steps=10
    )

    # 5. Start training
    print("\n🚀 Starting training on the GPU...")
    trainer.fit(model, train_loader, val_loader)
