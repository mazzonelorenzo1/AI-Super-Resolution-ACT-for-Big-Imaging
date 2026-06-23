import os
import torch
import argparse
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import the dataset and our two decoupled models
from dataset_double import DIV2KDataset
from gan_model_double import Stage1DenoisingModel, SRGANModel

def main():
    # ==========================================
    # CLI MENU (Choose which stage to train)
    # ==========================================
    parser = argparse.ArgumentParser(description="Training of the Two-Stage SRGAN Pipeline")
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help="Choose 1 for the Denoiser or 2 for the Upscaler (SRGAN with Attention)")
    args = parser.parse_args()

    print("\n" + "="*50)
    if args.stage == 1:
        print("🧹 STARTING STAGE 1: TRAINING 'THE DENOISER'")
        dataset_mode = 'stage1_denoise'
        ckpt_dir = 'checkpoints/stage1'
        ckpt_name = 'denoiser-epoch={epoch:02d}-val_psnr={val_psnr:.2f}'
    else:
        print("🎨 STARTING STAGE 2: TRAINING 'THE ARTIST' (Super Resolution with Attention)")
        dataset_mode = 'stage2_upscale'
        ckpt_dir = 'checkpoints/stage2'
        ckpt_name = 'srgan-attention-epoch={epoch:02d}-val_psnr={val_psnr:.2f}'
    print("="*50 + "\n")

    # ==========================================
    # 1. DATA PREPARATION
    # ==========================================
    print(f"⚙️ Data preparation in progress (Mode: {dataset_mode})...")
    DATA_ROOT = "./data/DIV2K_train_HR"

    # Load the dataset with the correct mode for the chosen stage
    full_dataset = DIV2KDataset(hr_dir=DATA_ROOT, patch_size=256, scale_factor=4, mode=dataset_mode)

    train_size = 750
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📊 Training Set Size: {len(train_dataset)} images")
    print(f"📊 Validation Set Size: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # ==========================================
    # 2. MODEL INITIALIZATION
    # ==========================================
    if args.stage == 1:
        model = Stage1DenoisingModel(lr=2e-4)
    else:
        model = SRGANModel(lr=1e-4)

    # ==========================================
    # 3. CHECKPOINTING (Saving Weights)
    # ==========================================
    # We separate the folders so we don't mix Stage 1 weights with Stage 2 weights
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        mode='max',
        dirpath=ckpt_dir,
        filename=ckpt_name,
        save_top_k=1,
        verbose=True
    )

    # ==========================================
    # 4. TRAINER CONFIGURATION
    # ==========================================
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed', # Mixed precision to reduce VRAM usage
        log_every_n_steps=10
    )

    # ==========================================
    # 5. START TRAINING
    # ==========================================
    print(f"\n🚀 Starting training on the GPU for Stage {args.stage}...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
