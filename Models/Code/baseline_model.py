import torch
from torch import nn
import pytorch_lightning as pl

# Import specific image metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class SuperResolutionModel(pl.LightningModule):
    def __init__(self, scale_factor=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves hyperparameters for deployment

        # --- NETWORK ARCHITECTURE (2D CNN) ---

        # 1. Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 2. Upsampler (The DLSS magic trick using sub-pixel convolutions)
        self.upsampler = nn.Sequential(
            nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)  # Technique to multiply spatial resolution
        )

        # --- LOSS AND METRICS ---
        self.loss_fn = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        # Data path: input x (Low Res) -> feature extraction -> upsampling -> output (High Res)
        features = self.feature_extractor(x)
        out = self.upsampler(features)
        return out

    def training_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        preds = self(lr_imgs)
        loss = self.loss_fn(preds, hr_imgs)

        # Log the training loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        preds = self(lr_imgs)

        # Calculate validation loss and metrics
        val_loss = self.loss_fn(preds, hr_imgs)
        val_psnr = self.psnr(preds, hr_imgs)
        val_ssim = self.ssim(preds, hr_imgs)

        # Log metrics to monitor progress
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_psnr", val_psnr, prog_bar=True)
        self.log("val_ssim", val_ssim, prog_bar=True)

    def configure_optimizers(self):
        # Adam optimizer configuration
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
