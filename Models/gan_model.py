import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import vgg19, VGG19_Weights


# ==========================================
# FEATURE EXTRACTOR (The Art Critic - VGG19)
# ==========================================
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Download the pre-trained VGG19 network
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # We extract only the first 36 layers (up to ReLU 5.4, as specified in the SRGAN paper)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36])

        # IMPORTANT: Freeze the weights! We don't want to train the VGG;
        # we only need it to evaluate the perceptual quality. This saves a lot of VRAM.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

    def forward(self, x):
        # VGG expects images to be normalized according to the ImageNet standard
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        return self.feature_extractor(x)


# ==========================================
# 1. THE GENERATOR (The Forger)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Skip Connection: adds the original input to the residual output
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual  # <- THE SECRET OF DEEP NETWORKS


class Generator(nn.Module):
    def __init__(self, scale_factor=4, num_res_blocks=8):  # 8 blocks = deep network
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        # Create a sequential chain of 8 Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1  # Global Skip Connection
        out = self.upsample(out)
        out = self.conv3(out)
        return (torch.tanh(out) + 1) / 2  # Normalize output between 0 and 1


# ==========================================
# 2. THE DISCRIMINATOR (The Cop)
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten the spatial dimensions for the final decision
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size, -1)


# ==========================================
# 3. THE LIGHTNING MODEL (The GAN Arena)
# ==========================================
class SRGANModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # IMPORTANT: With GANs we must manage optimization manually
        self.automatic_optimization = False

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.vgg = VGGFeatureExtractor()

        # Loss functions
        self.mse_loss = nn.MSELoss()  # For basic colors/shapes (Pixel Loss)
        self.bce_loss = nn.BCEWithLogitsLoss()  # For adversarial training (Real vs Fake)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        opt_g, opt_d = self.optimizers()

        # Generate fake images
        fake_imgs = self.generator(lr_imgs)

        # Create labels (1 for real, 0 for fake)
        valid = torch.ones((lr_imgs.size(0), 1), device=self.device)
        fake = torch.zeros((lr_imgs.size(0), 1), device=self.device)

        # ---------------------
        # TRAIN DISCRIMINATOR
        # ---------------------
        self.toggle_optimizer(opt_d)

        real_loss = self.bce_loss(self.discriminator(hr_imgs), valid)
        fake_loss = self.bce_loss(self.discriminator(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        self.manual_backward(d_loss)
        self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ---------------------
        # TRAIN GENERATOR
        # ---------------------
        self.toggle_optimizer(opt_g)

        # 1. Pixel Loss (Anchors colors and structural layout)
        pixel_loss = self.mse_loss(fake_imgs, hr_imgs)

        # 2. VGG Feature Loss (The Art Critic: multiplied by 0.006 to scale it to pixel levels)
        real_features = self.vgg(hr_imgs).detach()
        fake_features = self.vgg(fake_imgs)
        vgg_loss = self.mse_loss(fake_features, real_features) * 0.006

        # 3. Adversarial Loss (Fools the cop to generate micro-details)
        adversarial_loss = self.bce_loss(self.discriminator(fake_imgs), valid)

        # Combine all losses: Colors + Scaled Textures + Cop
        g_loss = pixel_loss + vgg_loss + (0.001 * adversarial_loss)

        self.manual_backward(g_loss)
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Logging
        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        fake_imgs = self.generator(lr_imgs)

        val_mse = self.mse_loss(fake_imgs, hr_imgs)
        val_psnr = self.psnr(fake_imgs, hr_imgs)
        val_ssim = self.ssim(fake_imgs, hr_imgs)

        self.log("val_loss", val_mse, prog_bar=True)
        self.log("val_psnr", val_psnr, prog_bar=True)
        self.log("val_ssim", val_ssim, prog_bar=True)

    def configure_optimizers(self):
        # Optimizers for Generator and Discriminator
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        return [opt_g, opt_d], []
