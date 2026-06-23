import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import resnet50, ResNet50_Weights


# ============================
# FEATURE EXTRACTOR
# ============================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Download the pre-trained ResNet50 on ImageNet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Extract the layers: we want to reach a deep layer to capture
        # complex patterns, but stop before the final classification head.
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3 
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

    def forward(self, x):
        # Standard ImageNet normalization (also applies to ResNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        return self.feature_extractor(x)


# ========================
# 1. THE GENERATOR
# ========================

# --- SELF-ATTENTION BLOCK ---
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 1x1 Convolutions to create Query, Key, and Value representations
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learnable scale parameter, initialized to 0 so the network learns
        # to apply attention gradually without disrupting initial training stability.
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Reshape to (Batch, Channels, N_pixels)
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # Matrix multiplication between Query and Key to get attention scores
        energy = torch.bmm(proj_query, proj_key)

        # Apply softmax to get attention probabilities
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        # Multiply Values by the attention map
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Add the attention map to the original input (Residual Connection)
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Skip Connection
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class Generator(nn.Module):
    def __init__(self, scale_factor=4, num_res_blocks=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        # Split Residual Blocks to insert Attention in the middle
        self.res_blocks_1 = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks // 2)])

        # Global context module
        self.attention = SelfAttentionBlock(in_dim=64)

        self.res_blocks_2 = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks // 2)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Upsampling
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

        # Passing through the first half of residual blocks
        out = self.res_blocks_1(out1)

        # Applying Self-Attention in the deep latent space
        out = self.attention(out)

        # Passing through the second half
        out = self.res_blocks_2(out)

        out = self.conv2(out)
        out = out + out1  # Global Skip Connection

        out = self.upsample(out)
        out = self.conv3(out)
        return (torch.tanh(out) + 1) / 2  # Output between 0 and 1


# ===========================
# 2. THE DISCRIMINATOR
# ===========================
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

            # Flatten to 1D for the classification head
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size, -1)


# =============================
# 3. THE LIGHTNING MODEL
# =============================
class SRGANModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.feature_extractor = ResNetFeatureExtractor()

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        opt_g, opt_d = self.optimizers()

        fake_imgs = self.generator(lr_imgs)

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

        # 1. Pixel Loss
        pixel_loss = self.mse_loss(fake_imgs, hr_imgs)

        # 2. ResNet Feature Loss
        real_features = self.feature_extractor(hr_imgs).detach()
        fake_features = self.feature_extractor(fake_imgs)
        feature_loss = self.mse_loss(fake_features, real_features) * 0.006

        # 3. Adversarial Loss
        adversarial_loss = self.bce_loss(self.discriminator(fake_imgs), valid)

        # Total Generator Loss
        g_loss = pixel_loss + feature_loss + (0.001 * adversarial_loss)

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
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        return [opt_g, opt_d], []
