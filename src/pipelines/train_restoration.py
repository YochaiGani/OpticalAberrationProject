import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import Project Modules
from networks.estimator import DifferentiableOpticalSimulator
from network.restoration_network import RestorationUNet
from pipelines.train_estimator import AberrationDataset # Reuse the robust dataset loader
from core.config import IMAGE_SIZE

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Restoration Training on: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 8           # UNet is memory intensive
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40

# Loss Weights
# L1: Enforces pixel-wise accuracy (Sharpness baseline)
L1_WEIGHT = 1.0        
# VGG: Enforces perceptual quality (Texture/Structure)
PERCEPTUAL_WEIGHT = 0.1 

# Paths
DATA_DIR = "../data"
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
IMG_DIR_ABERRATED = os.path.join(DATA_DIR, "aberrated")
IMG_DIR_CLEAN = os.path.join(DATA_DIR, "clean")
MODEL_SAVE_PATH = "../models/restoration_unet.pth"
PLOT_PATH = "../restoration_training_log.png"

# ==========================================
# 2. VGG PERCEPTUAL LOSS
# ==========================================
class VGGLoss(nn.Module):
    """Calculates perceptual distance using VGG19 features."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:35]).eval()
        for p in self.features.parameters(): p.requires_grad = False
        self.criterion = nn.MSELoss()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # Repeat 1-channel grayscale to 3-channel RGB for VGG
        x_rgb = x.repeat(1, 3, 1, 1)
        y_rgb = y.repeat(1, 3, 1, 1)
        
        x_norm = (x_rgb - self.mean) / self.std
        y_norm = (y_rgb - self.mean) / self.std
        
        x_feat = self.features(x_norm)
        y_feat = self.features(y_norm)
        return self.criterion(x_feat, y_feat)

# ==========================================
# 3. TRAINER CLASS
# ==========================================
class RestorationTrainer:
    def __init__(self):
        # 1. Initialize Networks
        # UNet takes 2 channels: [Blurry_Image, PSF_Map]
        self.unet = RestorationUNet(n_channels=2, n_classes=1).to(DEVICE)
        
        # Simulator (Used purely as a math engine to generate PSF maps)
        self.simulator = DifferentiableOpticalSimulator(output_size=IMAGE_SIZE).to(DEVICE)
        
        # 2. Optimization
        self.optimizer = optim.Adam(self.unet.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4)
        
        # 3. Loss Functions
        self.l1_loss = nn.L1Loss() 
        self.vgg_loss = VGGLoss().to(DEVICE)
        
        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # History
        self.history = {'train': [], 'val': []}
        
        self._load_data()

    def _load_data(self):
        print("[INFO] Loading Dataset...")
        dataset = AberrationDataset(LABELS_PATH, IMG_DIR_ABERRATED, IMG_DIR_CLEAN)
        
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
        
        # num_workers=0 for Windows safety
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"[INFO] Ready. Train: {train_len}, Val: {val_len}")

    def generate_psf_map(self, coeffs, scale):
        """
        Helper: Generates the 2D PSF image from coefficients using the differentiable simulator logic.
        This map serves as the 'Guide' for the UNet.
        """
        batch_size = coeffs.shape[0]
        
        # 1. Calculate Phase & Pupil
        phase = self.simulator.get_phase(coeffs)
        pupil = torch.exp(1j * 2 * np.pi * phase)
        
        # 2. Compute Raw PSF
        iPSF = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil)))**2
        iPSF = iPSF / (torch.sum(iPSF, dim=(1, 2), keepdim=True) + 1e-10)
        
        # 3. Apply Scale (Zoom) via Grid Sample
        base_grid = torch.stack([self.simulator.grid_x, self.simulator.grid_y], dim=-1)
        grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid = grid / scale.view(batch_size, 1, 1, 1)
        
        # Resample
        iPSF_scaled = torch.nn.functional.grid_sample(
            iPSF.unsqueeze(1), grid, align_corners=False, mode='bilinear'
        )
        
        # Normalize Min-Max for better network stability
        # We want the network to see the shape, not necessarily the exact energy density
        flat = iPSF_scaled.view(batch_size, -1)
        max_val = torch.max(flat, dim=1)[0].view(batch_size, 1, 1, 1)
        iPSF_norm = iPSF_scaled / (max_val + 1e-8)
        
        return iPSF_norm

    def train_epoch(self):
        self.unet.train()
        loop = tqdm(self.train_loader, desc="Training UNet", leave=False)
        epoch_loss = 0
        
        for img_blur, img_clean, coeffs, scale in loop:
            img_blur = img_blur.to(DEVICE)
            img_clean = img_clean.to(DEVICE)
            coeffs = coeffs.to(DEVICE)
            scale = scale.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            # Use Mixed Precision
            with torch.cuda.amp.autocast():
                # A. Prepare Inputs
                # Generate the specific PSF map for this image
                with torch.no_grad():
                    psf_map = self.generate_psf_map(coeffs, scale)
                
                # B. Forward Pass
                # The UNet takes the blurry image AND the PSF map
                img_restored = self.unet(img_blur, psf_map)
                
                # C. Loss Calculation
                loss_pixel = self.l1_loss(img_restored, img_clean)
                loss_perceptual = self.vgg_loss(img_restored, img_clean)
                
                total_loss = L1_WEIGHT * loss_pixel + PERCEPTUAL_WEIGHT * loss_perceptual
            
            # D. Backward
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.unet.eval()
        val_loss = 0
        with torch.no_grad():
            for img_blur, img_clean, coeffs, scale in self.val_loader:
                img_blur = img_blur.to(DEVICE)
                img_clean = img_clean.to(DEVICE)
                coeffs = coeffs.to(DEVICE)
                scale = scale.to(DEVICE)
                
                psf_map = self.generate_psf_map(coeffs, scale)
                img_restored = self.unet(img_blur, psf_map)
                
                loss_pixel = self.l1_loss(img_restored, img_clean)
                loss_perceptual = self.vgg_loss(img_restored, img_clean)
                
                total_loss = L1_WEIGHT * loss_pixel + PERCEPTUAL_WEIGHT * loss_perceptual
                val_loss += total_loss.item()
                
        return val_loss / len(self.val_loader)

    def run(self):
        print(f"\n[INFO] Starting Restoration Training for {NUM_EPOCHS} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.unet.state_dict(), MODEL_SAVE_PATH)
                print(f">>> New Best Model Saved! Loss: {best_val_loss:.5f}")
        
        self.plot_results()
        print("[INFO] Training Complete.")

    def plot_results(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.history['train'], label='Train')
        plt.plot(self.history['val'], label='Val')
        plt.title('Restoration Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (L1 + VGG)')
        plt.legend()
        plt.savefig(PLOT_PATH)

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)
    
    trainer = RestorationTrainer()
    trainer.run()