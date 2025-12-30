import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class RestorationUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        """
        n_channels = 2 (Input: 1 channel Blurry Image + 1 channel PSF Map)
        n_classes = 1 (Output: 1 channel Sharp Image)
        """
        super(RestorationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder (Downsampling) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck (The deepest part of the network)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # --- Decoder (Upsampling) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Final Output Layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid() # Ensure output pixels are in [0, 1] range
        )

    def forward(self, image, psf):
        """
        Args:
            image: [Batch, 1, H, W] - The blurry input
            psf:   [Batch, 1, H, W] - The estimated PSF map
        """
        # Concatenate Image and PSF along the channel dimension
        # Result shape: [Batch, 2, H, W]
        x = torch.cat([image, psf], dim=1)
        
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder Path with Skip Connections
        
        # Up 1
        x = self.up1(x5)
        # Handle padding issues if dimensions are not perfect powers of 2
        if x.size() != x4.size():
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x4, x], dim=1) # Skip connection
        x = self.conv1(x)
        
        # Up 2
        x = self.up2(x)
        if x.size() != x3.size():
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        # Up 3
        x = self.up3(x)
        if x.size() != x2.size():
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        # Up 4
        x = self.up4(x)
        if x.size() != x1.size():
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        # Output
        logits = self.outc(x)
        return logits