import torch
import torch.nn as nn
from torchvision import models
from core.config import ZERNIKE_NAMES, USE_PRETRAINED_WEIGHTS

class AberrationNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.num_zernikes = len(ZERNIKE_NAMES)
        output_dim = self.num_zernikes + 1
        if USE_PRETRAINED_WEIGHTS:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None            
        self.backbone = models.resnet18(weights=weights)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate), 
            nn.Linear(num_features, output_dim)
        )
        
    def forward(self, x):
        raw_output = self.backbone(x) 
        coeffs = raw_output[:, :self.num_zernikes]
        s_raw  = raw_output[:, self.num_zernikes:]
        s_factor = torch.nn.functional.softplus(s_raw) + 0.5
        return coeffs, s_factor