import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# THE NEURAL NETWORK (The Brain)
# ==========================================
class AberrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 7)
        
    def forward(self, x):
        raw_output = self.backbone(x) 
        coeffs = raw_output[:, :6]
        s_raw  = raw_output[:, 6:7]
        s_factor = torch.nn.functional.softplus(s_raw) + 0.5
        
        return coeffs, s_factor