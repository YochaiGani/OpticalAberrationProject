import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift, ifftshift  
import torch.nn.functional as F
import numpy as np
from core.config import IMAGE_SIZE

class DifferentiableOpticalSimulator(nn.Module):
    def __init__(self, output_size=IMAGE_SIZE):
        super().__init__()
        self.H = output_size
        self.W = output_size
        self.R_max = np.sqrt((self.H // 2)**2 + (self.W // 2)**2) + 1e-12
        y_idx = torch.arange(-(self.H // 2), (self.H + 1) // 2)
        x_idx = torch.arange(-(self.W // 2), (self.W + 1) // 2)
        y, x = torch.meshgrid(y_idx, x_idx, indexing='ij') 
        self.register_buffer('grid_y', y / self.R_max)
        self.register_buffer('grid_x', x / self.R_max)
        self.register_buffer('r2', (x/self.R_max)**2 + (y/self.R_max)**2)
        self.register_buffer('zernike_maps', self._precompute_zernike_basis())

    def _precompute_zernike_basis(self):
        """ Exact math formulas (PyTorch version) """
        x, y, r2 = self.grid_x, self.grid_y, self.r2        
        z4  = torch.sqrt(torch.tensor(3.0)) * (2 * r2 - 1)
        z5  = 2 * torch.sqrt(torch.tensor(6.0)) * x * y
        z6  = torch.sqrt(torch.tensor(6.0)) * (x**2 - y**2)
        z7  = torch.sqrt(torch.tensor(8.0)) * (3 * r2 - 2) * y
        z8  = torch.sqrt(torch.tensor(8.0)) * (3 * r2 - 2) * x
        z12 = torch.sqrt(torch.tensor(5.0)) * (6 * r2**2 - 6 * r2 + 1)
        
        return torch.stack([z4, z5, z6, z7, z8, z12], dim=0)

    def _scale_psf(self, psf, s_factor):
        """ Scaling using Grid Sample (GPU / Differentiable) """
        batch_size = psf.shape[0]
        base_grid = torch.stack([self.grid_x, self.grid_y], dim=-1)
        grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid = grid / s_factor.view(batch_size, 1, 1, 1)
        psf_scaled = F.grid_sample(
            psf.unsqueeze(1), 
            grid, 
            align_corners=False, 
            mode='bilinear', 
            padding_mode='zeros'
        ).squeeze(1)
        
        return psf_scaled

    def get_psf(self, coeffs, s_factor):
        phase = torch.einsum('bc,chw->bhw', coeffs, self.zernike_maps)
        pupil = torch.exp(1j * 2 * np.pi * phase)
        pupil_fft = fftshift(fft2(pupil), dim=(-2, -1))
        iPSF = torch.abs(pupil_fft)**2
        iPSF = iPSF / (torch.sum(iPSF, dim=(1, 2), keepdim=True) + 1e-10)
        iPSF_final = self._scale_psf(iPSF, s_factor)
        iPSF_final = iPSF_final / (torch.sum(iPSF_final, dim=(1, 2), keepdim=True) + 1e-10)
        return iPSF_final

    def forward(self, clean_image, coeffs, s_factor):
        """ Physics Simulation (Deterministic & Differentiable) """
        psf = self.get_psf(coeffs, s_factor)
        pad_h, pad_w = self.H // 2, self.W // 2
        img_padded = F.pad(clean_image, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        psf_padded = F.pad(psf.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), value=0)
        fft_img = fft2(img_padded)
        fft_psf = fft2(ifftshift(psf_padded, dim=(-2, -1)))
        res_complex = ifft2(fft_img * fft_psf)
        res_real = torch.real(res_complex)
        output = res_real[:, :, pad_h:pad_h+self.H, pad_w:pad_w+self.W]
        return torch.clamp(output, 0, 1)