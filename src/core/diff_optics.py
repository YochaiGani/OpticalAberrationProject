import torch
import torch.nn as nn
import numpy as np
from core.config import IMAGE_SIZE

class DifferentiableOpticalSimulator(nn.Module):
    def __init__(self, output_size=IMAGE_SIZE):
        super().__init__()
        self.output_size = output_size
        H, W = output_size, output_size
        R_max = np.sqrt((H // 2)**2 + (W // 2)**2) + 1e-12
        y, x = torch.meshgrid(torch.arange(-(H//2), (H+1)//2), torch.arange(-(W//2), (W+1)//2), indexing='ij')
        
        self.register_buffer('grid_y', y / R_max)
        self.register_buffer('grid_x', x / R_max)
        self.register_buffer('r2', (x/R_max)**2 + (y/R_max)**2)
        self.register_buffer('zernike_basis', self._precompute_zernike_basis())
        # self.zernike_basis = self._precompute_zernike_basis()

    def _precompute_zernike_basis(self):
        x, y, r2 = self.grid_x, self.grid_y, self.r2
        z4  = torch.sqrt(torch.tensor(3.0)) * (2 * r2 - 1)
        z5  = 2 * torch.sqrt(torch.tensor(6.0)) * x * y
        z6  = torch.sqrt(torch.tensor(6.0)) * (x**2 - y**2)
        z7  = torch.sqrt(torch.tensor(8.0)) * (3 * r2 - 2) * y
        z8  = torch.sqrt(torch.tensor(8.0)) * (3 * r2 - 2) * x
        z12 = torch.sqrt(torch.tensor(5.0)) * (6 * r2**2 - 6 * r2 + 1)
        return torch.stack([z4, z5, z6, z7, z8, z12], dim=0)

    def get_phase(self, coeffs):
        return torch.einsum('bc,chw->bhw', coeffs, self.zernike_basis)

    def forward(self, clean_image, coeffs, s_factor):
        batch_size = clean_image.shape[0]
        phase = self.get_phase(coeffs)
        pupil = torch.exp(1j * 2 * np.pi * phase)
        iPSF = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil)))**2
        iPSF = iPSF / (torch.sum(iPSF, dim=(1, 2), keepdim=True) + 1e-10)
        
        base_grid = torch.stack([self.grid_x, self.grid_y], dim=-1)
        grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid = grid / s_factor.view(batch_size, 1, 1, 1)
        
        iPSF_scaled = torch.nn.functional.grid_sample(iPSF.unsqueeze(1), grid, align_corners=False, mode='bilinear').squeeze(1)
        iPSF_scaled = iPSF_scaled / (torch.sum(iPSF_scaled, dim=(1, 2), keepdim=True) + 1e-10)

        pad_h, pad_w = self.output_size // 2, self.output_size // 2
        img_padded = torch.nn.functional.pad(clean_image, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        psf_padded = torch.nn.functional.pad(iPSF_scaled.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), value=0)

        res_real = torch.real(torch.fft.ifft2(torch.fft.fft2(img_padded) * torch.fft.fft2(torch.fft.fftshift(psf_padded))))
        return torch.clamp(res_real[:, :, pad_h:pad_h+self.output_size, pad_w:pad_w+self.output_size], 0, 1)