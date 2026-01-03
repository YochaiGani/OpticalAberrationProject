import numpy as np
import pyfftw
import cv2
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10.0)

from core.config import ZERNIKE_STDS, SCALE_RANGE, TARGET_SNR, IMAGE_SIZE


class OpticalSimulator:
    def __init__(self, output_size=IMAGE_SIZE):
        self.H = output_size
        self.W = output_size
        
        # Grid Setup
        self.R_max = np.sqrt((self.H // 2)**2 + (self.W // 2)**2) + 1e-12
        y_idx = np.arange(-(self.H // 2), (self.H + 1) // 2)
        x_idx = np.arange(-(self.W // 2), (self.W + 1) // 2)
        self.grid_y, self.grid_x = np.meshgrid(y_idx, x_idx, indexing='ij')
        
        # Normalize grids
        self.grid_y = self.grid_y / self.R_max
        self.grid_x = self.grid_x / self.R_max
        self.r2 = self.grid_x**2 + self.grid_y**2
        
        # Precompute Zernike Basis
        self.zernike_maps = self._precompute_zernike_basis()

    def _precompute_zernike_basis(self):
        """ Exact math formulas """
        x, y, r2 = self.grid_x, self.grid_y, self.r2
        z4  = np.sqrt(3.0) * (2 * r2 - 1)
        z5  = 2 * np.sqrt(6.0) * x * y
        z6  = np.sqrt(6.0) * (x**2 - y**2)
        z7  = np.sqrt(8.0) * (3 * r2 - 2) * y
        z8  = np.sqrt(8.0) * (3 * r2 - 2) * x
        z12 = np.sqrt(5.0) * (6 * r2**2 - 6 * r2 + 1)

        return np.stack([z4, z5, z6, z7, z8, z12], axis=0)

    def _scale_psf(self, psf, s_factor):
        """ Helper for scaling PSF using CPU-friendly cv2 """
        if s_factor == 1.0: return psf
        new_size = (int(self.W * s_factor), int(self.H * s_factor))
        psf_scaled = cv2.resize(psf.astype(np.float32), new_size, interpolation=cv2.INTER_LINEAR)
        return self._crop_pad_numpy(psf_scaled, (self.H, self.W))

    def get_psf(self, coeffs, s_factor):
        """ Math core - Same as PyTorch """
        phase = np.tensordot(coeffs, self.zernike_maps, axes=1)
        pupil = np.exp(1j * 2 * np.pi * phase)
        pupil_fft = fftshift(fft2(pupil))
        iPSF = np.abs(pupil_fft)**2
        iPSF /= (np.sum(iPSF) + 1e-10)
        iPSF_final = self._scale_psf(iPSF, s_factor)
        iPSF_final /= (np.sum(iPSF_final) + 1e-10)
        return iPSF_final
    
    def forward(self, clean_image, coeffs, s_factor):
        """ Physics core (Deterministic) - Same as PyTorch """
        psf = self.get_psf(coeffs, s_factor)
        pad_h, pad_w = self.H // 2, self.W // 2
        img_padded = np.pad(clean_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        psf_padded = np.pad(psf, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        fft_img = fft2(img_padded)
        fft_psf = fft2(fftshift(psf_padded))
        res_complex = ifft2(fft_img * fft_psf)
        output = np.real(res_complex)[pad_h:pad_h+self.H, pad_w:pad_w+self.W]
        return np.clip(output, 0, 1), psf

    @staticmethod
    def _resize_psf(arr, target_shape):
        H_s, W_s = arr.shape
        H_t, W_t = target_shape
        if H_s == H_t and W_s == W_t: return arr
        out = np.zeros(target_shape, dtype=arr.dtype)
        ch_s, cw_s, ch_t, cw_t = H_s//2, W_s//2, H_t//2, W_t//2
        dy, dx = min(H_s, H_t)//2, min(W_s, W_t)//2
        out[ch_t-dy:ch_t+dy, cw_t-dx:cw_t+dx] = arr[ch_s-dy:ch_s+dy, cw_s-dx:cw_s+dx]
        return out

    # =========================================================================
    # THE GENERATOR LAYER (NumPy Specific)
    # This logic handles randomness and defaults, using the config variables.
    # =========================================================================
    def generate_aberrated_image(self, clean_image, coeffs=None, s_factor=None):
        """
        Main entry point for Dataset Generation.
        Handles random parameter generation if inputs are None.
        """
        # 1. Random Generation (אם לא סיפקו ערכים - תגריל מהקונפיג)
        if coeffs is None:
            coeffs = np.random.normal(0, ZERNIKE_STDS)        
        if s_factor is None:
            s_factor = np.random.uniform(*SCALE_RANGE)
        aberrated, psf = self.forward(clean_image, coeffs, s_factor)
        current_snr = np.random.uniform(TARGET_SNR - 10, TARGET_SNR + 10)
        noise_sigma = 1.0 / current_snr
        noisy = aberrated + np.random.normal(0, noise_sigma, aberrated.shape)
        
        return np.clip(noisy, 0, 1), psf, coeffs, s_factor, current_snr