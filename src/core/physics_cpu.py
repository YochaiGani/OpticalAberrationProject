import numpy as np
import pyfftw
import cv2
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10.0)

from core.config import ZERNIKE_STDS, SCALE_RANGE, TARGET_SNR

class OpticalSimulator:
    def __init__(self, height=512, width=512):
        self.H = height
        self.W = width
        
        # 1. Pre-compute Grid and Basis
        self.R_max = np.sqrt((self.H // 2)**2 + (self.W // 2)**2) + 1e-12
        y, x = np.ogrid[-(self.H // 2):(self.H + 1) // 2, -(self.W // 2):(self.W + 1) // 2]
        self.i_grid = y / self.R_max
        self.j_grid = x / self.R_max
        self.zernike_maps = self._precompute_zernike_basis()

    def _precompute_zernike_basis(self):
        x, y = self.j_grid, self.i_grid
        r2 = x**2 + y**2
        maps = np.array([
            np.sqrt(3) * (2 * r2 - 1),                # Z4
            2 * np.sqrt(6) * x * y,                   # Z5
            np.sqrt(6) * (x**2 - y**2),               # Z6
            np.sqrt(8) * (3 * r2 - 2) * y,            # Z7
            np.sqrt(8) * (3 * r2 - 2) * x,            # Z8
            np.sqrt(5) * (6 * r2**2 - 6 * r2 + 1)     # Z12
        ])
        return maps

    def get_psf(self, coeffs_list, s_factor):
        """
        Generates ONLY the Point Spread Function (Kernel).
        Used by both the simulator (forward) and the restorer (inverse).
        """
        # 1. Phase Calculation
        phase = np.tensordot(coeffs_list, self.zernike_maps, axes=1)
        
        # 2. Pupil & Raw PSF
        pupil = np.exp(1j * 2 * np.pi * phase)
        pupil_fft = fftshift(fft2(pupil))
        iPSF = np.abs(pupil_fft)**2
        iPSF /= (np.sum(iPSF) + 1e-10)

        # 3. Scaling (Zoom)
        new_size = (int(self.W * s_factor), int(self.H * s_factor))
        iPSF_Scaled = cv2.resize(iPSF.astype(np.float32), new_size, interpolation=cv2.INTER_LINEAR)
        
        # 4. Crop/Pad to final size
        iPSF_final = self._resize_psf(iPSF_Scaled, (self.H, self.W))
        iPSF_final /= (np.sum(iPSF_final) + 1e-10)
        
        return iPSF_final

    def generate_aberrated_image(self, image_clean, coeffs_list=None, s_factor=None):
        if coeffs_list is None: coeffs_list = np.random.normal(0, ZERNIKE_STDS)
        if s_factor is None: s_factor = np.random.uniform(*SCALE_RANGE)
        iPSF_final = self.get_psf(coeffs_list, s_factor)

        pad_h, pad_w = self.H // 2, self.W // 2
        img_padded = np.pad(image_clean, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        psf_padded = np.pad(iPSF_final, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        fft_img = fft2(img_padded)
        fft_psf = fft2(fftshift(psf_padded))
        conv_result = np.real(ifft2(fft_img * fft_psf))
        
        conv_result = conv_result[pad_h:pad_h + self.H, pad_w:pad_w + self.W]
        conv_result = np.clip(conv_result, 0, 1)

        current_snr = np.random.uniform(TARGET_SNR - 10, TARGET_SNR + 10)
        noise_sigma = 1.0 / current_snr
        final = conv_result + np.random.normal(0, noise_sigma, conv_result.shape)
        
        return np.clip(final, 0, 1), iPSF_final, coeffs_list, s_factor, current_snr

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