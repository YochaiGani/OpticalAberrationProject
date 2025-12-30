import numpy as np
import pyfftw
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_cpu import OpticalSimulator
from core.config import IMAGE_SIZE
from core.tuner import AutoTuner  

pyfftw.interfaces.cache.enable()

class VariationalRestorer:
    def __init__(self):
        self.simulator = OpticalSimulator(height=IMAGE_SIZE, width=IMAGE_SIZE)

    def restore(self, I0, coeffs, s_factor, 
                iterations=150, dt=0.15, 
                lambda_tikhonov=None, lambda_tv=None, edge_sigma=None):
        
        # --- Auto Tuning Logic ---
        if lambda_tikhonov is None:
            sigma = AutoTuner.estimate_noise(I0)
            lambda_tikhonov = 2.5 * sigma
            lambda_tv = 0.4 * lambda_tikhonov
            edge_sigma = sigma * 1.2
        # -------------------------

        # 1. Get PSF
        h = self.simulator.get_psf(coeffs, s_factor)
        H = fft2(fftshift(h)) 
        R_hh = np.abs(H)**2
        H_conj = np.conj(H)

        # 2. Init
        I_est = I0.copy().astype(np.float32)
        I0_conv_h_dagger = np.real(ifft2(fft2(I0) * H_conj))

        # 3. Loop
        for i in range(iterations):
            I_conv_Rhh = np.real(ifft2(fft2(I_est) * R_hh))
            grad_fidelity = I_conv_Rhh - I0_conv_h_dagger
            grad_reg = self._compute_grads(I_est, lambda_tikhonov, lambda_tv, edge_sigma)
            
            I_est = np.clip(I_est - dt * (grad_fidelity + grad_reg), 0, 1)

        return I_est

    def _compute_grads(self, I, l_tik, l_tv, sigma):
        # OpenCV Gradients (Float32 enforce)
        I_cv = I.astype(np.float32)
        Ix = cv2.Sobel(I_cv, cv2.CV_32F, 1, 0, ksize=1)
        Iy = cv2.Sobel(I_cv, cv2.CV_32F, 0, 1, ksize=1)
        grad_sq = Ix**2 + Iy**2
        
        edge = np.exp(-grad_sq / (2 * sigma**2))
        
        lap = cv2.Laplacian(I_cv, cv2.CV_32F, ksize=1)
        term_tik = -2 * (l_tik * edge) * lap

        grad_mag = np.sqrt(grad_sq + 1e-6)
        div = cv2.Sobel(Ix/grad_mag, cv2.CV_32F, 1, 0, ksize=1) + \
              cv2.Sobel(Iy/grad_mag, cv2.CV_32F, 0, 1, ksize=1)
        term_tv = -(l_tv * (1.0 - edge)) * div
        
        return term_tik + term_tv