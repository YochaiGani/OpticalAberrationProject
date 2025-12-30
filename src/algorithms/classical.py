import numpy as np
import cv2
import sys 
import os
import pyfftw
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10.0)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tuner import AutoTuner 

class ClassicalRestoration:
    
    @staticmethod
    def wiener(img, psf, K=None):
        if K is None:
            K = AutoTuner.get_wiener_k(img)
            
        H = fft2(fftshift(psf))
        Y = fft2(img)
        H_conj = np.conj(H)
        Wiener = H_conj / (np.abs(H)**2 + K)
        return np.clip(np.real(ifft2(Y * Wiener)), 0, 1)

    @staticmethod
    def richardson_lucy(image, psf, iterations=None, damping=None):
        if iterations is None or damping is None:
            auto_iter, auto_damp = AutoTuner.get_rl_params(image)
            iterations = iterations or auto_iter
            damping = damping or auto_damp

        img = image.astype(np.float32)
        psf = psf.astype(np.float32)
        psf /= (psf.sum() + 1e-12)
        
        estimate = img.copy()
        psf_mirror = cv2.flip(psf, -1)
        H = fft2(fftshift(psf))
        H_mirror = fft2(fftshift(psf_mirror))
        
        for i in range(iterations):
            est_fft = fft2(estimate)
            est_blurred = np.real(ifft2(est_fft * H))
            
            ratio = img / (est_blurred + 1e-12)
            
            if damping > 0:
                mask = np.abs(ratio - 1.0) < damping
                ratio[mask] = 1.0
                
            ratio_fft = fft2(ratio)
            error_est = np.real(ifft2(ratio_fft * H_mirror))
            estimate = np.clip(estimate * error_est, 0, 1)

        return estimate