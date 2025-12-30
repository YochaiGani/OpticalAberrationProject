import numpy as np
import cv2

class AutoTuner:
    """
    Analyzes image statistics to recommend restoration parameters.
    """
    
    @staticmethod
    def estimate_noise(img):
        """Estimates Gaussian noise standard deviation using MAD."""
        # Convert to float32 for processing
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            
        laplacian = cv2.Laplacian(img, cv2.CV_32F)
        sigma_est = np.median(np.abs(laplacian)) / 0.6745
        return np.clip(sigma_est, 0.001, 0.1)

    @staticmethod
    def get_wiener_k(img):
        return 2.0 * AutoTuner.estimate_noise(img)

    @staticmethod
    def get_rl_params(img):
        sigma = AutoTuner.estimate_noise(img)
        damping = 3.0 * sigma
        iterations = int(60 - (sigma * 1000)) 
        return max(15, min(iterations, 100)), damping

    @staticmethod
    def get_variational_params(img):
        sigma = AutoTuner.estimate_noise(img)
        lambda_tik = 2.5 * sigma
        lambda_tv = 0.4 * lambda_tik
        edge_sig = sigma * 1.2
        return lambda_tik, lambda_tv, edge_sig