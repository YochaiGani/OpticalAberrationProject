import torch
import numpy as np
import os
import sys

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.estimator import AberrationNet
from core.physics_cpu import OpticalSimulator
from core.config import IMAGE_SIZE
from algorithms.classical import ClassicalRestoration

class AberrationCorrector:
    """
    Production-ready inference pipeline.
    Loads the trained model once and allows restoring images easily.
    """
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Core] Initializing Corrector on {self.device}...")
        
        # 1. Load Neural Network
        self.model = AberrationNet().to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("[Core] Model weights loaded successfully.")
        else:
            print(f"[Error] Model not found at {model_path}")
            raise FileNotFoundError(model_path)

        # 2. Load Physics Engine (for PSF generation)
        self.sim = OpticalSimulator(height=IMAGE_SIZE, width=IMAGE_SIZE)

    def predict(self, image):
        """Returns: coeffs, scale"""
        # Prepare input
        if isinstance(image, np.ndarray):
            img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            img_tensor = image
            
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            pred_coeffs, pred_scale = self.model(img_tensor)
            
        return pred_coeffs.cpu().numpy().flatten(), pred_scale.item()

    def restore(self, distorted_image, algo="rl", iterations=30):
        """
        Full Pipeline: Image -> Net -> Coeffs -> PSF -> Restoration
        """
        # A. Predict
        coeffs, scale = self.predict(distorted_image)
        
        # B. Physics (Build PSF)
        predicted_psf = self.sim.get_psf(coeffs, scale)
        
        # C. Restore
        if algo == "rl":
            restored = ClassicalRestoration.richardson_lucy(distorted_image, predicted_psf, iterations=iterations)
        elif algo == "wiener":
            restored = ClassicalRestoration.wiener(distorted_image, predicted_psf)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
            
        return restored, predicted_psf, coeffs, scale