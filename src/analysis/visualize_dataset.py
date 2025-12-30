import sys, os
import tkinter as tk
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_factory import ImageGenerator
from core.physics_cpu import OpticalSimulator
from core.config import IMAGE_SIZE
from analysis.ui_utils import UIUtils

class DatasetVisualizer:
    def __init__(self):
        self.root = UIUtils.setup_window("Dataset Visualizer", 1000, 600)
        self.info = UIUtils.create_panel(self.root, self.generate)
        self.sim = OpticalSimulator(height=IMAGE_SIZE, width=IMAGE_SIZE)
        
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=20)
        self.l1 = tk.Label(self.frame); self.l1.grid(row=0, col=0, padx=10)
        self.l2 = tk.Label(self.frame); self.l2.grid(row=0, col=1, padx=10)
        
        self.generate()
        self.root.mainloop()

    def generate(self):
        clean = ImageGenerator.create_clean_image()
        dirty, _, coeffs, s, snr = self.sim.generate_aberrated_image(clean)
        
        self.l1.config(image=UIUtils.np_to_tk(clean, 400))
        self.l2.config(image=UIUtils.np_to_tk(dirty, 400))
        self.l1.image = UIUtils.np_to_tk(clean, 400)
        self.l2.image = UIUtils.np_to_tk(dirty, 400)
        
        rms = np.linalg.norm(coeffs)
        self.info.config(text=f"Scale: {s:.2f} | SNR: {snr:.1f} | RMS: {rms:.3f}")

if __name__ == "__main__": DatasetVisualizer()