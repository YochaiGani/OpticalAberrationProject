import sys, os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_factory import ImageGenerator
from core.physics_cpu import OpticalSimulator
from core.metrics import ImageMetrics
from core.inference import AberrationCorrector
from core.config import ZERNIKE_NAMES

def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/best_model.pth")
    
    try:
        corrector = AberrationCorrector(model_path)
    except:
        return

    clean = ImageGenerator.create_clean_image()
    sim = OpticalSimulator()
    dirty, _, true_coeffs, true_s, _ = sim.generate_aberrated_image(clean)
    
    print("Running AI Inference...")
    restored, pred_psf, pred_c, pred_s = corrector.restore(dirty, algo="rl")
    
    # Stats
    print(f"\nScale: True={true_s:.3f}, Pred={pred_s:.3f}")
    for n, t, p in zip(ZERNIKE_NAMES, true_coeffs, pred_c):
        print(f"{n:<10}: True={t:.3f}, Pred={p:.3f}, Diff={abs(t-p):.3f}")
        
    m = ImageMetrics.get_all(clean, restored)
    print(f"Metrics: {ImageMetrics.format_metrics(m)}")

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(clean, cmap='gray'); ax[0].set_title("Original")
    ax[1].imshow(dirty, cmap='gray'); ax[1].set_title("Input (Aberrated)")
    ax[2].imshow(restored, cmap='gray'); ax[2].set_title(f"AI Restoration\nPSNR: {m['PSNR']:.2f}")
    plt.show()

if __name__ == "__main__": main()