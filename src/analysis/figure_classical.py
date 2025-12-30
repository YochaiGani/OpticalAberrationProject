import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_factory import ImageGenerator
from core.physics_cpu import OpticalSimulator
from core.metrics import ImageMetrics
from algorithms.classical import ClassicalRestoration
from algorithms.variational import VariationalRestorer

def main():
    print("Generating Figure...")
    clean = ImageGenerator.create_clean_image()
    sim = OpticalSimulator()
    dirty, psf, coeffs, scale, snr = sim.generate_aberrated_image(clean, coeffs_list=[0.5, 0.3, -0.2, 0,0,0], s_factor=1.1)
    
    res_wiener = ClassicalRestoration.wiener(dirty, psf)
    res_rl = ClassicalRestoration.richardson_lucy(dirty, psf, iterations=40)
    res_var = VariationalRestorer().restore(dirty, coeffs, scale, iterations=100)
    
    imgs = [clean, dirty, res_wiener, res_rl, res_var]
    titles = ["Original", "Input", "Wiener", "Richardson-Lucy", "Variational"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray')
        m = ImageMetrics.get_metrics(clean, img) if t != "Original" else None
        if m: t += "\n" + ImageMetrics.format_metrics(m)
        ax.set_title(t)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("../paper_figure_comparison.png", dpi=300)
    print("Saved ../paper_figure_comparison.png")

if __name__ == "__main__": main()