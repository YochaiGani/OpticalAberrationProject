import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import ZERNIKE_NAMES
from core.image_factory import ImageGenerator
from core.physics_cpu import OpticalSimulator
from core.metrics import ImageMetrics
from core.tuner import AutoTuner
from algorithms.classical import ClassicalRestoration
from algorithms.variational import VariationalRestorer
from analysis.ui_utils import UIUtils

class ClassicalLab:
    def __init__(self):
        self.root = UIUtils.setup_window("Classical Algorithms Lab", 1500, 600)
        self.info = UIUtils.create_panel(self.root, self.run)
        self.fig, self.axs, self.canvas = UIUtils.embed_plot(self.root, (1, 5))
        self.sim = OpticalSimulator()
        self.v_restorer = VariationalRestorer()
        self.run()
        self.root.mainloop()

    def run(self):
        self.info.config(text="Calculating...")
        self.root.update()
        
        clean = ImageGenerator.create_clean_image()
        dirty, psf, coeffs, s, snr = self.sim.generate_aberrated_image(clean)
        
        # Algorithmic Restoration
        res_wiener = ClassicalRestoration.wiener(dirty, psf)
        res_rl = ClassicalRestoration.richardson_lucy(dirty, psf)
        
        l_tik, l_tv, e_sig = AutoTuner.get_variational_params(dirty)
        res_var = self.v_restorer.restore(dirty, coeffs, s, iterations=150, 
                                          lambda_tikhonov=l_tik, lambda_tv=l_tv, edge_sigma=e_sig)
        
        coeff_lines = [f" {name:<15}: {val:+.4f}λ" 
                       for name, val in zip(ZERNIKE_NAMES, coeffs)]

        self.info.config(text=f"Scale: {s:.2f} | SNR: {snr:.1f} \nCoeffs:\n" + "\n".join(coeff_lines))
        
        self._show(0, clean, "Original")
        self._show(1, dirty, "Input", ImageMetrics.get_metrics(clean, dirty))
        self._show(2, res_wiener, "Wiener", ImageMetrics.get_metrics(clean, res_wiener))
        self._show(3, res_rl, "Richardson-Lucy", ImageMetrics.get_metrics(clean, res_rl))
        self._show(4, res_var, "Variational", ImageMetrics.get_metrics(clean, res_var))
        self.canvas.draw()

    def _show(self, idx, img, title, m=None):
        self.axs[idx].clear()
        self.axs[idx].imshow(img, cmap='gray')
        if m: title += "\n" + ImageMetrics.format_metrics(m)
        self.axs[idx].set_title(title, fontsize=8)
        self.axs[idx].axis('off')

if __name__ == "__main__": ClassicalLab()