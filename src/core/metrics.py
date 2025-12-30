import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips

# Initialize LPIPS model once (Heavy operation)
loss_fn_alex = lpips.LPIPS(net='alex', version='0.1') 
if torch.cuda.is_available():
    loss_fn_alex = loss_fn_alex.cuda()

class ImageMetrics:
    @staticmethod
    def calculate_psnr(img1, img2):
        """Higher is Better"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return 20 * np.log10(1.0 / np.sqrt(mse))

    @staticmethod
    def calculate_ssim(img1, img2):
        """Higher is Better (Max 1.0)"""
        return ssim(img1, img2, data_range=1.0)

    @staticmethod
    def calculate_lpips(img1, img2):
        """Lower is Better (0 is perfect)"""
        # Convert Numpy [0,1] to Tensor [-1,1]
        t1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
        t2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
        t1 = t1 * 2 - 1
        t2 = t2 * 2 - 1
        
        if torch.cuda.is_available():
            t1 = t1.cuda()
            t2 = t2.cuda()
            
        with torch.no_grad():
            dist = loss_fn_alex(t1, t2)
            
        return dist.item()

    @staticmethod
    def get_metrics(clean, restored):
        return {
            "PSNR": ImageMetrics.calculate_psnr(clean, restored),
            "SSIM": ImageMetrics.calculate_ssim(clean, restored),
            "LPIPS": ImageMetrics.calculate_lpips(clean, restored)
        }
    
    @staticmethod
    def format_metrics(metrics_dict):
        return f"PSNR: {metrics_dict['PSNR']:.2f} | SSIM: {metrics_dict['SSIM']:.3f} | LPIPS: {metrics_dict['LPIPS']:.3f}"
