import os
import sys

# --- Improved import guard for PyTorch / torchvision ---
try:
	import torch
	import torch.nn as nn
	import torch.optim as optim
	from torch.utils.data import Dataset, DataLoader
	from torchvision import transforms, models
except OSError as e:
	sys.stderr.write("[ERROR] Failed to load PyTorch native libraries (OSError: WinError 126 or similar).\n")
	sys.stderr.write(f"Detail: {e}\n\n")
	sys.stderr.write("Likely causes:\n")
	sys.stderr.write(" - Missing Microsoft Visual C++ Redistributable (2015-2022)\n")
	sys.stderr.write(" - Installed torch wheel expects CUDA/driver not present or mismatched\n")
	sys.stderr.write(" - 32-bit Python vs 64-bit torch wheel mismatch\n\n")
	sys.stderr.write("Suggested fixes (run inside your activated venv/conda env):\n")
	sys.stderr.write(" 1) Install MSVC redistributable:\n")
	sys.stderr.write("    https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n")
	sys.stderr.write(" 2) Reinstall torch (CPU-only example):\n")
	sys.stderr.write("    pip uninstall -y torch torchvision torchaudio\n")
	sys.stderr.write("    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision\n")
	sys.stderr.write(" 3) Or install matching CUDA build (example for CUDA 11.8):\n")
	sys.stderr.write("    pip uninstall -y torch torchvision torchaudio\n")
	sys.stderr.write("    pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision\n")
	sys.stderr.write(" 4) Verify Python is 64-bit: python -c \"import struct; print(struct.calcsize('P')*8)\"\n\n")
	sys.stderr.write("After fixing, re-run this script.\n")
	sys.exit(1)
except ImportError as e:
	sys.stderr.write("[ERROR] PyTorch import failed (ImportError).\n")
	sys.stderr.write(str(e) + "\n")
	sys.exit(1)

# --- Improved import guard for NumPy ---
try:
	import numpy as np
except ImportError as e:
	sys.stderr.write("[ERROR] NumPy import failed (ImportError).\n")
	sys.stderr.write("Ensure you are not in the NumPy source directory and reinstall NumPy if necessary.\n")
	sys.stderr.write(str(e) + "\n")
	sys.exit(1)

# --- Runtime compatibility check: NumPy <-> OpenCV ---
def _parse_ver(v):
	try:
		parts = v.split('.')
		return tuple(int(p) for p in parts[:3])
	except Exception:
		return None

try:
	import importlib
	if importlib.util.find_spec("cv2") is not None:
		import cv2
		np_v = _parse_ver(np.__version__)
		cv_v = cv2.__version__
		if np_v and (np_v[0] >= 2 and np_v[1] >= 3):
			# Known constraint: opencv-python 4.12.x requires numpy < 2.3.0
			print(f"[WARN] Detected numpy {np.__version__} and OpenCV {cv_v}.")
			print("[WARN] opencv-python (4.12.x) requires numpy<2.3.0. This can lead to runtime issues.")
			print("Suggested fixes (run inside your activated venv):")
			print(" 1) Downgrade numpy to a compatible release, e.g.:")
			print("    pip install --upgrade --force-reinstall 'numpy<2.3.0,>=2.0.0'  # e.g. numpy==2.2.4")
			print(" 2) OR upgrade opencv-python to a version that supports numpy 2.3+ (if available):")
			print("    pip install --upgrade --force-reinstall opencv-python")
			print("After adjusting packages, re-run this script.")
except Exception:
	# Non-fatal: continue if cv2 not installed or parse fails
	pass

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess

# Import project modules
from core.diff_optics import DifferentiableOpticalSimulator
from networks.estimator import AberrationNet
from core.config import IMAGE_SIZE

# ==========================================
# 0. CUDA / Driver Compatibility CHECK
# ==========================================
def _parse_version_tuple(vstr):
	# Safely parse "11.8" -> (11,8), "11" -> (11,0), handle None
	if not vstr:
		return None
	parts = vstr.split('.')
	try:
		major = int(parts[0])
		minor = int(parts[1]) if len(parts) > 1 else 0
		return (major, minor)
	except Exception:
		return None

def check_cuda_driver_compatibility():
	"""
	Prints PyTorch build CUDA, whether CUDA is available, and attempts to read driver/runtime CUDA via nvidia-smi.
	Provides actionable suggestions if versions mismatch.
	"""
	torch_cuda = getattr(torch.version, "cuda", None)
	cuda_available = torch.cuda.is_available()
	print(f"[INFO] torch.version.cuda = {torch_cuda} | torch.cuda.is_available = {cuda_available}")

	# Try nvidia-smi to get driver + runtime CUDA
	driver_ver = None
	nvidia_cuda = None
	try:
		out = subprocess.check_output(
			["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
			universal_newlines=True, stderr=subprocess.STDOUT
		).strip()
		if out:
			# Example out: "576.88, 12.2"
			first = out.splitlines()[0]
			parts = [p.strip() for p in first.split(',')]
			if len(parts) >= 1:
				driver_ver = parts[0]
			if len(parts) >= 2:
				nvidia_cuda = parts[1]
	except FileNotFoundError:
		print("[WARN] nvidia-smi not found on PATH. Cannot auto-detect system driver/CUDA. Use 'nvidia-smi' in a shell to inspect.")
	except subprocess.CalledProcessError as e:
		print(f"[WARN] nvidia-smi returned error: {e}. Output may be unavailable.")

	print(f"[INFO] System driver version: {driver_ver or 'unknown'} | system CUDA (nvidia-smi): {nvidia_cuda or 'unknown'}")

	# Compare parsed versions
	torch_v = _parse_version_tuple(torch_cuda)
	system_v = _parse_version_tuple(nvidia_cuda)

	if torch_v and system_v:
		# If driver/runtime CUDA (system_v) is older than torch build (torch_v) -> warn
		if system_v[0] < torch_v[0] or (system_v[0] == torch_v[0] and system_v[1] < torch_v[1]):
			print("[ERROR] Detected system CUDA/runtime is older than the PyTorch build's CUDA. This can cause DLL load errors.")
			print("Suggested fixes (pick one, run inside your activated venv):")
			print(" 1) Update NVIDIA driver to a version that supports the required CUDA runtime (recommended).")
			print("    After driver update, re-run: nvidia-smi")
			print(" 2) Install a PyTorch wheel built for your system CUDA (or install CPU-only wheel). Examples:")
			print("    # CPU-only (safe fallback):")
			print("    pip uninstall -y torch torchvision torchaudio")
			print("    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio")
			print("    # Example CUDA wheel (replace cuXXX with matching CUDA, e.g. cu118):")
			print("    pip uninstall -y torch torchvision torchaudio")
			print("    pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio")
			print(" 3) Use the PyTorch Get Started selector to pick the correct command: https://pytorch.org/get-started/locally/")
		else:
			print("[OK] System CUDA/runtime >= PyTorch build CUDA. Driver/CUDA should be compatible.")
	else:
		print("[WARN] Could not determine both PyTorch CUDA build and system CUDA/runtime. Manual check recommended:")
		print(" - Run: python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"")
		print(" - Run: nvidia-smi  (inspect 'CUDA Version' and 'Driver Version')")
		print("If you see a mismatch, follow the suggested fixes above.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running training on device: {DEVICE}")

# Call compatibility check early (non-fatal)
try:
	check_cuda_driver_compatibility()
except Exception as e:
	print(f"[WARN] Compatibility check failed: {e}")

# Hyperparameters
# הערה: VGG צורך הרבה זיכרון. אם יש שגיאת CUDA OOM, הקטן ל-4.
BATCH_SIZE = 8           
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40

# Loss Function Weights (Hybrid Loss)
ALPHA = 1.0    # Parameter Loss: Direct supervision on coeffs
BETA  = 0.5    # Physics Loss: Reconstruction pixel MSE
GAMMA = 0.1    # Perceptual Loss: Semantic/Feature similarity

# Paths
DATA_DIR = "data"

LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
IMG_DIR_ABERRATED = os.path.join(DATA_DIR, "aberrated")
IMG_DIR_CLEAN = os.path.join(DATA_DIR, "clean")
MODEL_SAVE_PATH = "../models/best_model.pth"
PLOT_SAVE_PATH = "../training_results.png"

# ==========================================
# 2. PERCEPTUAL LOSS (VGG19)
# ==========================================
class VGGLoss(nn.Module):
    """
    Computes perceptual loss using VGG19 features.
    Ensures the model learns structural sharpness, not just pixel average.
    """
    def __init__(self):
        super().__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        # Use first 35 layers (Low to Mid-level features)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:35]).eval()
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.criterion = nn.MSELoss()
        
        # ImageNet Normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Input x, y are Grayscale [Batch, 1, H, W]
        # VGG expects RGB. We repeat the channel 3 times.
        x_rgb = x.repeat(1, 3, 1, 1)
        y_rgb = y.repeat(1, 3, 1, 1)
        
        # Normalize
        x_norm = (x_rgb - self.mean) / self.std
        y_norm = (y_rgb - self.mean) / self.std
        
        # Extract features
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        
        return self.criterion(x_features, y_features)

# ==========================================
# 3. DATASET
# ==========================================
class AberrationDataset(Dataset):
    def __init__(self, labels_file, aberrated_dir, clean_dir):
        self.labels = np.load(labels_file)
        self.aberrated_dir = aberrated_dir
        self.clean_dir = clean_dir
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Sorted files match labels index
        self.file_list = sorted(os.listdir(aberrated_dir))
        
        if len(self.file_list) != len(self.labels):
            print(f"[WARN] Files ({len(self.file_list)}) != Labels ({len(self.labels)})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Load Input (Aberrated)
        fname = self.file_list[idx]
        img_in = Image.open(os.path.join(self.aberrated_dir, fname)).convert('L')
        
        # 2. Get Targets from Label Array
        # Structure: [Z4..Z12 (6), Scale (1), SNR (1), Clean_ID (1)]
        full_label = self.labels[idx]
        
        target_coeffs = torch.tensor(full_label[:6], dtype=torch.float32)
        target_scale = torch.tensor(full_label[6], dtype=torch.float32)
        
        # 3. Load Ground Truth (Clean) using ID
        clean_id = int(full_label[8])
        clean_fname = f"{clean_id:05d}.png"
        img_clean = Image.open(os.path.join(self.clean_dir, clean_fname)).convert('L')

        return self.transform(img_in), self.transform(img_clean), target_coeffs, target_scale

# ==========================================
# 4. TRAINER ENGINE
# ==========================================
class Trainer:
    def __init__(self):
        # 1. Initialize Components
        self.model = AberrationNet().to(DEVICE)
        self.simulator = DifferentiableOpticalSimulator(output_size=IMAGE_SIZE).to(DEVICE)
        self.vgg_loss = VGGLoss().to(DEVICE)
        
        # 2. Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion_mse = nn.MSELoss()
        
        # 3. Data Loading
        self._prepare_data()
        self.history = {'train_loss': [], 'val_loss': []}

    def _prepare_data(self):
        print("[INFO] Loading dataset...")
        dataset = AberrationDataset(LABELS_PATH, IMG_DIR_ABERRATED, IMG_DIR_CLEAN)
        
        # 80/20 Split
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
        
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"[INFO] Ready. Train: {train_len}, Val: {val_len}")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        
        for img_in, img_clean, gt_coeffs, gt_scale in loop:
            img_in, img_clean = img_in.to(DEVICE), img_clean.to(DEVICE)
            gt_coeffs, gt_scale = gt_coeffs.to(DEVICE), gt_scale.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            # --- 1. Forward Pass (Prediction) ---
            pred_coeffs, pred_scale = self.model(img_in)
            
            # --- 2. Parameter Loss (Alpha) ---
            loss_c = self.criterion_mse(pred_coeffs, gt_coeffs)
            loss_s = self.criterion_mse(pred_scale.squeeze(), gt_scale)
            loss_param = loss_c + loss_s
            
            # --- 3. Physics Simulation (Reconstruction) ---
            # Input: Clean image + Predicted params -> Output: Reconstructed Aberrated
            reconstructed = self.simulator(img_clean, pred_coeffs, pred_scale.squeeze())
            
            # --- 4. Pixel Loss (Beta) ---
            loss_pixel = self.criterion_mse(reconstructed, img_in)
            
            # --- 5. Perceptual Loss (Gamma) ---
            loss_perceptual = self.vgg_loss(reconstructed, img_in)
            
            # Total Loss
            final_loss = (ALPHA * loss_param) + (BETA * loss_pixel) + (GAMMA * loss_perceptual)
            
            # --- Optimization ---
            final_loss.backward()
            self.optimizer.step()
            
            total_loss += final_loss.item()
            loop.set_postfix(loss=final_loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for img_in, img_clean, gt_coeffs, gt_scale in self.val_loader:
                img_in, img_clean = img_in.to(DEVICE), img_clean.to(DEVICE)
                gt_coeffs, gt_scale = gt_coeffs.to(DEVICE), gt_scale.to(DEVICE)
                
                pred_coeffs, pred_scale = self.model(img_in)
                reconstructed = self.simulator(img_clean, pred_coeffs, pred_scale.squeeze())
                
                loss_param = self.criterion_mse(pred_coeffs, gt_coeffs) + \
                             self.criterion_mse(pred_scale.squeeze(), gt_scale)
                loss_pixel = self.criterion_mse(reconstructed, img_in)
                loss_perceptual = self.vgg_loss(reconstructed, img_in)
                
                final_loss = (ALPHA * loss_param) + (BETA * loss_pixel) + (GAMMA * loss_perceptual)
                total_loss += final_loss.item()
                
        return total_loss / len(self.val_loader)

    def run(self):
        print(f"\n[INFO] Starting training with VGG Perceptual Loss...")
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f">>> Saved Best Model! Loss: {best_val_loss:.5f}")
        
        self._plot_results()

    def _plot_results(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Training Loss (Param + Physics + VGG)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(PLOT_SAVE_PATH)
        print("Plot saved.")

if __name__ == "__main__":
    with open("data/labels.npy", "r") as f:
        print("\nsuccess\n")
    os.makedirs("../models", exist_ok=True)
    trainer = Trainer()
    trainer.run()