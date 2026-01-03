import os
import torch

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
IMAGE_SIZE = 512        
SUPERSAMPLE_FACTOR = 1.5 

# ==========================================
# 2. GENERATION CONTENT
# ==========================================
TEXTURE_TYPES = [
    "white", "gray", 
    "stripes2", "stripes3",     
    "checker",                  
    "noise",                    
    "multiband"
]

BACKGROUND_MODES = ["black", "white", "half", "stripes", "checker"]

OBJECT_TYPES = [
    "circle", "ellipse", "ring", "rect", "triangle", "polygon", "line",
    "siemens", "bullseye", "slanted_edge", 
    "lamp_row", "window_grid",             
    "scribble"                             
]

# Weights optimized for Optical Aberration training
SHAPE_WEIGHTS = {
    "siemens": 4.0, 
    "bullseye": 3.0,     
    "slanted_edge": 3.0, 
    "window_grid": 3.0,
    "scribble": 4.0,    
    "stripes3": 2.5,
    "circle": 0.5,
    "rect": 0.5,
    "blob": 0.2    
}

# ==========================================
# 3. OPTICAL PHYSICS CONFIGURATION
# ==========================================
ZERNIKE_NAMES = ['Z4 (Defocus)', 'Z5 (Astig)', 'Z6 (Astig)', 'Z7 (Coma)', 'Z8 (Coma)', 'Z12 (Sphere)']

# Standard deviations for Zernike coefficients
ZERNIKE_STDS = [0.35, 0.25, 0.25, 0.14, 0.14, 0.08]

SCALE_RANGE = (0.8, 1.2)
TARGET_SNR = 60

SAMPLES_PER_IMAGE = 5  
TOTAL_CLEAN_IMAGES = 20000 
CHECKPOINT_INTERVAL = 100

# ==========================================
# 4. TRAINING & RUNTIME CONFIGURATION (NEW)
# ==========================================

# --- Hardware Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False

# --- Paths --- 
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Training Hyperparameters ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
WEIGHT_DECAY = 1e-4
USE_PRETRAINED_WEIGHTS = True

# --- Loss Weights ---
ALPHA = 1.0  # Coeffs
BETA = 0.5   # Pixel
GAMMA = 0.1  # Perceptual