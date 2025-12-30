import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from core.image_factory import ImageGenerator
from core.physics_cpu import OpticalSimulator
from core.config import IMAGE_SIZE, TARGET_SNR, ZERNIKE_STDS, SAMPLES_PER_IMAGE, TOTAL_CLEAN_IMAGES, CHECKPOINT_INTERVAL
import cv2


class DatasetManager:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.clean_dir = os.path.join(output_dir, "clean")
        self.aberrated_dir = os.path.join(output_dir, "aberrated")
        self.labels_file = os.path.join(output_dir, "labels.npy")
        self.meta_file = os.path.join(output_dir, "metadata.json")

    def _prepare_directories(self):
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.aberrated_dir, exist_ok=True)

    def _save_image(self, arr, path):
        arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr_u8, mode='L').save(path)

    def _save_metadata(self):
        meta = {
            "created_at": str(datetime.now()),
            "clean_images_target": TOTAL_CLEAN_IMAGES,
            "distortions_per_image": SAMPLES_PER_IMAGE,
            "image_size": IMAGE_SIZE,
            "target_snr": TARGET_SNR,
            "zernike_stds": ZERNIKE_STDS
        }
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=4)
    
    def _get_resume_state(self, target_count):
        if os.path.exists(self.clean_dir):
            current_count = len([f for f in os.listdir(self.clean_dir) if f.endswith('.png')])
        else:
            current_count = 0

        if current_count >= target_count:
            print(f"[INFO] ✅ Dataset already has {current_count} images. Nothing to do.")
            return None, None  
        
        print(f"[INFO] ? Resuming: Found {current_count} images. Generating {target_count - current_count} more...")
        all_labels = []
        
        if os.path.exists(self.labels_file) and current_count > 0:
            try:
                all_labels = list(np.load(self.labels_file))
                print(f"[INFO] Loaded {len(all_labels)} existing labels.")
            except Exception as e:
                print(f"[WARN] Failed to load labels: {e}. Starting with empty labels.")
                
        return current_count, all_labels
    
    def generate(self):
        self._prepare_directories()
        self._save_metadata()
        
        start_idx, all_labels = self._get_resume_state(TOTAL_CLEAN_IMAGES)
        if start_idx is None:
            return

        print(f"[INFO] Initializing simulator...")
        simulator = OpticalSimulator(height=IMAGE_SIZE, width=IMAGE_SIZE)
        
        print(f"[INFO] Starting generation loop...")
        pbar = tqdm(range(start_idx, TOTAL_CLEAN_IMAGES), desc="Generating", initial=start_idx, total=TOTAL_CLEAN_IMAGES)

        """
        print(f"[INFO] Initializing simulator...")
        simulator = OpticalSimulator(height=IMAGE_SIZE, width=IMAGE_SIZE)
        
        all_labels = []
        
        print(f"[INFO] Starting generation (Safe Mode). Saving checkpoint every {CHECKPOINT_INTERVAL} images.")

        pbar = tqdm(total=TOTAL_CLEAN_IMAGES, desc="Generating")
        """
        try:
            for i in pbar: #range(TOTAL_CLEAN_IMAGES)
                # 1. Generate Clean
                try:
                    clean_img = ImageGenerator.create_clean_image()
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                clean_filename = f"{i:05d}.png"
                self._save_image(clean_img, os.path.join(self.clean_dir, clean_filename))

                # 2. Generate Distortions
                for j in range(SAMPLES_PER_IMAGE):
                    dirty_img, _, coeffs, s_factor, snr_val = simulator.generate_aberrated_image(clean_img)
                    
                    dirty_filename = f"{i:05d}_{j}.png"
                    self._save_image(dirty_img, os.path.join(self.aberrated_dir, dirty_filename))
                    
                    # Label: [Z4...Z12, Scale, SNR, Clean_ID]
                    label = list(coeffs) + [s_factor, snr_val, i] 
                    all_labels.append(label)
                
                pbar.update(1)
                
                # --- SAFETY CHECKPOINT ---
                # שומר את הלייבלים לדיסק תוך כדי ריצה
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    np.save(self.labels_file, np.array(all_labels, dtype=np.float32))
                    
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user! Saving progress so far...")
        except Exception as e:
            print(f"\n[ERROR] Crashed: {e}. Saving progress...")
        finally:
            # שמירה סופית בכל מקרה (גם אם קרס באמצע)
            if len(all_labels) > 0:
                np.save(self.labels_file, np.array(all_labels, dtype=np.float32))
                print(f"\n[SAVED] Saved {len(all_labels)} samples to labels.npy")
                print("You can start training with this partial dataset.")
            
            pbar.close()

if __name__ == "__main__":
    manager = DatasetManager()
    manager.generate()