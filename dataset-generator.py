#!/usr/bin/env python3
"""
AUTO DATASET CREATOR FOR TEETH DETECTOR
-----------------------------------------
Creates the folder:

teeth_detector_dataset/
    ├── teeth/        (copied from Dental_diseases_Model)
    └── not_teeth/    (automatically downloaded images)

Run this file with:
    python generate_teeth_dataset.py
"""

import os
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------
# PROJECT ROOT (YOUR MAIN FOLDER)
# --------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Machine Learning Projects\dental_diseases_detection")

# Existing disease dataset folder (source of teeth images)
DISEASE_DATASET = Path(r"D:\Machine Learning Projects\dental_diseases_detection\Dental diseases_Model")



# New dataset for Stage-1 teeth detector
TEETH_DETECTOR_DATASET = PROJECT_ROOT / "teeth_detector_dataset"

TEETH_DIR = TEETH_DETECTOR_DATASET / "teeth"
NOT_TEETH_DIR = TEETH_DETECTOR_DATASET / "not_teeth"

# Create folders
TEETH_DIR.mkdir(parents=True, exist_ok=True)
NOT_TEETH_DIR.mkdir(parents=True, exist_ok=True)

print("\n---------------------------------------------")
print(" AUTO TEETH DETECTOR DATASET CREATOR")
print("---------------------------------------------")
print(f"Project folder: {PROJECT_ROOT}")
print(f"Teeth detector dataset: {TEETH_DETECTOR_DATASET}")
print("---------------------------------------------\n")


# --------------------------------------------------------------------
# 1. COPY TEETH IMAGES FROM EXISTING DATASET
# --------------------------------------------------------------------
print("📌 Copying TEETH images from:", DISEASE_DATASET)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
count_teeth = 0

for class_folder in DISEASE_DATASET.iterdir():
    if class_folder.is_dir():
        for img in class_folder.rglob("*"):
            if img.suffix.lower() in image_extensions:
                target = TEETH_DIR / f"teeth_{count_teeth}{img.suffix.lower()}"
                shutil.copy(img, target)
                count_teeth += 1

print(f"✔ Total teeth images copied: {count_teeth}\n")


# --------------------------------------------------------------------
# 2. DOWNLOAD NON-TEETH IMAGES (RANDOM)
# --------------------------------------------------------------------
print("📌 Downloading NON-TEETH images (400 images)...")

urls = [
    f"https://picsum.photos/300/300?random={i}"
    for i in range(400)
]

for i, url in enumerate(tqdm(urls)):
    try:
        img_data = requests.get(url, timeout=5).content
        with open(NOT_TEETH_DIR / f"notteeth_{i}.jpg", "wb") as f:
            f.write(img_data)
    except Exception as e:
        print("Download failed:", e)

print(f"\n✔ Download complete — saved to: {NOT_TEETH_DIR}")


# --------------------------------------------------------------------
# DONE
# --------------------------------------------------------------------
print("\n---------------------------------------------")
print(" DATASET SUCCESSFULLY GENERATED!")
print("---------------------------------------------")
print("Teeth images folder:     ", TEETH_DIR)
print("Non-teeth images folder: ", NOT_TEETH_DIR)
print("---------------------------------------------\n")
