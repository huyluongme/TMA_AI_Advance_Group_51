# file name: split_dataset.py
import os
import shutil
import random
from tqdm import tqdm

# Configuration
SOURCE_DIR = 'PlantVillage-Dataset/raw/color'
TARGET_DIR = 'PlantVillage-Dataset/raw/split'
TRAIN_DIR = os.path.join(TARGET_DIR, 'train')
TEST_DIR = os.path.join(TARGET_DIR, 'test')
TRAIN_RATIO = 0.8 #edit if needed

# Ensure target directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Process each class folder in the color dataset
for class_name in tqdm(os.listdir(SOURCE_DIR), desc="Splitting dataset by class"):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue  # skip non-folder items

    # List all image files in the class folder
    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(image_files) == 0:
        continue  # skip empty class folders

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    # Create output subfolders
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Copy files to train
    for file in train_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(TRAIN_DIR, class_name, file)
        shutil.copy2(src, dst)

    # Copy files to test
    for file in test_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(TEST_DIR, class_name, file)
        shutil.copy2(src, dst)

print("\n✅ Dataset successfully split into training and testing folders.")
