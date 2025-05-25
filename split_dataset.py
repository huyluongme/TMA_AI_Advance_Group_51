import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    class_names = os.listdir(source_dir)
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=seed)

        for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for img in tqdm(split_imgs, desc=f"Copying {split}/{class_name}", leave=False):
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(split_dir, img)
                shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    source = './dataset'
    destination = './dataset_split'
    if os.path.exists(destination):
        shutil.rmtree(destination)
    split_dataset(source, destination)
