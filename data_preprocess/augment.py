import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch

# Configuration
SOURCE_DIR = 'PlantVillage-Dataset/raw/split/train'  # Apply augmentation only on training set
AUGMENTATIONS_PER_IMAGE = 3

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define PyTorch-style data augmentations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
])

# Loop through all class folders
for class_name in tqdm(os.listdir(SOURCE_DIR), desc="Augmenting dataset"):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('aug_')]

    for file_name in image_files:
        file_path = os.path.join(class_path, file_name)

        try:
            image = Image.open(file_path).convert('RGB')
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

            for i in range(AUGMENTATIONS_PER_IMAGE):
                # Note: transforms in torchvision don't operate on GPU tensors directly
                augmented_image = transform(image)
                aug_filename = f"aug_{i+1}_{file_name}"
                aug_path = os.path.join(class_path, aug_filename)
                augmented_image.save(aug_path)

        except Exception as e:
            print(f"Error augmenting {file_path}: {e}")

print("\n✅ Data augmentation complete using PyTorch and GPU (if available).")
