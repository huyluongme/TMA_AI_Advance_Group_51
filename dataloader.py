from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, input_size=224, batch_size=32, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"

    # Load the full training dataset
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # Calculate split sizes
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Test dataset and loader
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Return class names from the full train dataset
    return train_loader, val_loader, test_loader, full_train_dataset.classes
