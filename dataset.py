from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders_resnet18(data_dir, input_size=224, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    test_dir = f"{data_dir}/test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.classes
