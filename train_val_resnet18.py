import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders_resnet18
from tqdm import tqdm
import torchvision.models as models
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

checkpoint_dir = "./checkpoint"
checkpoint_resnet18_dir = "./checkpoint/resnet18"
result_dir = "./checkpoint/resnet18/result"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc="Training", leave=True)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct/total)

    return running_loss / total, correct / total

def evaluate(model, data_loader, criterion, device, mode="Evaluating"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(data_loader, desc=mode, leave=True)

    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)

    return running_loss / total, correct / total

if __name__ == "__main__":
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_resnet18_dir):
        shutil.rmtree(checkpoint_resnet18_dir)
    os.makedirs(checkpoint_resnet18_dir)

    train_loader, val_loader, _, class_names = get_data_loaders_resnet18("dataset_split", input_size=224, batch_size=32)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),                          # Thêm Dropout trước fully-connected layer
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 8
    best_val_acc = 0.0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, mode="Validating")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        model_path = f"{checkpoint_resnet18_dir}/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_resnet18_dir}/best.pth")

    os.makedirs(f"{checkpoint_resnet18_dir}/result", exist_ok=True)

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")
    plt.savefig(f"{result_dir}/accuracy.png")

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.savefig(f"{result_dir}/loss.png")