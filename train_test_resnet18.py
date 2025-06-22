import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_data_loaders
from tqdm import tqdm
import torchvision.models as models
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR

DATASET_DIR = './data_preprocess/PlantVillage-Dataset/raw/split'
CHECKPOINT_DIR = "./checkpoint"
SUB_TEST = "_1"
CHECKPOINT_RESNET18_DIR = f"{CHECKPOINT_DIR}/resnet18{SUB_TEST}"
TRAIN_RESULT_DIR = f"{CHECKPOINT_RESNET18_DIR}/train"
TEST_RESULT_DIR = f"{CHECKPOINT_RESNET18_DIR}/test"

EPOCH = 15
BATCH_SIZE = 32 # SUB_TEST _1 is 64
LEARNING_RATE = 0.001
INPUT_SIZE = 224

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

def test(test_loader, class_names):
    best_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    best_model.fc = nn.Sequential(
        nn.Linear(best_model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(class_names))
    )
    best_model.load_state_dict(torch.load(f"{TRAIN_RESULT_DIR}/best.pth", map_location=device))
    best_model = best_model.to(device)
    best_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join(TEST_RESULT_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=np.arange(len(class_names)), columns=np.arange(len(class_names)))

    plt.figure(figsize=(36, 24))
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 24}, cbar=False)
    plt.title("Confusion Matrix (Label = Class Index)", fontsize=24)
    plt.ylabel("Actual (Index)", fontsize=22)
    plt.xlabel("Predicted (Index)", fontsize=22)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(rotation=0, fontsize=22)

    # Legend mapping: class index → class name
    items_per_row = 4
    legend_items = [f"{i} = {name}" for i, name in enumerate(class_names)]
    legend_lines = [legend_items[i:i + items_per_row] for i in range(0, len(legend_items), items_per_row)]

    for idx, line_items in enumerate(legend_lines):
        line_text = "      ".join(line_items)
        plt.figtext(
            0.01, 0.02 - idx * 0.025,
            line_text,
            fontsize=24,
            ha='left',
            va='top',
            family='monospace'
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(TEST_RESULT_DIR, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_RESNET18_DIR):
        shutil.rmtree(CHECKPOINT_RESNET18_DIR)
    os.makedirs(TRAIN_RESULT_DIR)
    os.makedirs(TEST_RESULT_DIR)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(DATASET_DIR, input_size=INPUT_SIZE, batch_size=BATCH_SIZE)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    train_summary = {
        "model": "ResNet18",
        "num_epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "input_size": INPUT_SIZE,
        "num_classes": len(class_names),
        "device": str(device),
        "results": []
    }

    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch+1}/{EPOCH}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, mode="Validating")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        model_path = f"{TRAIN_RESULT_DIR}/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{TRAIN_RESULT_DIR}/best.pth")
        
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_losses[epoch],
            "train_acc": train_accs[epoch],
            "val_loss": val_losses[epoch],
            "val_acc": val_accs[epoch]
        }
        train_summary["results"].append(epoch_result)

        scheduler.step()

    os.makedirs(TRAIN_RESULT_DIR, exist_ok=True)

    json_path = os.path.join(TRAIN_RESULT_DIR, "train_result.json")
    with open(json_path, "w") as f:
        json.dump(train_summary, f, indent=4)
    
    plt.figure()
    plt.plot(range(1, EPOCH+1), train_accs, label='Train Acc')
    plt.plot(range(1, EPOCH+1), val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")
    plt.savefig(f"{TRAIN_RESULT_DIR}/accuracy.png")

    plt.figure()
    plt.plot(range(1, EPOCH+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCH+1), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.savefig(f"{TRAIN_RESULT_DIR}/loss.png")

    test(test_loader, class_names)