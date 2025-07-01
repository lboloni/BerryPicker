# leaf_count_classifier_padded_int_wandb.py

import os
import csv
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="leaf_count_classifier", name="padded_int_run", config={
    "batch_size": 3,
    "lr": 1e-3,
    "epochs": 30,
    "num_classes": 31,
    "model": "LeafClassifier_padded_int"
})
config = wandb.config

# Paths
labels_csv = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\videos\labels.csv"
npy_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids2\pre"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class with dynamic padding
class LeafVideoDataset(Dataset):
    def __init__(self, labels_csv, npy_dir):
        self.entries = []
        self.max_len = 0

        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name']
                leaves = int(round(float(row['leaves'])))
                diseased = int(round(float(row['diseased'])))
                npy_path = os.path.join(npy_dir, name + ".npz")
                data = np.load(npy_path)['frames']
                self.max_len = max(self.max_len, data.shape[0])
                self.entries.append((npy_path, [leaves, diseased]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        npy_path, targets = self.entries[idx]
        data = np.load(npy_path)['frames']
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)

        pad_len = self.max_len - data.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, *data.shape[1:]), dtype=torch.float32)
            data = torch.cat([data, pad_tensor], dim=0)

        targets = torch.tensor(targets, dtype=torch.long)
        return data, targets

# Classification model
class LeafClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LeafClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 32)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 16))
        )
        self.fc_leaves = nn.Sequential(
            nn.Linear(32 * 32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.fc_diseased = nn.Sequential(
            nn.Linear(32 * 32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.cnn(x)
        x = x.view(B, N, -1).mean(dim=1)
        out_leaves = self.fc_leaves(x)
        out_diseased = self.fc_diseased(x)
        return out_leaves, out_diseased

# Prepare dataset
dataset = LeafVideoDataset(labels_csv, npy_dir)

# Train-test split (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size)

# Model, loss, optimizer
model = LeafClassifier(config.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Training loop
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Training"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_leaves, outputs_diseased = model(data)

        loss_leaves = criterion(outputs_leaves, targets[:,0])
        loss_diseased = criterion(outputs_diseased, targets[:,1])
        loss = loss_leaves + loss_diseased

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)

    # Training accuracy evaluation
    model.eval()
    train_leaves_correct = 0
    train_diseased_correct = 0
    train_exact_matches = 0
    train_total = 0
    with torch.no_grad():
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            outputs_leaves, outputs_diseased = model(data)

            preds_leaves = outputs_leaves.argmax(dim=1)
            preds_diseased = outputs_diseased.argmax(dim=1)

            train_leaves_correct += (preds_leaves == targets[:,0]).sum().item()
            train_diseased_correct += (preds_diseased == targets[:,1]).sum().item()
            train_exact_matches += ((preds_leaves == targets[:,0]) & (preds_diseased == targets[:,1])).sum().item()
            train_total += data.size(0)

    train_leaves_acc = train_leaves_correct / train_total * 100
    train_diseased_acc = train_diseased_correct / train_total * 100
    train_exact_acc = train_exact_matches / train_total * 100

    # Validation
    val_loss = 0.0
    val_leaves_correct = 0
    val_diseased_correct = 0
    val_exact_matches = 0
    val_total = 0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} - Validation"):
            data, targets = data.to(device), targets.to(device)
            outputs_leaves, outputs_diseased = model(data)

            loss_leaves = criterion(outputs_leaves, targets[:,0])
            loss_diseased = criterion(outputs_diseased, targets[:,1])
            loss = loss_leaves + loss_diseased

            val_loss += loss.item() * data.size(0)

            preds_leaves = outputs_leaves.argmax(dim=1)
            preds_diseased = outputs_diseased.argmax(dim=1)

            val_leaves_correct += (preds_leaves == targets[:,0]).sum().item()
            val_diseased_correct += (preds_diseased == targets[:,1]).sum().item()
            val_exact_matches += ((preds_leaves == targets[:,0]) & (preds_diseased == targets[:,1])).sum().item()
            val_total += data.size(0)

    val_loss /= len(val_loader.dataset)
    val_leaves_acc = val_leaves_correct / val_total * 100
    val_diseased_acc = val_diseased_correct / val_total * 100
    val_exact_acc = val_exact_matches / val_total * 100

    # Logging to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_leaves_acc": train_leaves_acc,
        "train_diseased_acc": train_diseased_acc,
        "train_exact_acc": train_exact_acc,
        "val_loss": val_loss,
        "val_leaves_acc": val_leaves_acc,
        "val_diseased_acc": val_diseased_acc,
        "val_exact_acc": val_exact_acc
    })

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Leaves Acc: {train_leaves_acc:.2f}% | "
          f"Train Diseased Acc: {train_diseased_acc:.2f}% | Train Exact Match Acc: {train_exact_acc:.2f}% || "
          f"Val Loss: {val_loss:.4f} | Val Leaves Acc: {val_leaves_acc:.2f}% | "
          f"Val Diseased Acc: {val_diseased_acc:.2f}% | Val Exact Match Acc: {val_exact_acc:.2f}%")

print("Training complete.")
wandb.finish()
