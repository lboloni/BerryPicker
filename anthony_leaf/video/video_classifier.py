import os
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# Paths
frames_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\vid1"
labels_path = os.path.join(frames_dir, "labels.json")

# Load labels
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class LeafFrameDataset(Dataset):
    def __init__(self, labels_dict, frames_dir, transform=None):
        self.entries = list(labels_dict.items())
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path, label = self.entries[idx]
        img_path = os.path.join(self.frames_dir, os.path.basename(rel_path))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Dataset and DataLoader
dataset = LeafFrameDataset(labels, frames_dir=frames_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
class LeafCNN(nn.Module):
    def __init__(self):
        super(LeafCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = LeafCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop with per-class accuracy
num_epochs = 10
num_classes = 3

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

        # Per-class accuracy
        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

    train_loss /= len(dataset)
    train_acc = train_correct / len(dataset)

    print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
    print("Per-class accuracy:")
    for cls in range(num_classes):
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls]
            print(f"  Class {cls}: {acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
        else:
            print(f"  Class {cls}: No samples")

print("Training complete.")
