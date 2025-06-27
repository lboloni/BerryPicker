import os
import json
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
import torch.nn.functional as F

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

# Dataset
dataset = LeafFrameDataset(labels, frames_dir=frames_dir)

# Calculate class weights for imbalanced dataset
labels_list = [label for _, label in labels.items()]
class_counts = Counter(labels_list)
num_classes = 3
total_count = sum(class_counts.values())

weights = []
for i in range(num_classes):
    count = class_counts.get(i, 0)
    if count > 0:
        weights.append(total_count / (num_classes * count))
    else:
        weights.append(0.0)  # Avoid division by zero

class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

# Weighted sampler to balance batches
sample_weights = [class_weights[label].item() for _, label in dataset.entries]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader with sampler
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
criterion = FocalLoss(alpha=class_weights, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop with per-class accuracy
num_epochs = 10

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
