import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set fixed random seed for reproducibility
torch.manual_seed(42)

# Pathing and labels
labels_path = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips\labels.json"
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Dataset class for 3D inputs
class LeafDataset(Dataset):
    def __init__(self, labels_dict, indices=None):
        self.entries = list(labels_dict.items())
        if indices is not None:
            self.entries = [self.entries[i] for i in indices]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        clip_name, info = self.entries[idx]
        npy_path = info["npy_path"]
        label = info["label"]
        data = np.load(npy_path)  # shape: (7, 640, 360, 3)
        # Transpose to (D, H, W, C) -> (D, C, H, W)
        data = data.transpose(0, 3, 1, 2).astype(np.float32)  # (7, 3, 640, 360)
        data = torch.from_numpy(data)

        # Desired shape
        target_shape0 = 59

        # Compute padding needed
        pad_size = target_shape0 - data.shape[0]

        if pad_size > 0:
            # Create padding tensor of zeros with matching dtype and device
            padding = torch.zeros(
                pad_size, *data.shape[1:],
                dtype=data.dtype, device=data.device
            )
            # Concatenate along dim=0
            data = torch.cat([data, padding], dim=0)
        else:
            data = data  # No padding needed

        return data, torch.tensor(label, dtype=torch.float32)

# Prepare train/test splits
all_indices = list(range(len(labels)))
all_labels = [labels[clip]["label"] for clip in labels.keys()]
train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, stratify=all_labels, random_state=42
)

train_dataset = LeafDataset(labels, indices=train_indices)
test_dataset = LeafDataset(labels, indices=test_indices)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
train_loader_eval = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 3D CNN Model for binary classification
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 4, kernel_size=3, stride=1, padding=1),  # (B,16,D,H,W)
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,16,D,H/2,W/2)
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,32,D,H/4,W/4)
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,32,D,H/4,W/4)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,32,D,H/4,W/4)
            nn.Conv3d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,32,D,H/4,W/4)
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # (B,32,D,H/4,W/4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten to (B,64*D)
            nn.Linear(2950, 1024),  # assuming D=7
            nn.ReLU(),
            nn.Linear(1024, 512),  # assuming D=7
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Input x: (B, D, C, H, W)
        # Permute to (B, C, D, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_train_acc = 0
best_model_state = None
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels_batch in train_loader:
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate on train set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels_batch in train_loader_eval:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds.float() == labels_batch).sum().item()
            total += labels_batch.size(0)
    train_acc = correct / total

    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_model_state = model.state_dict()

# Evaluate best model on train set
model.load_state_dict(best_model_state)
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for inputs, labels_batch in train_loader_eval:
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        test_loss += loss.item()
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == labels_batch).sum().item()
        total += labels_batch.size(0)
print(f"Train Loss: {test_loss / len(train_loader_eval):.4f}, Train Accuracy: {int((correct / total)*100)}%")

# Evaluate on test set
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for inputs, labels_batch in test_loader:
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        test_loss += loss.item()
        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == labels_batch).sum().item()
        total += labels_batch.size(0)
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {int((correct / total)*100)}%")
