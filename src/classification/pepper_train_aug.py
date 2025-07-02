import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# ----- Parameters -----
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Load JSON -----
with open('src/classification/pepper_balanced_split.json', 'r') as f:
    data = json.load(f)

# ----- Custom Dataset -----
class PepperDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path = self.entries[idx]['path']
        label = self.entries[idx]['label']
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label

# ----- Augmentation Transforms -----
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value='random'),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ----- Dataloaders -----
train_dataset = PepperDataset(data['train'], transform=augment_transform)
test_dataset = PepperDataset(data['test'], transform=augment_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ----- Simple CNN -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)

# ----- Training -----
model = SimpleCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# ----- Evaluation -----
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images).cpu()
        preds = (outputs > 0.5).int()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.2%}")
