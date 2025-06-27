import os
import json
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set fixed random seed for reproducibility
torch.manual_seed(42)

# Pathing and labels
image_dir = "../data/leaf_imgs1/pre/"
labels_path = os.path.join(image_dir, "labels.json")
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Initialize the dataset class
class LeafDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None, indices=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_names = list(labels.keys())
        if indices is not None:
            self.image_names = [self.image_names[i] for i in indices]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[img_name], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Preparing the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
# Stratified split
all_indices = list(range(len(labels)))
all_labels = [labels[name] for name in labels.keys()]
train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, stratify=all_labels, random_state=42
)
train_dataset = LeafDataset(image_dir, labels, transform=transform, indices=train_indices)
test_dataset = LeafDataset(image_dir, labels, transform=transform, indices=test_indices)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
train_loader_eval = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Simple CNN model for binary classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*2, 512*4),
            nn.ReLU(),
            nn.Linear(512*4, 512*2),
            nn.ReLU(),
            nn.Linear(512*2, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initializing the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
best_train_acc = 0
best_model_state = None
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels_batch in train_loader_eval:
            images = images.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            outputs = model(images)
            preds = torch.sigmoid(outputs) >= 0.5
            correct += (preds.float() == labels_batch).sum().item()
            total += labels_batch.size(0)
    train_acc = correct / total

    # Save best model if improved
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_model_state = model.state_dict()

print()

# Evaluating on the train set
model.load_state_dict(best_model_state)
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels_batch in train_loader_eval:
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        test_loss += loss.item()

        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == labels_batch).sum().item()
        total += labels_batch.size(0)
print(f"Train Loss: {test_loss / len(test_loader):.4f}, Train Accuracy: {int((correct / total)*100)}%")

# Evaluating on the test set
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        test_loss += loss.item()

        preds = torch.sigmoid(outputs) >= 0.5
        correct += (preds.float() == labels_batch).sum().item()
        total += labels_batch.size(0)
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {int((correct / total)*100)}%")