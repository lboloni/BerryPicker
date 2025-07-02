import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import wandb

# ===== Determine run name based on current script file =====
run_name = os.path.splitext(os.path.basename(__file__))[0]

# ===== Initialize wandb =====
wandb.init(project="leaf-warmup", name=run_name, config={
    "lr": 1e-3,
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "Adam",
    "loss": "BCE",
    "architecture": "SlidingWindow3DCNN",
    "early_stopping_patience": 10
})
config = wandb.config

# ===== Set random seed for reproducibility =====
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===== Paths =====
data_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids\pre"
labels_dir = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\videos2\labels2"

# ===== Hyperparameters =====
batch_size = config.batch_size
lr = config.lr
epochs = config.epochs
patience = config.early_stopping_patience
window_size = 7  # 3 before, current, 3 after
half_window = window_size // 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset Class =====
class SlidingWindowDataset(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None):
        self.entries = []
        self.transform = transform

        npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        for npz_file in npz_files:
            base_name = os.path.splitext(npz_file)[0]
            npz_path = os.path.join(data_dir, npz_file)
            label_path = os.path.join(labels_dir, f"{base_name}.npy")
            if not os.path.exists(label_path):
                continue

            labels = np.load(label_path)
            npz_data = np.load(npz_path)
            frames = None
            for key in npz_data:
                frames = npz_data[key]
                break
            if frames is None:
                continue

            assert frames.shape[0] == labels.shape[0], f"Frame-label mismatch in {base_name}"

            # Store per-frame entries with window extraction logic
            for i in range(labels.shape[0]):
                self.entries.append( (frames, labels, i) )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        frames, labels, center_idx = self.entries[idx]
        num_frames, H, W, C = frames.shape

        window_imgs = []
        for offset in range(-half_window, half_window+1):
            frame_idx = center_idx + offset
            if 0 <= frame_idx < num_frames:
                img = frames[frame_idx].astype(np.uint8)
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
            else:
                # Empty frame (zeros)
                img = torch.zeros(3, H, W)
            window_imgs.append(img)

        # window_imgs: list of (C,H,W), stack to (window_size, C, H, W), then permute to (C, window_size, H, W)
        window_imgs = torch.stack(window_imgs)  # (window_size, C, H, W)
        window_imgs = window_imgs.permute(1, 0, 2, 3)  # (C, window_size, H, W)

        label = torch.tensor([labels[center_idx]], dtype=torch.float32)
        return window_imgs, label

# ===== Transform (no resize) =====
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ===== Load dataset =====
dataset = SlidingWindowDataset(data_dir, labels_dir, transform=transform)

# ===== Train-validation split =====
generator = torch.Generator().manual_seed(seed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ===== Sliding Window 3D CNN Model =====
class SlidingWindow3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Flatten()
        )
        self.fc1 = nn.Linear(64 * window_size * (256//8) * (128//8), 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)  # (B, features)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SlidingWindow3DCNN().to(device)

# ===== Loss and Optimizer =====
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===== Early stopping variables =====
best_val_acc = 0
best_epoch = -1
epochs_no_improve = 0

# ===== Training Loop =====
for epoch in range(epochs):
    model.train()
    total_loss, total_samples, correct_preds = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        preds = (outputs >= 0.5).float()
        correct_preds += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = correct_preds / total_samples if total_samples > 0 else 0

    # ===== Validation =====
    model.eval()
    val_loss, val_samples, val_correct_preds = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_samples += labels.size(0)

            preds = (outputs >= 0.5).float()
            val_correct_preds += (preds == labels).sum().item()

    avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
    avg_val_acc = val_correct_preds / val_samples if val_samples > 0 else 0

    # ===== Log to wandb =====
    wandb.log({
        "Train Loss": avg_loss,
        "Train Acc": avg_acc,
        "Val Loss": avg_val_loss,
        "Val Acc": avg_val_acc,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")

    # ===== Early stopping check =====
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        best_epoch = epoch + 1
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
            break

# ===== Final report =====
print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
wandb.log({
    "Best Val Acc": best_val_acc,
    "Best Epoch": best_epoch
})