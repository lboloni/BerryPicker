import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import argparse
import time

# ===== Command line argument parsing =====
parser = argparse.ArgumentParser(description="Leaf video training script")
parser.add_argument("--wandb", action="store_true", help="Activate wandb logging")
args = parser.parse_args()
use_wandb = args.wandb

if use_wandb:
    import wandb

# ===== Determine run name based on current script file =====
run_name = os.path.splitext(os.path.basename(__file__))[0]

# ===== Initialize wandb if enabled =====
if use_wandb:
    wandb.init(project="leaf-warmup", name=run_name, config={
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 1,
        "optimizer": "Adam",
        "loss": "BCE",
        "architecture": "FrameCNN",
        "early_stopping_patience": 10
    })
    config = wandb.config
else:
    config = {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 1,
        "early_stopping_patience": 10
    }

# ===== Set random seed for reproducibility =====
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===== Paths =====
data_dir = "/data/breast-cancer/leaf_vids/pre"
labels_dir = "anthony_leaf/labels2"

# ===== Hyperparameters =====
batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
patience = 10  # stops after 10 consecutive failures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset Class =====
class VideoDataset(Dataset):
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
            self.entries.append( (frames, labels) )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        frames, labels = self.entries[idx]

        images = []
        for frame in frames:
            img = frame.astype(np.uint8)
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)  # (num_frames, C, H, W)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (num_frames, 1)
        return images, labels

# ===== Transform (no resize) =====
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ===== Load dataset =====
dataset = VideoDataset(data_dir, labels_dir, transform=transform)

# ===== Train-validation split =====
generator = torch.Generator().manual_seed(seed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ===== Model =====
class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16

            nn.Flatten(),
            nn.Linear(64 * 32 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = FrameCNN().to(device)

# ===== Loss and Optimizer =====
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===== Early stopping variables =====
best_val_acc = 0
best_epoch = -1
epochs_no_improve = 0

# ===== Training Loop =====
for epoch in range(epochs):
    start_time = time.time()

    model.train()
    total_loss, total_frames, correct_preds = 0, 0, 0

    for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        videos = videos.squeeze(0).to(device)
        labels = labels.squeeze(0).to(device)

        outputs = model(videos)

        mask = (labels != -1)
        outputs_masked = outputs[mask]
        labels_masked = labels[mask]

        if labels_masked.numel() == 0:
            continue

        loss = criterion(outputs_masked, labels_masked)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels_masked.size(0)
        total_frames += labels_masked.size(0)

        preds = (outputs_masked >= 0.5).float()
        correct_preds += (preds == labels_masked).sum().item()

    avg_loss = total_loss / total_frames if total_frames > 0 else 0
    avg_acc = correct_preds / total_frames if total_frames > 0 else 0

    # ===== Validation =====
    model.eval()
    val_loss, val_frames, val_correct_preds = 0, 0, 0

    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            videos = videos.squeeze(0).to(device)
            labels = labels.squeeze(0).to(device)

            outputs = model(videos)

            mask = (labels != -1)
            outputs_masked = outputs[mask]
            labels_masked = labels[mask]

            if labels_masked.numel() == 0:
                continue

            loss = criterion(outputs_masked, labels_masked)
            loss = loss.mean()

            val_loss += loss.item() * labels_masked.size(0)
            val_frames += labels_masked.size(0)

            preds = (outputs_masked >= 0.5).float()
            val_correct_preds += (preds == labels_masked).sum().item()

    avg_val_loss = val_loss / val_frames if val_frames > 0 else 0
    avg_val_acc = val_correct_preds / val_frames if val_frames > 0 else 0

    # ===== GPU memory and runtime tracking =====
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # bytes to MB
        torch.cuda.reset_peak_memory_stats(device=device)
    else:
        gpu_mem_mb = 0

    epoch_time = time.time() - start_time

    # ===== Log to wandb if enabled =====
    if use_wandb:
        wandb.log({
            "Train Loss": avg_loss,
            "Train Acc": avg_acc,
            "Val Loss": avg_val_loss,
            "Val Acc": avg_val_acc,
            "GPU Memory MB": gpu_mem_mb,
            "Epoch Time (s)": epoch_time,
            "epoch": epoch + 1
        })

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}, GPU Mem={gpu_mem_mb:.2f}MB, Time={epoch_time:.2f}s")

    # ===== Early stopping check =====
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        best_epoch = epoch + 1
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
            break

# ===== Final report =====
print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
if use_wandb:
    wandb.log({
        "Best Val Acc": best_val_acc,
        "Best Epoch": best_epoch
    })
