#!/usr/bin/env python
# coding: utf-8

# # Train a proprioception-tuned Multi-View Vision Transformer (ViT)
# We create a sensor processing model using multiple Vision Transformer (ViT) based visual encoders
# finetuned with proprioception.
# We start with pretrained ViT models, then train them to:
# 1. Create a meaningful 128-dimensional latent representation from multiple camera views
# 2. Learn to map this representation to robot positions (proprioception)
#  The sensor processing object associated with the trained model is in sensorprocessing/sp_vit_multiview.py
# 

# In[ ]:


#
# The sensor processing object associated with the trained model is in
# sensorprocessing/sp_vit_multiview.py

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from demonstration.demonstration import Demonstration
from sensorprocessing.sp_vit_multiview import MultiViewVitSensorProcessing
from robot.al5d_position_controller import RobotPosition
import sensorprocessing.sp_helper as sp_helper


# In[ ]:


# =============================================================================
# PAPERMILL PARAMETERS
# =============================================================================

# If it is set to discard-old, the exprun will be recreated from scratch
creation_style = "exist-ok"

experiment = "sensorprocessing_propriotuned_Vit_multiview"
run = "vit_base_multiview"
#concat_proj
# run = "vit_base_multiview"  # ViT Base
# run = "vit_large_multiview" # ViT Large
# run = "vit_huge_multiview" # ViT Huge

##  indiv_proj
# run = "vit_base_multiview_indiv_proj"  # ViT Base_indiv_proj
# run = "vit_large_multiview_indiv_proj" # ViT Large_indiv_proj
# run = "vit_huge_multiview_indiv_proj" # ViT Huge


##  attention
# run = "vit_base_multiview_attention"  # ViT Base_attention
# run = "vit_large_multiview_attention" # ViT Large_attention
# run = "vit_huge_multiview_attention" # ViT Huge_attention

##  weighted_sum
# run = "vit_base_multiview_weighted_sum"  # ViT Base_weighted_sum
# run = "vit_large_multiview_weighted_sum" # ViT Large_weighted_sum
# run = "vit_huge_multiview_weighted_sum" # ViT Huge_weighted_sum

##  gated
# run = "vit_base_multiview_gated"  # ViT Base_gated
# run = "vit_large_multiview_gated" # ViT Large_gated
# run = "vit_huge_multiview_gated" # ViT Huge_gated
epochs = None

# If not None, set an external experiment path
external_path = None
# If not None, set an output path
data_path = None


# In[ ]:


# =============================================================================
# INITIALIZATION
# =============================================================================

# Handle external paths (when called from Flow)
if external_path:
    external_path = pathlib.Path(external_path).expanduser()
    external_path.mkdir(parents=True, exist_ok=True)
    Config().set_exprun_path(external_path)
    Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")

if data_path:
    data_path = pathlib.Path(data_path).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    Config().set_results_path(data_path)

# Load the experiment/run configuration
exp = Config().get_experiment(experiment, run, creation_style=creation_style)
exp_robot = Config().get_experiment(exp["robot_exp"], exp["robot_run"])

# Create output directory if it doesn't exist
data_dir = pathlib.Path(exp["data_dir"])
data_dir.mkdir(parents=True, exist_ok=True)
print(f"Data directory: {data_dir}")


# In[ ]:


# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================

def load_multiview_images_as_proprioception_training(exp, exp_robot):
    """Loads the training images specified in the exp/run for multiple views.

    Processes them as tensors for multiview proprioception training.
    Caches the processed results into the input and target file.
    Remove those files to recalculate.

    Args:
        exp: Experiment configuration
        exp_robot: Robot experiment for normalization

    Returns:
        Dictionary with view_inputs (list of tensors) and targets
    """
    retval = {}
    proprioception_input_path = pathlib.Path(exp.data_dir(), "proprio_input_multiview.pth")
    proprioception_target_path = pathlib.Path(exp.data_dir(), "proprio_target_multiview.pth")

    if proprioception_input_path.exists():
        print(f"Loading cached multiview data from {proprioception_input_path}")
        retval["view_inputs"] = torch.load(proprioception_input_path, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_path, weights_only=True)
    else:
        view_lists = {}  # Dictionary to organize views by camera
        targetlist = []
        num_views = exp.get("num_views", 2)
        transform = sp_helper.get_transform_to_sp(exp)

        print(f"Loading multiview training data from demonstrations with {num_views} views...")

        for val in exp["training_data"]:
            run_name, demo_name, cameras = val  # cameras is now a list
            exp_demo = Config().get_experiment("demonstration", run_name)
            demo = Demonstration(exp_demo, demo_name)

            # Initialize view lists for cameras
            if not view_lists:
                for camera in cameras[:num_views]:
                    view_lists[camera] = []

            for i in range(demo.metadata["maxsteps"]):
                # Get images from all cameras
                frame_images = []
                skip_frame = False

                for camera in cameras[:num_views]:
                    try:
                        sensor_readings, _ = demo.get_image(
                            i, device=device, transform=transform, camera=camera
                        )
                        frame_images.append(sensor_readings[0])
                    except Exception as e:
                        print(f"Skipping frame {i} - missing camera {camera}: {e}")
                        skip_frame = True
                        break

                if skip_frame:
                    continue

                # Store images by camera
                for camera, img in zip(cameras[:num_views], frame_images):
                    view_lists[camera].append(img)

                # Get robot position
                rp = demo.get_action(i, "rc-position-target", exp_robot)
                anorm = rp.to_normalized_vector(exp_robot)
                targetlist.append(torch.from_numpy(anorm))

        # Ensure we have the same number of frames for each view
        min_frames = min(len(view_list) for view_list in view_lists.values())
        if min_frames < len(targetlist):
            print(f"Truncating dataset to {min_frames} frames (from {len(targetlist)})")
            targetlist = targetlist[:min_frames]
            for camera in view_lists:
                view_lists[camera] = view_lists[camera][:min_frames]

        # Stack tensors for each view
        view_tensors = []
        for camera in sorted(view_lists.keys())[:num_views]:
            view_tensors.append(torch.stack(view_lists[camera]))

        retval["view_inputs"] = view_tensors
        retval["targets"] = torch.stack(targetlist)

        # Save processed data
        torch.save(retval["view_inputs"], proprioception_input_path)
        torch.save(retval["targets"], proprioception_target_path)
        print(f"Saved {len(targetlist)} training examples with {num_views} views each")

    # Separate the training and validation data
    length = len(retval["targets"])
    rows = torch.randperm(length)

    # Shuffle targets
    shuffled_targets = retval["targets"][rows]

    # Shuffle each view input using the same row indices
    shuffled_view_inputs = []
    for view_tensor in retval["view_inputs"]:
        shuffled_view_inputs.append(view_tensor[rows])

    # Split into training (67%) and validation (33%) sets
    training_size = int(length * 0.67)

    # Training data
    retval["view_inputs_training"] = [view[:training_size] for view in shuffled_view_inputs]
    retval["targets_training"] = shuffled_targets[:training_size]

    # Validation data
    retval["view_inputs_validation"] = [view[training_size:] for view in shuffled_view_inputs]
    retval["targets_validation"] = shuffled_targets[training_size:]

    print(f"Created {training_size} training examples and {length - training_size} validation examples")
    return retval


# In[ ]:


# =============================================================================
# CUSTOM DATASET FOR MULTI-VIEW DATA
# =============================================================================

class MultiViewDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for multi-view image data."""

    def __init__(self, view_inputs, targets):
        """
        Args:
            view_inputs: List of tensors, one per view. Each tensor: [N, C, H, W]
            targets: Tensor of targets: [N, output_dim]
        """
        self.view_inputs = view_inputs
        self.targets = targets
        self.num_samples = len(targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Returns tuple: (list of view images, target)"""
        views = [view[idx] for view in self.view_inputs]
        target = self.targets[idx]
        return views, target


def multiview_collate_fn(batch):
    """Custom collate function for multi-view data."""
    views_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    # Transpose from [batch, views] to [views, batch]
    num_views = len(views_list[0])
    batched_views = []
    for v in range(num_views):
        view_batch = torch.stack([views_list[i][v] for i in range(len(views_list))])
        batched_views.append(view_batch)

    return batched_views, targets


# In[ ]:


# =============================================================================
# CHECK FOR EXISTING MODEL AND LOAD DATA IF NEEDED
# =============================================================================

modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])

if modelfile.exists():
    print("*** Train-ProprioTuned-ViT-Multiview ***: NOT training; model already exists")
    # Load data anyway for testing
    tr = load_multiview_images_as_proprioception_training(exp, exp_robot)
else:
    # Load data for training
    tr = load_multiview_images_as_proprioception_training(exp, exp_robot)

view_inputs_training = tr["view_inputs_training"]
targets_training = tr["targets_training"]
view_inputs_validation = tr["view_inputs_validation"]
targets_validation = tr["targets_validation"]


# In[ ]:


# =============================================================================
# CREATE THE MODEL
# =============================================================================

# Create the multi-view ViT model
sp = MultiViewVitSensorProcessing(exp, device)
model = sp.enc  # Get the actual encoder model for training

print("Model created successfully")

try:
    params = model.parameters()
    print("Parameters accessed successfully")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
except Exception as e:
    print(f"Error accessing parameters: {e}")

# Select loss function
loss_type = exp.get('loss', 'MSELoss')
if loss_type == 'MSELoss':
    criterion = nn.MSELoss()
elif loss_type == 'L1Loss':
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()

# Set up optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=exp.get('learning_rate', 0.001),
    weight_decay=exp.get('weight_decay', 0.01)
)

# Learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)


# In[ ]:


# =============================================================================
# CREATE DATALOADERS
# =============================================================================

batch_size = exp.get('batch_size', 32)

train_dataset = MultiViewDataset(view_inputs_training, targets_training)
test_dataset = MultiViewDataset(view_inputs_validation, targets_validation)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=multiview_collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=multiview_collate_fn
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")


# In[ ]:


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_and_save_multiview_proprioception_model(
    model, criterion, optimizer, modelfile,
    device="cpu", epochs=20, scheduler=None, log_interval=1
):
    """Trains and saves the multiview ViT proprioception model.

    Args:
        model: Multi-view ViT model
        criterion: Loss function
        optimizer: Optimizer
        modelfile: Path to save the model
        device: Training device
        epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        log_interval: How often to print logs

    Returns:
        Trained model
    """
    model = model.to(device)

    # Checkpoint directory
    checkpoint_dir = modelfile.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')
    start_epoch = 0

    # Check for existing checkpoints
    def get_epoch_number(checkpoint_file):
        try:
            filename = checkpoint_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                return int(parts[1])
            return 0
        except:
            return 0

    checkpoint_files = list(checkpoint_dir.glob("epoch_*.pth"))
    if checkpoint_files:
        checkpoint_files.sort(key=get_epoch_number)
        latest_checkpoint = checkpoint_files[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    num_epochs = epochs

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_views, batch_y in train_loader:
            # Move views and targets to device
            batch_views = [view.to(device) for view in batch_views]
            batch_y = batch_y.to(device)

            # Forward pass
            predictions = model(batch_views)
            loss = criterion(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / max(batch_count, 1)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for batch_views, batch_y in test_loader:
                batch_views = [view.to(device) for view in batch_views]
                batch_y = batch_y.to(device)

                predictions = model(batch_views)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / max(val_batch_count, 1)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Log progress
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:06d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_loss': best_val_loss
        }, checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_val_loss
            }, best_model_path)
            print(f"  New best model saved with val loss: {best_val_loss:.4f}")

        # Cleanup old checkpoints (keep last 3)
        old_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"), key=get_epoch_number)
        for old_ckpt in old_checkpoints[:-3]:
            old_ckpt.unlink()

    # Load best model for final save
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']+1} "
              f"with loss {best_checkpoint['val_loss']:.4f}")

    # Save to final model file
    torch.save(model.state_dict(), modelfile)
    print(f"Final model saved to {modelfile}")

    return model


# In[ ]:


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

epochs_to_train = exp.get("epochs", 20)

if modelfile.exists() and exp.get("reload_existing_model", True):
    print(f"Loading existing final model from {modelfile}")
    model.load_state_dict(torch.load(modelfile, map_location=device))

    # Evaluate loaded model
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_views, batch_y in test_loader:
            batch_views = [view.to(device) for view in batch_views]
            batch_y = batch_y.to(device)
            predictions = model(batch_views)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Loaded model validation loss: {avg_val_loss:.4f}")
else:
    print(f"Training for {epochs_to_train} epochs")
    model = train_and_save_multiview_proprioception_model(
        model, criterion, optimizer, modelfile,
        device=device, epochs=epochs_to_train, scheduler=lr_scheduler
    )


# In[ ]:


# =============================================================================
# TESTING THE TRAINED MODEL
# =============================================================================

# Create sensor processing module using trained model
sp = MultiViewVitSensorProcessing(exp, device)

def test_multiview_sensor_processing(sp, test_view_inputs, test_targets, n_samples=5):
    """Test the multi-view sensor processing module on a few examples."""
    if n_samples > len(test_targets):
        n_samples = len(test_targets)

    indices = torch.randperm(len(test_targets))[:n_samples]

    print("\nTesting multi-view sensor processing on random examples:")
    print("-" * 60)

    for i, idx in enumerate(indices):
        views = [view[idx].unsqueeze(0).to(device) for view in test_view_inputs]
        target = test_targets[idx].cpu().numpy()

        latent = sp.process(views)

        print(f"Example {i+1}:")
        for j, view in enumerate(views):
            print(f"  View {j+1} shape: {view.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Target position: {target}")
        print()

test_multiview_sensor_processing(sp, view_inputs_validation, targets_validation)


# In[ ]:


# =============================================================================
# VERIFY MODEL ENCODING AND FORWARD METHODS
# =============================================================================

model.eval()
with torch.no_grad():
    sample_views = [view[0].unsqueeze(0).to(device) for view in view_inputs_validation]

    # Get latent representation
    latent = model.encode(sample_views)
    print(f"Latent representation shape: {latent.shape}")

    # Get robot position prediction
    position = model.forward(sample_views)
    print(f"Robot position prediction shape: {position.shape}")

    # Verify dimensions
    expected_latent_size = exp["latent_size"]
    assert latent.shape[1] == expected_latent_size, \
        f"Expected latent size {expected_latent_size}, got {latent.shape[1]}"

    expected_output_size = exp["output_size"]
    assert position.shape[1] == expected_output_size, \
        f"Expected output size {expected_output_size}, got {position.shape[1]}"

    print("Verification successful!")


# In[ ]:


# =============================================================================
# SAVE FINAL MODEL AND SUMMARY
# =============================================================================

final_modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
torch.save(model.state_dict(), final_modelfile)
print(f"Model saved to {final_modelfile}")

print("\nTraining complete!")
print(f"Vision Transformer type: {exp['vit_model']}")
print(f"Number of views: {exp.get('num_views', 2)}")
print(f"Fusion method: {exp.get('fusion_type', 'concat_proj')}")
print(f"Latent space dimension: {exp['latent_size']}")
print(f"Output dimension (robot DOF): {exp['output_size']}")
print(f"Use the MultiViewVitSensorProcessing class to load and use this model for inference.")

