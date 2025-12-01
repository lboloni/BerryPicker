#!/usr/bin/env python
# coding: utf-8

# # Train a proprioception-tuned Vision Transformer (ViT)
# 
# We create a sensor processing model using Vision Transformer (ViT) based visual encoding finetuned with proprioception.
# 
# We start with a pretrained ViT model, then train it to:
# 1. Create a meaningful 128 or 258 dimensional latent representation
# 2. Learn to map this representation to robot positions (proprioception)
# 
# The sensor processing object associated with the trained model is in sensorprocessing/sp_vit.py

# In[16]:


import sys
sys.path.append("..")

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from behavior_cloning.demo_to_trainingdata import BCDemonstration
from sensorprocessing.sp_vit import VitSensorProcessing
from robot.al5d_position_controller import RobotPosition

from demonstration.demonstration import Demonstration

import sensorprocessing.sp_helper as sp_helper
from sensorprocessing.sp_vit import VitSensorProcessing
from robot.al5d_position_controller import RobotPosition

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# ### Exp-run initialization
# Create the exp/run-s that describe the parameters of the training. 
# Some of the code here is structured in such a way as to make the notebook automatizable with papermill.

# In[17]:


# *** Initialize the variables with default values
# *** This cell should be tagged as parameters
# *** If papermill is used, some of the values will be overwritten

# If it is set to true, the exprun will be recreated from scratch
creation_style = "exist-ok"

experiment = "sensorprocessing_propriotuned_Vit"
# Other possible configurations:
# run = "vit_base_128"  # ViT Base
# run = "vit_large_128" # ViT Large
run = "vit_base_256_009"  # ViT Base
# run = "vit_large_256" # ViT Large
# If not None, set the epochs to something different than the exp
epochs = None

# If not None, set an external experiment path
external_path = None


# If not None, set an external experiment path

# If not None, set an output path
data_path = None



# In[18]:


# Option Use papermill-style paths (when called from Flow)
# To:
if external_path:
    external_path = pathlib.Path(external_path).expanduser()  # Add .expanduser()
    external_path.mkdir(parents=True, exist_ok=True)  # Create if needed instead of assert
    Config().set_exprun_path(external_path)
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")  # Match the experiment name!
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")

if data_path:
    data_path = pathlib.Path(data_path).expanduser()  # Add .expanduser()
    data_path.mkdir(parents=True, exist_ok=True)  # Create if needed
    Config().set_results_path(data_path)

# Option 3: Use default paths (no external_path or flow_name set)
# Just uses ~/WORK/BerryPicker/data/ and source experiment_configs/

# The experiment/run we are going to run
exp = Config().get_experiment(experiment, run, creation_style=creation_style)
exp_robot = Config().get_experiment(exp["robot_exp"], exp["robot_run"])


# In[19]:


# if external_path:
#     external_path = pathlib.Path(external_path)
#     assert external_path.exists()
#     Config().set_exprun_path(external_path)
#     Config().copy_experiment("sensorprocessing_propriotuned_cnn")
#     Config().copy_experiment("robot_al5d")
#     Config().copy_experiment("demonstration")
# if data_path:
#     data_path = pathlib.Path(data_path)
#     assert data_path.exists()
#     Config().set_results_path(data_path)

# # This is an example of how to run an exprun variant
# # Config().create_exprun_variant("sensorprocessing_propriotuned_cnn","resnet50_128", {"epochs": 17}, new_run_name="boo")

# # The experiment/run we are going to run: the specified model will be created
# exp = Config().get_experiment(experiment, run, creation_style=creation_style)
# exp_robot = Config().get_experiment(exp["robot_exp"], exp["robot_run"])


# ### Create regression training data (image to proprioception)
# The training data (X, Y) is all the pictures from a demonstration with the corresponding proprioception data.

# In[20]:


def load_images_as_proprioception_training(exp: Experiment, exp_robot: Experiment):
    """Loads the training images specified in the exp/run. Processes them as two tensors as input and target data for proprioception training.
    Caches the processed results into the input and target file specified in the exp/run.

    Remove those files to recalculate
    """
    retval = {}
    proprioception_input_path = pathlib.Path(exp.data_dir(), "proprio_input.pth")
    proprioception_target_path = pathlib.Path(exp.data_dir(), "proprio_target.pth")

    if proprioception_input_path.exists():
        retval["inputs"] = torch.load(proprioception_input_path, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_path, weights_only=True)
    else:
        inputlist = []
        targetlist = []
        transform = sp_helper.get_transform_to_sp(exp)
        for val in exp["training_data"]:
            run, demo_name, camera = val
            #run = val[0]
            #demo_name = val[1]
            #camera = val[2]
            exp_demo = Config().get_experiment("demonstration", run)
            demo = Demonstration(exp_demo, demo_name)
            for i in range(demo.metadata["maxsteps"]):
                sensor_readings, _ = demo.get_image(i, device=device, transform=transform, camera=camera)
                inputlist.append(sensor_readings[0])
                rp = demo.get_action(i, "rc-position-target", exp_robot)
                # rp = RobotPosition.from_vector(exp_robot, a)
                anorm = rp.to_normalized_vector(exp_robot)
                targetlist.append(torch.from_numpy(anorm))
        retval["inputs"] = torch.stack(inputlist)
        retval["targets"] = torch.stack(targetlist)
        torch.save(retval["inputs"], proprioception_input_path)
        torch.save(retval["targets"], proprioception_target_path)

    # Separate the training and validation data.
    # We will be shuffling the demonstrations
    length = retval["inputs"].size(0)
    rows = torch.randperm(length)
    shuffled_inputs = retval["inputs"][rows]
    shuffled_targets = retval["targets"][rows]

    training_size = int( length * 0.67 )
    retval["inputs_training"] = shuffled_inputs[1:training_size]
    retval["targets_training"] = shuffled_targets[1:training_size]

    retval["inputs_validation"] = shuffled_inputs[training_size:]
    retval["targets_validation"] = shuffled_targets[training_size:]

    return retval


# In[21]:


# # Create output directory if it doesn't exist
# modelfile = pathlib.Path(
#     exp["data_dir"], exp["proprioception_mlp_model_file"])


# # data_dir = pathlib.Path(exp["data_dir"])
# # data_dir.mkdir(parents=True, exist_ok=True)
# # print(f"Data directory: {data_dir}")

# # task = exp["proprioception_training_task"]
# # proprioception_input_file = pathlib.Path(exp["data_dir"], exp["proprioception_input_file"])
# # proprioception_target_file = pathlib.Path(exp["data_dir"], exp["proprioception_target_file"])


# tr = load_images_as_proprioception_training(exp, exp_robot)
# inputs_training = tr["inputs_training"]
# targets_training = tr["targets_training"]
# inputs_validation = tr["inputs_validation"]
# targets_validation = tr["targets_validation"]



modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])

if modelfile.exists():
    print("*** Train-ProprioTuned-ViT ***: NOT training; model already exists")
else:
    # Load data only when training
    tr = load_images_as_proprioception_training(exp, exp_robot)
    inputs_training = tr["inputs_training"]
    targets_training = tr["targets_training"]
    inputs_validation = tr["inputs_validation"]
    targets_validation = tr["targets_validation"]

    # Create DataLoaders
    batch_size = exp.get('batch_size', 32)
    train_dataset = TensorDataset(inputs_training, targets_training)
    test_dataset = TensorDataset(inputs_validation, targets_validation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# ### Create the ViT model with proprioception regression

# In[22]:


# Create the ViT model with proprioception
sp = VitSensorProcessing(exp, device)
model = sp.enc  # Get the actual encoder model for training


# Debug code

print("Model created successfully")

try:
    params = model.parameters()
    print("Parameters accessed successfully")
    param_count = sum(p.numel() for p in params)
    print(f"Total parameters: {param_count}")
except Exception as e:
    print(f"Error accessing parameters: {e}")

    # Check individual components
    try:
        backbone_params = model.backbone.parameters()
        print("Backbone parameters accessed successfully")
    except Exception as e:
        print(f"Error accessing backbone parameters: {e}")

    try:
        projection_params = model.projection.parameters()
        print("Projection parameters accessed successfully")
    except Exception as e:
        print(f"Error accessing projection parameters: {e}")

    try:
        proprioceptor_params = model.proprioceptor.parameters()
        print("Proprioceptor parameters accessed successfully")
    except Exception as e:
        print(f"Error accessing proprioceptor parameters: {e}")

# Select loss function
loss_type = exp.get('loss', 'MSELoss')
if loss_type == 'MSELoss':
    criterion = nn.MSELoss()
elif loss_type == 'L1Loss':
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()  # Default to MSE

# Set up optimizer with appropriate learning rate and weight decay
optimizer = optim.Adam(
    model.parameters(),
    lr=exp.get('learning_rate', 0.001),
    weight_decay=exp.get('weight_decay', 0.01)
)

# Optional learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


# In[23]:


# # Create DataLoaders for batching
# batch_size = exp.get('batch_size', 32)
# train_dataset = TensorDataset(inputs_training, targets_training)
# test_dataset = TensorDataset(inputs_validation, targets_validation)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# In[24]:


import os
import glob
import re
import torch
import pathlib

def train_and_save_proprioception_model(model, criterion, optimizer, modelfile,
                                        device="cpu", epochs=20, scheduler=None,
                                        log_interval=1, start_epoch=0):
    """Trains and saves the ViT proprioception model with checkpointing and resume capability

    Args:
        model: ViT model with proprioception
        criterion: Loss function
        optimizer: Optimizer
        modelfile: Path to save the model
        device: Training device (cpu/cuda)
        epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        log_interval: How often to print logs
        start_epoch: Starting epoch for resumed training
    """
    # Ensure model is on the right device
    model = model.to(device)
    criterion = criterion.to(device)

    # Keep track of the best validation loss
    best_val_loss = float('inf')
    best_model_state = None

    # Create checkpoints directory if it doesn't exist
    model_dir = os.path.dirname(modelfile)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Path for best model in checkpoints directory
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # List to keep track of saved checkpoint files
    saved_checkpoints = []

    # Find existing checkpoints to add to our tracking list
    pattern = os.path.join(checkpoint_dir, "epoch_*.pth")
    existing_checkpoints = glob.glob(pattern)
    for checkpoint in existing_checkpoints:
        match = re.search(r'epoch_(\d+)\.pth$', checkpoint)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num < start_epoch:  # Only add checkpoints from before our start epoch
                saved_checkpoints.append((epoch_num, checkpoint))

    # Sort by epoch number
    saved_checkpoints.sort()

    # Keep only the 2 most recent existing checkpoints
    while len(saved_checkpoints) > 2:
        epoch_num, oldest_checkpoint = saved_checkpoints.pop(0)  # Remove the oldest
        try:
            os.remove(oldest_checkpoint)
            print(f"Deleted old existing checkpoint: {oldest_checkpoint}")
        except Exception as e:
            print(f"Failed to delete checkpoint {oldest_checkpoint}: {e}")

    # Convert to just filenames for simplicity
    saved_checkpoints = [checkpoint for _, checkpoint in saved_checkpoints]

    # Training loop
    num_epochs = epochs
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass through the full model (including proprioceptor)
            predictions = model.forward(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        # Save checkpoint for current epoch - using 6-digit epoch number format
        checkpoint_file = os.path.join(checkpoint_dir, f"epoch_{epoch+1:06d}.pth")

        # Create checkpoint with all necessary information to resume training
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

        # Add to our list of saved checkpoints
        saved_checkpoints.append(checkpoint_file)

        # Keep only the 2 most recent checkpoints
        while len(saved_checkpoints) > 2:
            oldest_checkpoint = saved_checkpoints.pop(0)  # Remove the oldest
            try:
                os.remove(oldest_checkpoint)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                print(f"Failed to delete checkpoint {oldest_checkpoint}: {e}")

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Track the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save a copy of the best model state
            best_model_state = model.state_dict().copy()
            # Save as best_model.pth in checkpoints directory
            torch.save(best_model_state, best_model_path)
            print(f"  New best model saved in checkpoints with validation loss: {best_val_loss:.4f}")

        # Log progress
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Final evaluation
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    # Save the best model to the final location only at the end of training
    if best_model_state is not None:
        torch.save(best_model_state, modelfile)
        print(f"Best model saved to {modelfile}")
    else:
        # If for some reason we don't have a best model, save the final one
        torch.save(model.state_dict(), modelfile)
        print(f"Final model saved to {modelfile}")

    return model

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint file to resume training"""
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return None, 0

    # Look for checkpoint files
    pattern = os.path.join(checkpoint_dir, "epoch_*.pth")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None, 0

    # Extract epoch numbers and find the latest one
    epoch_numbers = []
    for file in checkpoint_files:
        try:
            epoch = int(re.search(r'epoch_(\d+)\.pth$', file).group(1))
            epoch_numbers.append((epoch, file))
        except (ValueError, IndexError, AttributeError):
            continue

    if not epoch_numbers:
        return None, 0

    # Sort and get the latest
    epoch_numbers.sort(reverse=True)
    latest_epoch, latest_file = epoch_numbers[0]

    return latest_file, latest_epoch

# Main code to use the updated training function
model_dir = pathlib.Path(exp["data_dir"])
modelfile = model_dir / exp["proprioception_mlp_model_file"]
epochs = exp.get("epochs", 300)  # Default to 300 epochs

# Check for latest checkpoint or existing model
latest_checkpoint, start_epoch = find_latest_checkpoint(model_dir)

if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint} (Epoch {start_epoch})")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Previous best validation loss: {best_val_loss:.4f}")

    # Continue training from the next epoch
    print(f"Continuing training for {epochs - start_epoch} more epochs")
    model = train_and_save_proprioception_model(
        model, criterion, optimizer, modelfile,
        device=device, epochs=epochs, scheduler=lr_scheduler,
        start_epoch=start_epoch
    )

elif modelfile.exists() and exp.get("reload_existing_model", True):
    print(f"Loading existing final model from {modelfile}")
    model.load_state_dict(torch.load(modelfile, map_location=device))

    # Optional: evaluate the loaded model
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Loaded model validation loss: {avg_val_loss:.4f}")

    # Start fresh training
    print(f"Starting fresh training for {epochs} epochs")
    model = train_and_save_proprioception_model(
        model, criterion, optimizer, modelfile,
        device=device, epochs=epochs, scheduler=lr_scheduler
    )

else:
    print(f"Training new model for {epochs} epochs")
    model = train_and_save_proprioception_model(
        model, criterion, optimizer, modelfile,
        device=device, epochs=epochs, scheduler=lr_scheduler
    )


# In[25]:


# import os
# import glob

# def train_and_save_proprioception_model(model, criterion, optimizer, modelfile,
#                                        device="cpu", epochs=20, scheduler=None,
#                                        log_interval=1, start_epoch=0):
#     """Trains and saves the ViT proprioception model with checkpointing and resume capability

#     Args:
#         model: ViT model with proprioception
#         criterion: Loss function
#         optimizer: Optimizer
#         modelfile: Path to save the model
#         device: Training device (cpu/cuda)
#         epochs: Number of training epochs
#         scheduler: Optional learning rate scheduler
#         log_interval: How often to print logs
#         start_epoch: Starting epoch for resumed training
#     """
#     # Ensure model is on the right device
#     model = model.to(device)
#     criterion = criterion.to(device)

#     # Keep track of the best validation loss
#     best_val_loss = float('inf')

#     # Create directory for checkpoint files if it doesn't exist
#     checkpoint_dir = os.path.dirname(modelfile)
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # Base filename for checkpoints
#     base_filename = os.path.basename(modelfile).split('.')[0]

#     # Training loop
#     num_epochs = epochs
#     for epoch in range(start_epoch, num_epochs):
#         # Training phase
#         model.train()
#         total_loss = 0
#         for batch_X, batch_y in train_loader:
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)

#             # Forward pass through the full model (including proprioceptor)
#             predictions = model.forward(batch_X)
#             loss = criterion(predictions, batch_y)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)

#         # Validation phase
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch_X, batch_y in test_loader:
#                 batch_X = batch_X.to(device)
#                 batch_y = batch_y.to(device)
#                 predictions = model(batch_X)
#                 loss = criterion(predictions, batch_y)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(test_loader)

#         # Save checkpoint for every epoch
#         checkpoint_file = os.path.join(checkpoint_dir, f"{base_filename}_epoch_{epoch+1}.pt")

#         # Create checkpoint with all necessary information to resume training
#         checkpoint = {
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'best_val_loss': best_val_loss,
#             'train_loss': avg_train_loss,
#             'val_loss': avg_val_loss
#         }

#         if scheduler is not None:
#             checkpoint['scheduler_state_dict'] = scheduler.state_dict()

#         torch.save(checkpoint, checkpoint_file)
#         print(f"Checkpoint saved: {checkpoint_file}")

#         # Update learning rate if scheduler is provided
#         if scheduler is not None:
#             scheduler.step(avg_val_loss)

#         # Save the best model separately
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), modelfile)
#             print(f"  New best model saved with validation loss: {best_val_loss:.4f}")

#         # Log progress
#         if (epoch + 1) % log_interval == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

#         # Delete older checkpoints (keep only every 2nd epoch)
#         if epoch >= 2:  # Start deleting after we have some checkpoints
#             old_checkpoint = os.path.join(checkpoint_dir, f"{base_filename}_epoch_{epoch-1}.pt")
#             if os.path.exists(old_checkpoint) and (epoch-1) % 2 != 0:  # Delete if not divisible by 2
#                 try:
#                     os.remove(old_checkpoint)
#                     print(f"Deleted old checkpoint: {old_checkpoint}")
#                 except Exception as e:
#                     print(f"Failed to delete checkpoint {old_checkpoint}: {e}")

#     # Final evaluation
#     print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
#     return model

# def find_latest_checkpoint(modelfile):
#     """Find the latest checkpoint file to resume training"""
#     checkpoint_dir = os.path.dirname(modelfile)
#     base_filename = os.path.basename(modelfile).split('.')[0]

#     # Look for checkpoint files
#     pattern = os.path.join(checkpoint_dir, f"{base_filename}_epoch_*.pt")
#     checkpoint_files = glob.glob(pattern)

#     if not checkpoint_files:
#         return None, 0

#     # Extract epoch numbers and find the latest one
#     epoch_numbers = []
#     for file in checkpoint_files:
#         try:
#             epoch = int(file.split('_epoch_')[1].split('.pt')[0])
#             epoch_numbers.append((epoch, file))
#         except (ValueError, IndexError):
#             continue

#     if not epoch_numbers:
#         return None, 0

#     # Sort and get the latest
#     epoch_numbers.sort(reverse=True)
#     latest_epoch, latest_file = epoch_numbers[0]

#     return latest_file, latest_epoch

# # Modified main code to use the updated training function
# modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
# epochs = exp.get("epochs", 20)

# # Check for latest checkpoint or existing model
# latest_checkpoint, start_epoch = find_latest_checkpoint(modelfile)

# if latest_checkpoint:
#     print(f"Resuming training from checkpoint: {latest_checkpoint} (Epoch {start_epoch})")
#     checkpoint = torch.load(latest_checkpoint, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     if 'scheduler_state_dict' in checkpoint and lr_scheduler is not None:
#         lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#     best_val_loss = checkpoint['best_val_loss']
#     print(f"Previous best validation loss: {best_val_loss:.4f}")

#     # Continue training from the next epoch
#     print(f"Continuing training for {epochs - start_epoch} more epochs")
#     model = train_and_save_proprioception_model(
#         model, criterion, optimizer, modelfile,
#         device=device, epochs=epochs, scheduler=lr_scheduler,
#         start_epoch=start_epoch
#     )

# elif modelfile.exists() and exp.get("reload_existing_model", True):
#     print(f"Loading existing final model from {modelfile}")
#     model.load_state_dict(torch.load(modelfile, map_location=device))

#     # Optional: evaluate the loaded model
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for batch_X, batch_y in test_loader:
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)
#             predictions = model(batch_X)
#             loss = criterion(predictions, batch_y)
#             val_loss += loss.item()

#         avg_val_loss = val_loss / len(test_loader)
#         print(f"Loaded model validation loss: {avg_val_loss:.4f}")

#     # Ask if we should continue training
#     print(f"Starting fresh training for {epochs} epochs")
#     model = train_and_save_proprioception_model(
#         model, criterion, optimizer, modelfile,
#         device=device, epochs=epochs, scheduler=lr_scheduler
#     )

# else:
#     print(f"Training new model for {epochs} epochs")
#     model = train_and_save_proprioception_model(
#         model, criterion, optimizer, modelfile,
#         device=device, epochs=epochs, scheduler=lr_scheduler
#     )


# In[26]:


# def train_and_save_proprioception_model(model, criterion, optimizer, modelfile,
#                                         device="cpu", epochs=20, scheduler=None,
#                                         log_interval=1):
#     """Trains and saves the ViT proprioception model

#     Args:
#         model: ViT model with proprioception
#         criterion: Loss function
#         optimizer: Optimizer
#         modelfile: Path to save the model
#         device: Training device (cpu/cuda)
#         epochs: Number of training epochs
#         scheduler: Optional learning rate scheduler
#         log_interval: How often to print logs
#     """
#     # Ensure model is on the right device
#     model = model.to(device)
#     criterion = criterion.to(device)

#     # Keep track of the best validation loss
#     best_val_loss = float('inf')

#     # Training loop
#     num_epochs = epochs
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         total_loss = 0
#         for batch_X, batch_y in train_loader:
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)

#             # Forward pass through the full model (including proprioceptor)
#             predictions = model.forward(batch_X)
#             loss = criterion(predictions, batch_y)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)

#         # Validation phase
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch_X, batch_y in test_loader:
#                 batch_X = batch_X.to(device)
#                 batch_y = batch_y.to(device)
#                 predictions = model(batch_X)
#                 loss = criterion(predictions, batch_y)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(test_loader)

#         # Update learning rate if scheduler is provided
#         if scheduler is not None:
#             scheduler.step(avg_val_loss)

#         # Save the best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), modelfile)
#             print(f"  New best model saved with validation loss: {best_val_loss:.4f}")

#         # Log progress
#         if (epoch + 1) % log_interval == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

#     # Final evaluation
#     print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
#     return model


# In[27]:


# modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
# epochs = exp.get("epochs", 20)

# # Check if model already exists
# if modelfile.exists() and exp.get("reload_existing_model", True):
#     print(f"Loading existing model from {modelfile}")
#     model.load_state_dict(torch.load(modelfile, map_location=device))

#     # Optional: evaluate the loaded model
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for batch_X, batch_y in test_loader:
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)
#             predictions = model(batch_X)
#             loss = criterion(predictions, batch_y)
#             val_loss += loss.item()

#         avg_val_loss = val_loss / len(test_loader)
#         print(f"Loaded model validation loss: {avg_val_loss:.4f}")
# else:
#     print(f"Training new model for {epochs} epochs")
#     model = train_and_save_proprioception_model(
#         model, criterion, optimizer, modelfile,
#         device=device, epochs=epochs, scheduler=lr_scheduler
#     )


# ### Test the trained model

# In[28]:


# Create the sensor processing module using the trained model
sp = VitSensorProcessing(exp, device)

# Test it on a few validation examples
def test_sensor_processing(sp, test_images, test_targets, n_samples=5):
    """Test the sensor processing module on a few examples."""
    if n_samples > len(test_images):
        n_samples = len(test_images)

    # Get random indices
    indices = torch.randperm(len(test_images))[:n_samples]

    print("\nTesting sensor processing on random examples:")
    print("-" * 50)

    for i, idx in enumerate(indices):
        # Get image and target
        image = test_images[idx].unsqueeze(0).to(device)  # Add batch dimension
        target = test_targets[idx].cpu().numpy()

        # Process the image to get the latent representation
        latent = sp.process(image)

        # Print the results
        print(f"Example {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Target position: {target}")
        print()

# Test the sensor processing
test_sensor_processing(sp, inputs_validation, targets_validation)


# ### Verify the model's encoding and forward methods

# In[29]:


# Verify that the encoding method works correctly
model.eval()
with torch.no_grad():
    # Get a sample image
    sample_image = inputs_validation[0].unsqueeze(0).to(device)

    # Get the latent representation using encode
    latent = model.encode(sample_image)
    print(f"Latent representation shape: {latent.shape}")

    # Get the robot position prediction using forward
    position = model.forward(sample_image)
    print(f"Robot position prediction shape: {position.shape}")

    # Check that the latent representation has the expected size
    expected_latent_size = exp["latent_size"]
    assert latent.shape[1] == expected_latent_size, f"Expected latent size {expected_latent_size}, got {latent.shape[1]}"

    # Check that the position prediction has the expected size
    expected_output_size = exp["output_size"]
    assert position.shape[1] == expected_output_size, f"Expected output size {expected_output_size}, got {position.shape[1]}"

    print("Verification successful!")


# ### Save final model and summary

# In[30]:


# Save the model and print summary
final_modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
torch.save(model.state_dict(), final_modelfile)
print(f"Model saved to {final_modelfile}")

print("\nTraining complete!")
print(f"Vision Transformer type: {exp['vit_model']}")
print(f"Latent space dimension: {exp['latent_size']}")
print(f"Output dimension (robot DOF): {exp['output_size']}")
print(f"Use the VitSensorProcessing class to load and use this model for inference.")

