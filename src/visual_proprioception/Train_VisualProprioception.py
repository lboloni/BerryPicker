#!/usr/bin/env python
# coding: utf-8

# # Train models for visual proprioception
# 
# Train a regression model for visual proprioception. The input is sensory data (eg. a camera image). This is encoded by a p;predefined sensorprocessing component into a latent representation. What we are training and saving here is a regressor that is mapping the latent representation to the position of the robot (eg. a vector of 6 degrees of freedom).
# 
# The specification of this regressor is specified in an experiment of the type "visual_proprioception". Running this notebook will train and save this model.

# In[1]:


import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
from pprint import pprint
# import matplotlib.pyplot as plt

# import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

# from demonstration.demonstration import Demonstration


from sensorprocessing.sp_factory import create_sp
from visual_proprioception.visproprio_helper import load_demonstrations_as_proprioception_training, load_multiview_demonstrations_as_proprioception_training

from visual_proprioception.visproprio_models import VisProprio_SimpleMLPRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


#
# Code for deterministic run, from Robi Konievic
#
superpower=777
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(superpower)
import random
random.seed(superpower)
import numpy as np
np.random.seed(superpower)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(superpower)


# ### Exp-run initialization
# Create the exp/run-s that describe the parameters of the training. 
# Some of the code here is structured in such a way as to make the notebook automatizable with papermill.

# In[3]:


# *** Initialize the variables with default values
# *** This cell should be tagged as parameters
# *** If papermill is used, some of the values will be overwritten

# If it is set to discard-old, the exprun will be recreated from scratch
creation_style = "exist-ok"

experiment = "visual_proprioception"

# If not None, set the epochs to something different than the exp
epochs = None

# If not None, set an external experiment path
external_path = None

# If not None, set an output path
data_path = None

# Dr. Boloni's path
#external_path = pathlib.Path(Config()["experiment_external"])
# Sahara's path
# external_path = pathlib.Path("/home/sa641631/SaharaBerryPickerData/experiment_data")

##############################################
#                 SingleView                 #
##############################################

# the latent space 128 ones
# run = "vp_aruco_128"
# run = "vp_convvae_128"
# run = "vp_convvae_128"
# run = "vp_ptun_vgg19_128"
# run = "vp_ptun_resnet50_128"

# the latent space 256 ones
# run = "vp_convvae_256"
# run = "vp_ptun_vgg19_256"
# run = "vp_ptun_resnet50_256"

# #vits
# run ="vit_base_256_007"
# run ="vit_large"
# run ="vit_huge"

##############################################
#                 MultiViews                 #
##############################################

#concat_proj

# run ="vit_base_multiview"
# run ="vit_large_multiview"
# run =vit_huge_multiview


##  indiv_proj
# run = "vit_base_multiview_indiv_proj"  # ViT Base_indiv_proj
# run = "vit_large_multiview_indiv_proj" # ViT Large_indiv_proj
# run = "vit_huge_multiview_indiv_proj" # ViT Huge_indiv_proj

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


# In[4]:


if external_path:
    external_path = pathlib.Path(external_path)
    assert external_path.exists()
    Config().set_exprun_path(external_path)
    Config().copy_experiment("sensorprocessing_aruco")
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")
    Config().copy_experiment("visual_proprioception")
if data_path:
    data_path = pathlib.Path(data_path)
    assert data_path.exists()
    Config().set_results_path(data_path)

exp = Config().get_experiment(experiment, run, creation_style=creation_style)
pprint(exp)

# Create the sp object described in the experiment
spexp = Config().get_experiment(exp["sp_experiment"], exp["sp_run"])
sp = create_sp(spexp, device)
exp_robot = Config().get_experiment(exp["robot_exp"], exp["robot_run"])


# In[5]:


# Create the regression model

model = VisProprio_SimpleMLPRegression(exp)
model.to(device)
if exp["loss"] == "MSE":
    criterion = nn.MSELoss()
elif exp["loss"] == "L1":
    criterion = nn.L1Loss()
else:
    raise Exception(f'Unknown loss type {exp["loss"]}')

optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[6]:


print(f"spexp image_size: {spexp['image_size']}, type: {type(spexp['image_size'])}")


# Create the training and validation data which maps latent encodings into robot position

# In[7]:


# Use the original loading function

proprioception_input_file = pathlib.Path(
    exp.data_dir(), exp["proprioception_input_file"])
proprioception_target_file = pathlib.Path(
    exp.data_dir(), exp["proprioception_target_file"])
tr = load_demonstrations_as_proprioception_training(
    sp, exp, spexp, exp_robot, "training_data", proprioception_input_file, proprioception_target_file, device=device
)

inputs_training = tr["inputs"]
targets_training = tr["targets"]

proprioception_test_input_file = pathlib.Path(
    exp.data_dir(), exp["proprioception_test_input_file"])
proprioception_test_target_file = pathlib.Path(
    exp.data_dir(), exp["proprioception_test_target_file"])


exp.start_timer("load-demos-as-proprioception-training")
tr_test = load_demonstrations_as_proprioception_training(
    sp, exp, spexp, exp_robot, "validation_data", proprioception_test_input_file, proprioception_test_target_file, device=device
)
exp.end_timer("load-demos-as-proprioception-training")

inputs_validation = tr_test["inputs"]
targets_validation = tr_test["targets"]

# Create standard DataLoaders for single-view data
batch_size = exp.get('batch_size', 32)
# batch_size = exp['batch_size']
train_dataset = TensorDataset(inputs_training, targets_training)
test_dataset = TensorDataset(inputs_validation, targets_validation)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ### Perform the training

# In[8]:


def train_and_save_proprioception_model(exp):
    """Trains and saves the proprioception model, handling both single and multi-view inputs
    with checkpoint support for resuming interrupted training
    """
    final_modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
    checkpoint_dir = pathlib.Path(exp["data_dir"], "checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Maximum number of checkpoints to keep (excluding the best model)
    max_checkpoints = 2

    # Check if we're using a multi-view approach
    is_multiview = exp.get("sensor_processing", "").endswith("_multiview") or exp.get("num_views", 1) > 1
    num_views = exp.get("num_views", 2)

    # First check for existing final model
    if final_modelfile.exists() and exp.get("reload_existing_model", True):
        print(f"Loading existing final model from {final_modelfile}")
        model.load_state_dict(torch.load(final_modelfile, map_location=device))

        # Evaluate the loaded model
        model.eval()
        with torch.no_grad():
            total_loss = 0
            batch_count = 0

            for batch_data in test_loader:
                if is_multiview:
                    batch_views, batch_y = batch_data

                    # Process the batch for evaluation
                    batch_size = batch_views[0].size(0)
                    batch_features = []

                    for i in range(batch_size):
                        sample_views = [view[i].unsqueeze(0).to(device) for view in batch_views]
                        sample_features = sp.process(sample_views)
                        # Convert numpy array to tensor and move to device
                        sample_features_tensor = torch.tensor(sample_features, dtype=torch.float32).to(device)
                        batch_features.append(sample_features_tensor)

                    batch_X = torch.stack(batch_features).to(device)
                    predictions = model(batch_X)
                else:
                    batch_X, batch_y = batch_data
                    batch_X = batch_X.to(device)
                    predictions = model(batch_X)

                # Make sure batch_y is on the same device
                batch_y = batch_y.to(device)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / max(batch_count, 1)
            print(f"Loaded model evaluation loss: {avg_loss:.4f}")

        return model

    # Function to extract epoch number from checkpoint file
    def get_epoch_number(checkpoint_file):
        try:
            # Use a more robust approach to extract epoch number
            # Format: epoch_XXXX.pth where XXXX is the epoch number
            filename = checkpoint_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                return int(parts[1])  # Get the number after "epoch_"
            return 0
        except:
            return 0

    # Function to clean up old checkpoints
    def cleanup_old_checkpoints():
        # Get all epoch checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("epoch_*.pth"))

        # Sort by actual epoch number, not just filename
        checkpoint_files.sort(key=get_epoch_number)

        if len(checkpoint_files) > max_checkpoints:
            files_to_delete = checkpoint_files[:-max_checkpoints]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"Deleted old checkpoint: {file_path.name}")
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")

    # Make sure model is on the correct device
    model.to(device)
    print(f"Model moved to {device}")

    # Set training parameters
    num_epochs = exp["epochs"]
    start_epoch = 0
    best_loss = float('inf')

    # Check for existing checkpoints to resume from
    checkpoint_files = list(checkpoint_dir.glob("epoch_*.pth"))
    if checkpoint_files:
        # Sort by epoch number for more reliable ordering
        checkpoint_files.sort(key=get_epoch_number)

        # Get the most recent checkpoint
        latest_checkpoint = checkpoint_files[-1]
        epoch_num = get_epoch_number(latest_checkpoint)

        print(f"Found checkpoint from epoch {epoch_num}. Resuming training...")

        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Resuming from epoch {start_epoch}/{num_epochs} with best loss: {best_loss:.4f}")
    else:
        print(f"Starting new training for {num_epochs} epochs")

    # Start or resume training
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        batch_count = 0

        # Training loop handles both single and multi-view cases
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if is_multiview:
                    batch_views, batch_y = batch_data

                    # With multi-view, batch_views is a list of tensors, each with shape [batch_size, C, H, W]
                    batch_size = batch_views[0].size(0)
                    batch_features = []

                    # Process each sample in the batch
                    for i in range(batch_size):
                        # Extract this sample's views
                        sample_views = [view[i].unsqueeze(0).to(device) for view in batch_views]

                        # Process this sample through sp
                        sample_features = sp.process(sample_views)

                        # Convert numpy array to tensor and move to device
                        sample_features_tensor = torch.tensor(sample_features, dtype=torch.float32).to(device)
                        batch_features.append(sample_features_tensor)

                    # Stack all samples' features into a batch
                    batch_X = torch.stack(batch_features).to(device)

                    # Forward pass
                    predictions = model(batch_X)
                else:
                    batch_X, batch_y = batch_data
                    # Move to device
                    batch_X = batch_X.to(device)
                    # Standard single-view processing
                    predictions = model(batch_X)

                # Make sure batch_y is on the same device
                batch_y = batch_y.to(device)
                loss = criterion(predictions, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Print progress every few batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Save emergency checkpoint in case of error - use formatted epoch and batch numbers
                save_path = checkpoint_dir / f"emergency_epoch_{epoch:06d}_batch_{batch_idx:06d}.pth"
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / max(batch_count, 1),
                    'best_loss': best_loss
                }, save_path)
                print(f"Emergency checkpoint saved to {save_path}")
                continue

        # Calculate average loss for the epoch
        avg_loss = total_loss / max(batch_count, 1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Evaluate the model
        model.eval()
        test_loss = 0
        eval_batch_count = 0
        with torch.no_grad():
            for batch_data in test_loader:
                if is_multiview:
                    batch_views, batch_y = batch_data

                    # Process the batch the same way as in training
                    batch_size = batch_views[0].size(0)
                    batch_features = []

                    for i in range(batch_size):
                        sample_views = [view[i].unsqueeze(0).to(device) for view in batch_views]
                        sample_features = sp.process(sample_views)
                        # Convert numpy array to tensor and move to device
                        sample_features_tensor = torch.tensor(sample_features, dtype=torch.float32).to(device)
                        batch_features.append(sample_features_tensor)

                    batch_X = torch.stack(batch_features).to(device)
                    predictions = model(batch_X)
                else:
                    batch_X, batch_y = batch_data
                    batch_X = batch_X.to(device)
                    predictions = model(batch_X)

                # Make sure batch_y is on the same device
                batch_y = batch_y.to(device)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
                eval_batch_count += 1

        avg_test_loss = test_loss / max(eval_batch_count, 1)
        print(f'Validation Loss: {avg_test_loss:.4f}')

        # Save checkpoint after each epoch - using formatted epoch numbers for reliable sorting
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:06d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'test_loss': avg_test_loss,
            'best_loss': best_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Clean up old checkpoints to save space
        cleanup_old_checkpoints()

        # Update best model if improved
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'best_loss': best_loss
            }, best_model_path)
            print(f"New best model saved with test loss: {best_loss:.4f}")

    # Training completed successfully
    print(f"Training complete. Best test loss: {best_loss:.4f}")

    # Load the best model for final save
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']+1} with test loss {best_checkpoint['test_loss']:.4f}")

    # Save the final model
    torch.save(model.state_dict(), final_modelfile)
    print(f"Final model saved to {final_modelfile}")

    return model


# In[9]:


# modelfile = pathlib.Path(Config()["explorations"]["proprioception_mlp_model_file"])

#if modelfile.exists():
#    model.load_state_dict(torch.load(modelfile))
#else:
exp.start_timer("train")
train_and_save_proprioception_model(exp)
exp.end_timer("train", verbose=True)


# In[10]:


exp["timer-load-demos-as-proprioception-training-end"]


# In[ ]:





# In[ ]:




