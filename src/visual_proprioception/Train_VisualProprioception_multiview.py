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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

from visual_proprioception.visproprio_helper import (
    load_demonstrations_as_proprioception_training,
    load_multiview_demonstrations_as_proprioception_training,
    get_visual_proprioception_sp,
    MultiViewDataset,
    collate_multiview
)
from visual_proprioception.visproprio_models import VisProprio_SimpleMLPRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


# ##############################################
# #                 OLD STYLE                  #
# ##############################################

# experiment = "visual_proprioception"

# ##############################################
# #                 SingleView                 #
# ##############################################

# # the latent space 128 ones

# # run = "vp_aruco_128"  #DONE
# # run = "vp_convvae_128" #DONE
# # run = "vp_ptun_vgg19_128" #DONE
# # run = "vp_ptun_resnet50_128" #DONE

# # the latent space 256 ones
# # run = "vp_convvae_256" #DONE
# # run = "vp_ptun_vgg19_256" #DONE
# # run = "vp_ptun_resnet50_256" #DONE

# #vits
# run ="vit_base_128" #DONE
# # run ="vit_base_256" #DONE

# # run ="vit_large_128" #DONE
# # run ="vit_large_256" #DONE


# ##############################################
# #                 MultiViews  - NEW!         #
# ##############################################

# ############  latent space: 128  ############
# #concat_proj

# # run ="vit_base_multiview_128"  #DONE
# # run ="vit_large_multiview_128"  #DONE


# ##  indiv_proj
# # run = "vit_base_multiview_indiv_proj_128"  # ViT Base_indiv_proj_128  #DONE
# # run = "vit_large_multiview_indiv_proj_128" # ViT Large_indiv_proj_128  #DONE

# ##  attention
# # run = "vit_base_multiview_attention_128"  # ViT Base_attention  #DONE
# # run = "vit_large_multiview_attention_128" # ViT Large_attention  #DONE


# ##  weighted_sum
# # run = "vit_base_multiview_weighted_sum_128"  # ViT Base_weighted_sum  #DONE
# # run = "vit_large_multiview_weighted_sum_128" # ViT Large_weighted_sum  #DONE

# ##  gated
# # run = "vit_base_multiview_gated_128"  # ViT Base_gated  #DONE
# # run = "vit_large_multiview_gated_128" # ViT Large_gated  #DONE

# ########## the latent space 256 ones #########


# # run ="vit_base_multiview_256"  #DONE
# # run ="vit_large_multiview_256"  #DONE


# ##  indiv_proj
# # run = "vit_base_multiview_indiv_proj_256"  # ViT Base_indiv_proj_256   #DONE
# # run = "vit_large_multiview_indiv_proj_256" # ViT Large_indiv_proj_256 #DONE

# ##  attention
# # run = "vit_base_multiview_attention_256"  # DONE
# # run = "vit_large_multiview_attention_256" # ViT Large_attention #DONE


# ##  weighted_sum
# # run = "vit_base_multiview_weighted_sum_256"  # ViT Base_weighted_sum   #DONE
# # run = "vit_large_multiview_weighted_sum_256" # ViT Large_weighted_sum  #DONE


# ##  gated
# # run = "vit_base_multiview_gated_256"  # ViT Base_gated  #DONE
# # run = "vit_large_multiview_gated_256" # ViT Large_gated  #DONE


# ##############################################
# #          MultiViews Image Concat - NEW!    #
# ##############################################
# # the latent space 128 ones
# # run = "vit_base_concat_multiview_128" # ViT Base  #DONE
# # run = "vit_large_concat_multiview_128"  # ViT Large  #DONE
# # run = "vp_convvae_128_concat_multiview"  #DONE

# # the latent space 256 ones

# # run = "vit_base_concat_multiview_256" # ViT Base  #DONE
# # run = "vit_large_concat_multiview_256"  # ViT Large  #DONE
# # run = "vp_convvae_256_concat_multiview" #DONE


# ##############################################
# #          MultiViews CNN - NEW!             #
# ##############################################

# # run = "vp_ptun_vgg19_128_multiview" #DONE
# # run = "vp_ptun_resnet50_128_multiview" #DONE
# # run = "vp_ptun_vgg19_256_multiview" #DONE
# # run = "vp_ptun_resnet50_256_multiview" #DONE




# exp = Config().get_experiment(experiment, run)
# pprint(exp)

# sp = get_visual_proprioception_sp(exp, device)


# In[ ]:


# =============================================================================
# PAPERMILL PARAMETERS - This cell should be tagged as 'parameters'
# =============================================================================

creation_style = "exist-ok"

experiment = "visual_proprioception"
run = "vp_vit_base_multiview_concat_proj"

# If not None, set an external experiment path
external_path = None
# If not None, set an output path
data_path = None


# In[ ]:


# =============================================================================
# INITIALIZATION
# =============================================================================

if external_path:
    external_path = pathlib.Path(external_path).expanduser()
    external_path.mkdir(parents=True, exist_ok=True)
    Config().set_exprun_path(external_path)
    Config().copy_experiment("visual_proprioception")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn_multiview")
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")

if data_path:
    data_path = pathlib.Path(data_path).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    Config().set_results_path(data_path)

exp = Config().get_experiment(experiment, run, creation_style=creation_style)
pprint(exp)

# Get sensor processor
sp = get_visual_proprioception_sp(exp, device)

# Get robot experiment for normalization
exp_robot = Config().get_experiment(
    exp.get("robot_exp", "robot_al5d"),
    exp.get("robot_run", "position_controller_00")
)

# Get SP experiment for transform
spexp = Config().get_experiment(exp["sp_experiment"], exp["sp_run"])


# In[ ]:


# =============================================================================
# MODEL CREATION
# =============================================================================

model = VisProprio_SimpleMLPRegression(exp)
model.to(device)

if exp["loss"] == "MSE":
    criterion = nn.MSELoss()
elif exp["loss"] == "L1":
    criterion = nn.L1Loss()
else:
    raise Exception(f'Unknown loss type {exp["loss"]}')

optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


# =============================================================================
# DETERMINE MODEL TYPE
# =============================================================================

is_multiview = (
    exp.get("sensor_processing", "").endswith("_multiview") or
    exp.get("sensor_processing", "").lower().startswith("multiview") or
    "multiview" in exp.get("sensor_processing", "").lower() or
    exp.get("num_views", 1) > 1
)

is_cnn_multiview = (
    exp.get("sensor_processing", "") == "VGG19ProprioTunedSensorProcessing_multiview" or
    exp.get("sensor_processing", "") == "ResNetProprioTunedSensorProcessing_multiview"
)

print(f"Is multiview: {is_multiview}")
print(f"Is CNN multiview: {is_cnn_multiview}")


# In[ ]:


# =============================================================================
# DATA LOADING
# =============================================================================

proprioception_input_file = pathlib.Path(
    exp["data_dir"], exp.get("proprioception_input_file", "proprioception_input.pth")
)
proprioception_target_file = pathlib.Path(
    exp["data_dir"], exp.get("proprioception_target_file", "proprioception_target.pth")
)

if is_multiview:
    print(f"Using multi-view approach with {exp.get('num_views', 2)} views")

    # Use multiview loading function
    tr = load_multiview_demonstrations_as_proprioception_training(
        sp, exp, spexp, exp_robot,
        "training_data",
        proprioception_input_file,
        proprioception_target_file,
        device=device
    )

    # For multiview, inputs are already encoded latents
    inputs_training = tr["inputs"]
    targets_training = tr["targets"]

    # Load validation data
    val_input_file = pathlib.Path(
        exp["data_dir"], exp.get("proprioception_val_input_file", "proprioception_val_input.pth")
    )
    val_target_file = pathlib.Path(
        exp["data_dir"], exp.get("proprioception_val_target_file", "proprioception_val_target.pth")
    )

    tr_val = load_multiview_demonstrations_as_proprioception_training(
        sp, exp, spexp, exp_robot,
        "validation_data",
        val_input_file,
        val_target_file,
        device=device
    )

    inputs_validation = tr_val["inputs"]
    targets_validation = tr_val["targets"]

    # Create standard TensorDataset (inputs are already encoded)
    batch_size = exp.get('batch_size', 32)
    train_dataset = TensorDataset(inputs_training, targets_training)
    test_dataset = TensorDataset(inputs_validation, targets_validation)

else:
    print("Using single-view approach")

    tr = load_demonstrations_as_proprioception_training(
        sp, exp, spexp, exp_robot,
        "training_data",
        proprioception_input_file,
        proprioception_target_file,
        device=device
    )

    inputs_training = tr["inputs"]
    targets_training = tr["targets"]

    # Load validation data
    val_input_file = pathlib.Path(
        exp["data_dir"], exp.get("proprioception_val_input_file", "proprioception_val_input.pth")
    )
    val_target_file = pathlib.Path(
        exp["data_dir"], exp.get("proprioception_val_target_file", "proprioception_val_target.pth")
    )

    tr_val = load_demonstrations_as_proprioception_training(
        sp, exp, spexp, exp_robot,
        "validation_data",
        val_input_file,
        val_target_file,
        device=device
    )

    inputs_validation = tr_val["inputs"]
    targets_validation = tr_val["targets"]

    batch_size = exp.get('batch_size', 32)
    train_dataset = TensorDataset(inputs_training, targets_training)
    test_dataset = TensorDataset(inputs_validation, targets_validation)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(test_dataset)}")


# In[ ]:


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def has_batch_norm(model):
    """Check if model contains BatchNorm layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def train_and_save_proprioception_model(
    model, criterion, optimizer, train_loader, test_loader,
    modelfile, device="cpu", epochs=100
):
    """Train and save the visual proprioception model.

    Args:
        model: The MLP regressor model
        criterion: Loss function
        optimizer: Optimizer
        train_loader: Training data loader
        test_loader: Validation data loader
        modelfile: Path to save the model
        device: Training device
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    model = model.to(device)

    # Checkpoint directory
    checkpoint_dir = modelfile.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    best_loss = float('inf')
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
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    def cleanup_old_checkpoints():
        """Keep only the last 3 checkpoints."""
        old_checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"), key=get_epoch_number)
        for old_ckpt in old_checkpoints[:-3]:
            old_ckpt.unlink()

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Skip single-sample batches if model has BatchNorm
            if batch_X.size(0) == 1 and has_batch_norm(model):
                continue

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)

        # Validation phase
        model.eval()
        test_loss = 0
        eval_batch_count = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                if batch_X.size(0) == 1 and has_batch_norm(model):
                    continue

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
                eval_batch_count += 1

        avg_test_loss = test_loss / max(eval_batch_count, 1)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_test_loss:.4f}')

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:06d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'test_loss': avg_test_loss,
            'best_loss': best_loss
        }, checkpoint_path)

        cleanup_old_checkpoints()

        # Save best model
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
            print(f"  New best model saved with test loss: {best_loss:.4f}")

    print(f"Training complete. Best test loss: {best_loss:.4f}")

    # Load best model
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']+1}")

    # Save final model
    torch.save(model.state_dict(), modelfile)
    print(f"Final model saved to {modelfile}")

    return model


# In[ ]:


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
epochs = exp.get("epochs", 100)

if modelfile.exists():
    print(f"Loading existing model from {modelfile}")
    model.load_state_dict(torch.load(modelfile, map_location=device))

    # Evaluate loaded model
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Loaded model validation loss: {avg_test_loss:.4f}")
else:
    print(f"Training new model for {epochs} epochs")
    model = train_and_save_proprioception_model(
        model, criterion, optimizer,
        train_loader, test_loader,
        modelfile, device=device, epochs=epochs
    )


# In[ ]:


# =============================================================================
# MODEL SUMMARY
# =============================================================================

print("\n*** Model Information ***")
if hasattr(sp, 'enc'):
    is_mv = hasattr(sp.enc, 'feature_extractors') and isinstance(sp.enc.feature_extractors, nn.ModuleList)
    if is_mv:
        num_views = len(sp.enc.feature_extractors)
        print(f"✓ Multi-view model detected with {num_views} views")
        print(f"  Model type: {type(sp.enc).__name__}")
        print(f"  Latent size: {sp.enc.latent_size}")
        if hasattr(sp.enc, 'fusion_type'):
            print(f"  Fusion method: {sp.enc.fusion_type}")
    else:
        print(f"✓ Single-view model detected")
        print(f"  Model type: {type(sp.enc).__name__}")
        if hasattr(sp.enc, 'latent_size'):
            print(f"  Latent size: {sp.enc.latent_size}")
else:
    print("Cannot determine model type - no 'enc' attribute found")

print(f"\nTraining complete! Model saved to:")
print(f"  {modelfile}")

