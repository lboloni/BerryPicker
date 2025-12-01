#!/usr/bin/env python
# coding: utf-8

# # Compare models for visual proprioception
# 
# Compares regression models for visual proprioception, by running them on specific test data, and creating comparison graphs that put all of them onto the graphs. 
# 
# Each configuration is specified by a run of type visual_proprioception.

# In[ ]:


import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import math
import torch
import torch.nn as nn
import csv

torch.manual_seed(1)

from visual_proprioception.visproprio_helper import (
    load_demonstrations_as_proprioception_training,
    load_multiview_demonstrations_as_proprioception_training,
    get_visual_proprioception_sp
)
from visual_proprioception.visproprio_models import VisProprio_SimpleMLPRegression
import sensorprocessing.sp_factory as sp_factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


# =============================================================================
# DETERMINISTIC RUN SETUP
# =============================================================================

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
superpower = 777
torch.use_deterministic_algorithms(True)
torch.manual_seed(superpower)
import random
random.seed(superpower)
np.random.seed(superpower)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(superpower)


# In[ ]:


# =============================================================================
# PAPERMILL PARAMETERS
# =============================================================================

creation_style = "exist-ok"

experiment = "visual_proprioception"
run = "vp_comp_flow_all"

external_path = None
data_path = None


# In[ ]:


# =============================================================================
# INITIALIZATION
# =============================================================================

if external_path:
    external_path = pathlib.Path(external_path).expanduser()
    assert external_path.exists()
    Config().set_exprun_path(external_path)
    Config().copy_experiment("sensorprocessing_aruco")
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn_multiview")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")

if data_path:
    data_path = pathlib.Path(data_path).expanduser()
    assert data_path.exists()
    Config().set_results_path(data_path)


# In[ ]:


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def force_to_device(model, target_device):
    """Make sure model and all its parameters are on the target device."""
    if model is None:
        return None
    model = model.to(target_device)
    for param in model.parameters():
        if param.device != target_device:
            param.data = param.data.to(target_device)
    return model


def ensure_tensor_on_device(x, target_device):
    """Make sure x is a tensor and is on the correct device."""
    if not isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = torch.tensor(x, dtype=torch.float32)
    return x.to(target_device)


def is_model_multiview(exp):
    """Check if an experiment uses a multiview model."""
    sp_name = exp.get("sensor_processing", "")
    return (
        sp_name.endswith("_multiview") or
        sp_name.lower().startswith("multiview") or
        "multiview" in sp_name.lower() or
        exp.get("num_views", 1) > 1
    )


# In[ ]:


# =============================================================================
# LOAD COMPARISON EXPERIMENT
# =============================================================================

exp = Config().get_experiment(experiment, run)
runs = exp["tocompare"]

exp_robot = Config().get_experiment(
    exp.get("robot_exp", "robot_al5d"),
    exp.get("robot_run", "position_controller_00")
)

print(f"Comparing {len(runs)} models:")
for r in runs:
    print(f"  - {r}")


# In[ ]:


# =============================================================================
# LOAD ALL MODELS AND SENSOR PROCESSORS
# =============================================================================

exps = []
sps = []
spexps = []
models = []

for subrun in runs:
    subexp = Config().get_experiment(experiment, subrun)
    exps.append(subexp)

    spexp = Config().get_experiment(subexp["sp_experiment"], subexp["sp_run"])
    spexps.append(spexp)

    # Load and ensure sensor processor is on the right device
    sp = sp_factory.create_sp(spexp, device)
    if hasattr(sp, 'enc') and hasattr(sp.enc, 'to'):
        sp.enc = force_to_device(sp.enc, device)
    if hasattr(sp, 'model') and hasattr(sp.model, 'to'):
        sp.model = force_to_device(sp.model, device)
    sps.append(sp)

    # Load regression model
    model = VisProprio_SimpleMLPRegression(subexp)
    modelfile = pathlib.Path(subexp["data_dir"], subexp["proprioception_mlp_model_file"])

    # Handle different checkpoint formats
    checkpoint = torch.load(modelfile, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = force_to_device(model, device)
    model.eval()
    models.append(model)

    print(f"Loaded {subexp.get('name', subrun)}: multiview={is_model_multiview(subexp)}")


# In[ ]:


# =============================================================================
# EVALUATE EACH MODEL ON ITS TEST DATA
# =============================================================================

print("\n=== Evaluating Each Model ===")
ypreds = []
all_targets = []
all_t = []

for i, (subexp, sp, spexp, model) in enumerate(zip(exps, sps, spexps, models)):
    model_name = subexp.get('name', f'model_{i}')
    print(f"\nProcessing model {i+1}/{len(models)}: {model_name}")

    # Get test data paths
    proprioception_input_file = pathlib.Path(
        subexp.data_dir(), subexp.get("proprioception_test_input_file", "proprioception_test_input.pth")
    )
    proprioception_target_file = pathlib.Path(
        subexp.data_dir(), subexp.get("proprioception_test_target_file", "proprioception_test_target.pth")
    )

    # Determine if multiview
    is_multiview = is_model_multiview(subexp)

    if is_multiview:
        print(f"  Using multi-view evaluation with {subexp.get('num_views', 2)} views")

        tr = load_multiview_demonstrations_as_proprioception_training(
            sp, subexp, spexp, exp_robot,
            "validation_data",
            proprioception_input_file,
            proprioception_target_file,
            device=device
        )

        inputs = tr["inputs"]
        targets = tr["targets"]

        # Run predictions
        ypred = []
        y = []
        t = []

        model.eval()
        with torch.no_grad():
            for idx in range(len(targets)):
                x = inputs[idx].to(device)
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)

                predictions = model(x)

                t.append(idx)
                y.append(targets[idx].cpu().numpy())
                ypred.append(predictions[0].cpu().numpy())

    else:
        print(f"  Using single-view evaluation")

        tr = load_demonstrations_as_proprioception_training(
            sp, subexp, spexp, exp_robot,
            "validation_data",
            proprioception_input_file,
            proprioception_target_file,
            device=device
        )

        inputs = tr["inputs"]
        targets = tr["targets"]

        # Run predictions
        ypred = []
        y = []
        t = []

        model.eval()
        with torch.no_grad():
            for idx in range(len(targets)):
                x = inputs[idx].to(device)
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)

                predictions = model(x)

                t.append(idx)
                y.append(targets[idx].cpu().numpy())
                ypred.append(predictions[0].cpu().numpy())

    ypred = np.vstack(ypred)
    y = np.vstack(y)
    t = np.array(t)

    ypreds.append(ypred)
    all_targets.append(y)
    all_t.append(t)

    # Calculate MSE
    mse = np.mean(np.sum((ypred - y) ** 2, axis=1))
    print(f"  MSE: {mse:.6f}")

# Use first model's targets and time for plotting
y = all_targets[0]
t = all_t[0]


# In[ ]:


# =============================================================================
# PLOTTING
# =============================================================================

titles = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]
output_dir = pathlib.Path(exp["data_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# --- Time comparison plot (3x2 layout with bottom legend) ---
fig, axs = plt.subplots(3, 2, figsize=(4.6, 6))
for i in range(len(titles)):
    ax = axs[i // 2, i % 2]

    for ypred, subexp in zip(ypreds, exps):
        ax.plot(t, ypred[:, i], label=subexp.get("name", "model"), linewidth=1)
    ax.plot(t, y[:, i], label="ground truth", linewidth=2, color="black")

    if i == 4:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(exps) + 1,
                   bbox_to_anchor=(0.5, 0), loc="upper center", fontsize=8)
    ax.set_title(titles[i])

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
graphfilename = output_dir / "comparison23.pdf"
plt.savefig(graphfilename, bbox_inches='tight')
graphfilename = output_dir / "comparison23.jpg"
plt.savefig(graphfilename, bbox_inches='tight')
print(f"Saved: {graphfilename}")


# In[ ]:


# --- MSE comparison plot (3x2 layout) ---
fig, axs = plt.subplots(3, 2, figsize=(4.6, 6))

for i in range(len(titles)):
    ax = axs[i // 2, i % 2]
    ax.set_ylim(0, 0.5)

    bars = []
    names = []

    for ypred, subexp in zip(ypreds, exps):
        error = math.sqrt(np.mean((y[:, i] - ypred[:, i]) ** 2))
        br = ax.bar(subexp.get("name", "model"), error, label=subexp.get("name", "model"))
        bars.append(br)
        names.append(subexp.get("name", "model"))

    ax.set_xticks([])

    if i == 4:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2,
                   bbox_to_anchor=(0.5, 0), loc="upper center", fontsize=8)
    ax.set_title(titles[i])

fig.tight_layout()
plt.subplots_adjust(bottom=0.15)
graphfilename = output_dir / "msecomparison23.pdf"
plt.savefig(graphfilename, bbox_inches='tight')
graphfilename = output_dir / "msecomparison23.jpg"
plt.savefig(graphfilename, bbox_inches='tight')
print(f"Saved: {graphfilename}")


# In[ ]:


# =============================================================================
# GENERATE CSV FILE
# =============================================================================

csv_path = output_dir / "msecomparison_values.csv"

# Clear existing file
if csv_path.exists():
    csv_path.unlink()

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Header
    header = ["Metric"] + [subexp.get("name", f"model_{j}") for j, subexp in enumerate(exps)]
    writer.writerow(header)

    # Write MSE for each DOF
    for i, title in enumerate(titles):
        errors = []
        for ypred in ypreds:
            error = math.sqrt(np.mean((y[:, i] - ypred[:, i]) ** 2))
            errors.append(error)
        writer.writerow([title] + errors)

    # Write overall MSE
    overall_errors = []
    for ypred in ypreds:
        overall_mse = np.mean(np.sum((ypred - y) ** 2, axis=1))
        overall_errors.append(math.sqrt(overall_mse))
    writer.writerow(["Overall RMSE"] + overall_errors)

print(f"Saved: {csv_path}")


# In[ ]:


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n=== Summary ===")
print("-" * 80)
print(f"{'Model':<40} {'Type':<15} {'Overall RMSE':>15}")
print("-" * 80)

for j, (subexp, ypred) in enumerate(zip(exps, ypreds)):
    model_name = subexp.get("name", f"model_{j}")[:40]
    model_type = "Multiview" if is_model_multiview(subexp) else "Single-view"
    overall_rmse = math.sqrt(np.mean(np.sum((ypred - y) ** 2, axis=1)))
    print(f"{model_name:<40} {model_type:<15} {overall_rmse:>15.6f}")

print("-" * 80)

# Find best model
best_idx = np.argmin([math.sqrt(np.mean(np.sum((ypred - y) ** 2, axis=1))) for ypred in ypreds])
best_name = exps[best_idx].get("name", f"model_{best_idx}")
print(f"\nBest model: {best_name}")

print("\nComparison complete!")

