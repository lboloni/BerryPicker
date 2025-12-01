#!/usr/bin/env python
# coding: utf-8

# # Visual proprioception flow
# 
# Create the full flow for training models for visual proprioception. This notebook programmatically generates a set of exp/runs that cover all the necessary components for a visual proprioception system (sensor processing,  visual proprioception regressor and verification notebooks).
# 
# Then, it writes the exp/runs into an external directory full separated from the github source, and creates an automation script that runs them. A separate directory for the results is also created. 
# 
# Finally, it runs the necessary notebooks to execute the whole flow using papermill.
# 
# The results directory contain the output of this flow, both in terms of trained models, as well as results (in the verification exp/run).

# In[64]:


import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import copy
import pprint
import pathlib
import yaml
import tqdm
import papermill
import visproprio_helper
from demonstration.demonstration import list_demos
from demonstration.demopack import import_demopack, group_chooser_sp_vp_standard, group_chooser_sp_vp_multiview



# # Setting up the separate directory
# Setting up a separate directory for generated exp/run config files and the results. This cell will create a new directory. 

# In[65]:


flow_name = "VisualProprioception_flow_multiview_53"
demopack_name = "random-both-cameras-video"

# Camera configuration
# For single-view models:
demonstration_cam = "dev2"
# For multi-view models:
multiview_cameras = ["dev2", "dev3"]
num_views = 2

# Model selection flags
do_VAE = True
do_VGG = True
do_RESNET = True
do_VIT = True
do_VIT_MULTIVIEW = True  # Enable multiview ViT models

# Training parameters
epochs_sp = 10
epochs_vp = 10
image_size = [256, 256]  # for VGG, ResNet etc.
vit_image_size = [224, 224]  # ViT models require 224x224

# Latent sizes to explore
latent_sizes = [128, 256]

# CNN and ViT architectures
cnntypes = ["vgg19", "resnet50"]
vit_types = ["vit_base"]  # Can add "vit_large"

# Fusion methods for multiview models
# fusion_types = ["concat_proj", "indiv_proj", "attention", "weighted_sum", "gated"]
fusion_types = ["concat_proj"]  # Use this for faster testing

# Creation style: "exist-ok" to skip existing, "discard-old" to recreate
creation_style = "exist-ok"


# ### SETUP EXTERNAL DIRECTORY

# In[66]:


# =============================================================================
# SETUP EXTERNAL DIRECTORY
# =============================================================================

demopacks_base = pathlib.Path(Config()["demopacks_path"]).expanduser()
demopack_path = pathlib.Path(demopacks_base, demopack_name)
print(f"Demopacks base: {demopacks_base}")
print(f"Base exists: {demopacks_base.exists()}")
print(f"Demopack path: {demopack_path}")
print(f"Demopack exists: {demopack_path.exists()}")

exprun_path, result_path = visproprio_helper.external_setup(
    flow_name,
    pathlib.Path(Config()["flows_path"]).expanduser()
)

# Import demopack using appropriate group chooser
demopack_path = pathlib.Path(Config()["demopacks_path"], demopack_name).expanduser()
selection = import_demopack(demopack_path, group_chooser_sp_vp_standard)

# Configure demonstration experiment
experiment = "demonstration"
exp = Config().get_experiment(experiment, demopack_name)


# ### TRAINING DATA CONFIGURATION

# In[67]:


# =============================================================================
# TRAINING DATA CONFIGURATION
# =============================================================================

# Single-view training data: [run, demo_name, camera]
sp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_training"]]
sp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_validation"]]
vp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_training"]]
vp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_validation"]]

# Multi-view training data: [run, demo_name, [camera_list]]
sp_training_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["sp_training"]]
sp_validation_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["sp_validation"]]
vp_training_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["vp_training"]]
vp_validation_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["vp_validation"]]

print(f"Selection: {selection}")
demos = list_demos(exp)
print(f"SP training demos: {list_demos(exp, 'sp')}")


# In[68]:


print(selection)


# ### TRAINING DATA CONFIGURATION

# In[69]:


# =============================================================================
# TRAINING DATA CONFIGURATION
# =============================================================================

# Single-view training data: [run, demo_name, camera]
sp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_training"]]
sp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["sp_validation"]]
vp_training_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_training"]]
vp_validation_data = [[demopack_name, demo, demonstration_cam] for demo in selection["vp_validation"]]

# Multi-view training data: [run, demo_name, [camera_list]]
sp_training_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["sp_training"]]
sp_validation_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["sp_validation"]]
vp_training_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["vp_training"]]
vp_validation_data_multiview = [[demopack_name, demo, multiview_cameras] for demo in selection["vp_validation"]]

print(f"Selection: {selection}")
demos = list_demos(exp)
print(f"SP training demos: {list_demos(exp, 'sp')}")


# In[70]:


demos = list_demos(exp)
# print(demos)
print(list_demos(exp, "sp"))
[s for s in demos if s.startswith("sp_training" + "_")]


# ### GENERATOR FUNCTIONS

# In[71]:


# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

def generate_sp_conv_vae(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the conv-vae sensorprocessing."""
    val = {}
    val["latent_size"] = params["latent_size"]
    val["epochs"] = params["epochs"]
    val["save_period"] = 5
    val["training_data"] = params["training_data"]
    val["validation_data"] = params["validation_data"]

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_Conv-VAE"
    v["notebook"] = "sensorprocessing/Train_Conv_VAE.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_sp_propriotuned_cnn(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned CNN."""
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_CNN"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_CNN.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v



def generate_sp_propriotuned_cnn_multiview(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned multiview CNN."""
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32
    val["num_views"] = params.get("num_views", 2)
    val["fusion_type"] = params.get("fusion_type", "concat_proj")

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_CNN_Multiview"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_CNN_multiview.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_sp_propriotuned_vit(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned ViT."""
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32

    # Robot configuration
    val["robot_exp"] = "robot_al5d"
    val["robot_run"] = "robot_al5d"
    val["proprioception_mlp_model_file"] = "proprioception_mlp.pth"

    # ViT architecture parameters based on model type
    if params["vit_model"] == "vit_base":
        val["vit_model"] = "vit_b_16"
        val["vit_output_dim"] = 768
        val["projection_hidden_dim"] = 512
    elif params["vit_model"] == "vit_large":
        val["vit_model"] = "vit_l_16"
        val["vit_output_dim"] = 1024
        val["projection_hidden_dim"] = 768

    val["vit_weights"] = "DEFAULT"
    val["proprio_step_1"] = 64
    val["proprio_step_2"] = 64

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_ViT"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_VIT.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_sp_propriotuned_vit_multiview(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned multiview ViT."""
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32

    # Multiview-specific
    val["num_views"] = params.get("num_views", 2)
    val["fusion_type"] = params.get("fusion_type", "concat_proj")
    val["cameras"] = params.get("cameras", multiview_cameras)

    # Robot configuration
    val["robot_exp"] = "robot_al5d"
    val["robot_run"] = "robot_al5d"
    val["proprioception_mlp_model_file"] = "proprioception_mlp.pth"

    # ViT architecture parameters based on model type
    if params["vit_model"] == "vit_base":
        val["vit_model"] = "vit_b_16"
        val["vit_output_dim"] = 768
        val["projection_hidden_dim"] = 512
    elif params["vit_model"] == "vit_large":
        val["vit_model"] = "vit_l_16"
        val["vit_output_dim"] = 1024
        val["projection_hidden_dim"] = 768

    val["vit_weights"] = "DEFAULT"
    val["proprio_step_1"] = 64
    val["proprio_step_2"] = 64

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_ViT_Multiview"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_VIT_multiview.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_vp_train(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training visual proprioception regressor."""
    val = copy.copy(params)

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Determine which notebook to use based on model type
    v = {}
    v["name"] = f"Train_{run_name}"

    is_multiview = (
        "multiview" in params.get("sensor_processing", "").lower() or
        params.get("num_views", 1) > 1
    )

    if is_multiview:
        v["notebook"] = "visual_proprioception/Train_VisualProprioception_multiview.ipynb"
    else:
        v["notebook"] = "visual_proprioception/Train_VisualProprioception.ipynb"

    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_vp_verify(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the verification of the visual proprioception regressor."""
    val = copy.copy(params)

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Generate the entry in the automation file
    v = {}
    v["name"] = f"Verify_{run_name}"
    v["notebook"] = "TODO Verify.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


def generate_vp_compare(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the comparison of visual proprioception models."""
    val = copy.copy(params)
    val["name"] = exp_name

    # Save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # Determine if any multiview models are in the comparison
    has_multiview = any("multiview" in r.lower() for r in params.get("tocompare", []))

    # Generate the entry in the automation file
    v = {}
    v["name"] = f"Compare_{run_name}"
    if has_multiview or do_VIT_MULTIVIEW:
        v["notebook"] = "visual_proprioception/Compare_VisualProprioception_multiview_and_singleview.ipynb"
    else:
        v["notebook"] = "visual_proprioception/Compare_VisualProprioception.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v



# ### GENERATE EXP/RUNS

# In[72]:


# =============================================================================
# GENERATE EXP/RUNS
# =============================================================================

expruns = []
sps = []  # list of sensor processing models (exp/run)
sps_singleview = []
sps_multiview = []
vpruns = []
vpruns_singleview = []
vpruns_multiview = []
vpruns_latent = {128: [], 256: []}


# ### GENERATE EXP/RUNS

# In[73]:


# =============================================================================
# GENERATE EXP/RUNS
# =============================================================================

expruns = []
sps = []  # list of sensor processing models (exp/run)
sps_singleview = []
sps_multiview = []
vpruns = []
vpruns_singleview = []
vpruns_multiview = []
vpruns_latent = {128: [], 256: []}

# -----------------------------------------
# Generate SENSOR PROCESSING models
# -----------------------------------------
for latent_size in latent_sizes:

    # --- Conv-VAE (single-view only) ---
    if do_VAE:
        exp_name = "sensorprocessing_conv_vae"
        run_name = f"sp_conv_vae_{latent_size}_0001"
        params = {}
        params["latent_size"] = latent_size
        params["epochs"] = epochs_sp
        params["training_data"] = sp_training_data
        params["validation_data"] = sp_validation_data
        exprun = generate_sp_conv_vae(
            exprun_path=exprun_path, result_path=result_path,
            params=params, exp_name=exp_name, run_name=run_name
        )
        exprun["latent_size"] = latent_size
        sps.append(exprun)
        sps_singleview.append(exprun)
        expruns.append(exprun)

    # --- Single-view ViT models ---
    if do_VIT:
        for vit_type in vit_types:
            exp_name = "sensorprocessing_propriotuned_Vit"
            run_name = f"sp_{vit_type}_{latent_size}_0001"
            params = {}
            params["image_size"] = vit_image_size
            params["latent_size"] = latent_size
            params["epochs"] = epochs_sp
            params["training_data"] = sp_training_data
            params["validation_data"] = sp_validation_data
            params["vit_model"] = vit_type
            params["class"] = "Vit"
            params["model"] = "VitProprioTunedRegression"
            params["freeze_feature_extractor"] = False
            params["loss"] = "MSELoss"
            params["learning_rate"] = 0.0001

            exprun = generate_sp_propriotuned_vit(
                exprun_path=exprun_path, result_path=result_path,
                params=params, exp_name=exp_name, run_name=run_name
            )
            exprun["latent_size"] = latent_size
            exprun["vittype"] = vit_type
            sps.append(exprun)
            sps_singleview.append(exprun)
            expruns.append(exprun)

    # --- Multi-view ViT models ---
    if do_VIT_MULTIVIEW:
        for vit_type in vit_types:
            for fusion_type in fusion_types:
                exp_name = "sensorprocessing_propriotuned_Vit_multiview"
                run_name = f"sp_{vit_type}_multiview_{fusion_type}_{latent_size}_0001"
                params = {}
                params["image_size"] = vit_image_size
                params["latent_size"] = latent_size
                params["epochs"] = epochs_sp
                params["num_views"] = num_views
                params["cameras"] = multiview_cameras
                params["fusion_type"] = fusion_type
                params["training_data"] = sp_training_data_multiview
                params["validation_data"] = sp_validation_data_multiview
                params["vit_model"] = vit_type
                params["class"] = "Vit_multiview"
                params["model"] = "VitProprioTunedRegression_multiview"
                params["freeze_feature_extractor"] = False
                params["loss"] = "MSELoss"
                params["learning_rate"] = 0.0001

                exprun = generate_sp_propriotuned_vit_multiview(
                    exprun_path=exprun_path, result_path=result_path,
                    params=params, exp_name=exp_name, run_name=run_name
                )
                exprun["latent_size"] = latent_size
                exprun["vittype"] = vit_type
                exprun["fusion"] = fusion_type
                exprun["is_multiview"] = True
                sps.append(exprun)
                sps_multiview.append(exprun)
                expruns.append(exprun)

    # --- Single-view CNN models (VGG, ResNet) ---
    for cnntype in cnntypes:
        exp_name = "sensorprocessing_propriotuned_cnn"
        run_name = f"sp_{cnntype}_{latent_size}_0001"
        params = {}
        params["image_size"] = image_size
        params["latent_size"] = latent_size
        params["epochs"] = epochs_sp
        params["training_data"] = sp_training_data
        params["validation_data"] = sp_validation_data

        if cnntype == "vgg19":
            if not do_VGG:
                continue
            params["class"] = "VGG19ProprioTunedSensorProcessing"
            params["model"] = "VGG19ProprioTunedRegression"
        elif cnntype == "resnet50":
            if not do_RESNET:
                continue
            params["class"] = "ResNetProprioTunedSensorProcessing"
            params["model"] = "ResNetProprioTunedRegression"
            params["freeze_feature_extractor"] = True
            params["reductor_step_1"] = 512
            params["proprio_step_1"] = 64
            params["proprio_step_2"] = 16
        else:
            raise Exception(f"Unknown cnntype {cnntype}")

        params["loss"] = "MSELoss"
        params["learning_rate"] = 0.001

        exprun = generate_sp_propriotuned_cnn(
            exprun_path=exprun_path, result_path=result_path,
            params=params, exp_name=exp_name, run_name=run_name
        )
        exprun["latent_size"] = latent_size
        exprun["cnntype"] = cnntype
        sps.append(exprun)
        sps_singleview.append(exprun)
        expruns.append(exprun)

# -----------------------------------------
# Generate VISUAL PROPRIOCEPTION models
# -----------------------------------------
for spexp in sps:
    spexp_name = spexp["experiment"]
    sprun = spexp["run"]
    latent_size = spexp["latent_size"]
    is_multiview = spexp.get("is_multiview", False) or "multiview" in sprun.lower()

    print(f"Generating VP for: {spexp_name}/{sprun} (multiview={is_multiview}, latent={latent_size})")

    # Generate the VP train exp/run
    exp_name = "visual_proprioception"
    run_name = "vp_" + sprun[3:]  # Remove "sp_" prefix
    vpruns.append(run_name)
    vpruns_latent[latent_size].append(run_name)

    if is_multiview:
        vpruns_multiview.append(run_name)
    else:
        vpruns_singleview.append(run_name)

    params = {}
    params["name"] = run_name
    params["output_size"] = 6
    params["encoding_size"] = latent_size
    params["regressor_hidden_size_1"] = 64
    params["regressor_hidden_size_2"] = 64
    params["loss"] = "MSE"
    params["epochs"] = epochs_vp
    params["batch_size"] = 64

    # Use appropriate training data
    if is_multiview:
        params["training_data"] = vp_training_data_multiview
        params["validation_data"] = vp_validation_data_multiview
        params["num_views"] = num_views
        params["cameras"] = multiview_cameras
    else:
        params["training_data"] = vp_training_data
        params["validation_data"] = vp_validation_data

    # Determine sensor processing class
    if "vae" in sprun.lower():
        params["sensor_processing"] = "ConvVaeSensorProcessing"
    elif "resnet" in sprun.lower():
        if is_multiview:
            params["sensor_processing"] = "ResNetProprioTunedSensorProcessing_multiview"
        else:
            params["sensor_processing"] = "ResNetProprioTunedSensorProcessing"
    elif "vgg19" in sprun.lower():
        if is_multiview:
            params["sensor_processing"] = "VGG19ProprioTunedSensorProcessing_multiview"
        else:
            params["sensor_processing"] = "VGG19ProprioTunedSensorProcessing"
    elif "vit" in sprun.lower():
        if is_multiview:
            params["sensor_processing"] = "MultiViewVitSensorProcessing"
        else:
            params["sensor_processing"] = "VitSensorProcessing"
    else:
        raise Exception(f"Unexpected sprun {sprun}")

    params["sp_experiment"] = spexp_name
    params["sp_run"] = sprun

    exprun = generate_vp_train(
        exprun_path=exprun_path, result_path=result_path,
        params=params, exp_name=exp_name, run_name=run_name
    )
    expruns.append(exprun)


# In[74]:


def generate_sp_conv_vae(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the conv-vae sensorprocessing with the right training data and parameters. Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    NOTE: a similar function is in Flow_BehaviorCloning.
    """
    val = {}
    val["latent_size"] = params["latent_size"]
    val["epochs"] = params["epochs"]
    val["save_period"] = 5
    val["training_data"] = params["training_data"]
    val["validation_data"] = params["validation_data"]
    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_Conv-VAE"
    v["notebook"] = "sensorprocessing/Train_Conv_VAE.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[75]:


def generate_sp_propriotuned_cnn(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned CNN with the right training data and parameters.
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    """
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32
    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_CNN"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_CNN.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[76]:


def generate_sp_propriotuned_vit(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned ViT with the right training data and parameters.
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    """
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32

    # val["robot_exp"] = "robot_al5d"
    # val["robot_run"] = "robot_al5d"
    # val["proprioception_mlp_model_file"] = "proprioception_mlp.pth"

    # ViT-specific additions
    # val["robot_exp"] = "robot_al5d"
    # val["robot_run"] = "robot_al5d"
    # val["proprioception_mlp_model_file"] = "proprioception_mlp.pth"

    # ViT architecture parameters based on model type
    if params["vit_model"] == "vit_base":
        val["vit_model"] = "vit_b_16"
        val["vit_output_dim"] = 768
        val["projection_hidden_dim"] = 512
    elif params["vit_model"] == "vit_large":
        val["vit_model"] = "vit_l_16"
        val["vit_output_dim"] = 1024
        val["projection_hidden_dim"] = 768

    val["vit_weights"] = "DEFAULT"
    val["proprio_step_1"] = 64
    val["proprio_step_2"] = 64

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_ViT"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_VIT.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[77]:


def generate_sp_propriotuned_vit_multiview(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training of the propriotuned multiview ViT."""
    val = copy.copy(params)
    val["output_size"] = 6
    val["batch_size"] = 32

    # Multiview-specific
    val["num_views"] = params.get("num_views", 2)  # ✅ Default to 2

    # ViT-specific additions
    val["robot_exp"] = "robot_al5d"
    val["robot_run"] = "robot_al5d"
    val["proprioception_mlp_model_file"] = "proprioception_mlp.pth"

    # ViT architecture parameters based on model type
    if params["vit_model"] == "vit_base":
        val["vit_model"] = "vit_b_16"
        val["vit_output_dim"] = 768
        val["projection_hidden_dim"] = 512
    elif params["vit_model"] == "vit_large":
        val["vit_model"] = "vit_l_16"
        val["vit_output_dim"] = 1024
        val["projection_hidden_dim"] = 768

    val["vit_weights"] = "DEFAULT"
    val["proprio_step_1"] = 64
    val["proprio_step_2"] = 64

    # Fusion method - THIS WAS THE BUG!
    val["fusion_type"] = params.get("fusion_type", "concat_proj")  # CORRECT

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)

    # generate the entry in the automation file
    v = {}
    v["name"] = "Train_SP_ViT_Multiview"
    v["notebook"] = "sensorprocessing/Train_ProprioTuned_VIT_multiview.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v



# In[78]:


def generate_vp_train(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the training visual proprioception regressor.
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    """
    val = copy.copy(params)

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = f"Train_{run_name}"
    if "multiview" in params.get("sensor_processing", "").lower():
        v["notebook"] = "visual_proprioception/Train_VisualProprioception_multiview.ipynb"
    else:
        v["notebook"] = "visual_proprioception/Train_VisualProprioception.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[79]:


def generate_vp_verify(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the verification of the visual proprioception regressor.
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    """
    val = copy.copy(params)

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = f"Verify_{run_name}"
    v["notebook"] = "TODO Verify.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# In[80]:


def generate_vp_compare(exprun_path, result_path, params, exp_name, run_name):
    """Generate the experiment for the verification of the visual proprioception regressor.
    Returns a dictionary with the experiment, runname as well as an entry that will be used for the automation.
    """
    val = copy.copy(params)
    val["name"] = exp_name

    # save the generated exprun spec
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    # now, generate the entry in the automation file
    v = {}
    v["name"] = f"Compare_{run_name}"
    if do_VIT_MULTIVIEW:
        v["notebook"] = "visual_proprioception/Compare_VisualProprioception_multiview_and_singleview.ipynb"
    else:
        v["notebook"] = "visual_proprioception/Compare_VisualProprioception.ipynb"
    v["experiment"] = exp_name
    v["run"] = run_name
    v["external_path"] = exprun_path.as_posix()
    v["data_path"] = result_path.as_posix()
    return v


# ### Generate the exp/runs to be run

# In[81]:


expruns = []
# overall values
latent_sizes = [128, 256] # the possible latent sizes we consider
cnntypes = ["vgg19", "resnet50"] # the CNN architectures we consider
vit_types = ["vit_base"] #VIT Types, we can add vit-huge here as well
# vit_types = ["vit_base", "vit_large"] #VIT Types, we can add vit-huge here as well
# fusion_methods = ["concat_proj", "indiv_proj", "attention", "weighted_sum", "gated"]
fusion_types = ["concat_proj"]


# *******************************************
# generate the sensorprocessing models
# *******************************************
sps = [] # the list of the sensorprocessing models (exp/run)
for latent_size in latent_sizes:

    # generate the vae exprun
    exp_name = "sensorprocessing_conv_vae"
    run_name = f"sp_conv_vae_{latent_size}_0001"
    params = {}
    params["latent_size"] = latent_size
    params["epochs"] = epochs_sp
    params["training_data"] = sp_training_data
    params["validation_data"] = sp_validation_data
    exprun = generate_sp_conv_vae(
        exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name = run_name)
    exprun["latent_size"] = latent_size
    if do_VAE:
        sps.append(exprun)
        expruns.append(exprun)

    if do_VIT:
        for vit_type in vit_types:
            exp_name = "sensorprocessing_propriotuned_Vit"
            run_name = f"sp_{vit_type}_{latent_size}_0001"
            params = {}
            params["image_size"] = [224, 224]  # ViT needs 224x224!
            params["latent_size"] = latent_size
            params["epochs"] = epochs_sp
            params["training_data"] = sp_training_data
            params["validation_data"] = sp_validation_data
            params["vit_model"] = vit_type  # Pass as "vit_base" or "vit_large"
            params["class"] = "Vit"
            params["model"] = "VitProprioTunedRegression"
            params["freeze_feature_extractor"] = False
            params["loss"] = "MSELoss"
            params["learning_rate"] = 0.0001

            exprun = generate_sp_propriotuned_vit(
                exprun_path=exprun_path, result_path=result_path, params=params, exp_name=exp_name, run_name=run_name)
            exprun["latent_size"] = latent_size
            exprun["vittype"] = vit_type
            sps.append(exprun)
            expruns.append(exprun)

    # Generate multiview ViT models
    if do_VIT_MULTIVIEW:
        for vit_type in vit_types:
            for fusion_type in fusion_types:
                exp_name = "sensorprocessing_propriotuned_Vit_multiview"
                run_name = f"sp_{vit_type}_multiview_{fusion_type}_{latent_size}_0001"
                params = {}
                params["image_size"] = [224, 224]
                params["latent_size"] = latent_size
                params["epochs"] = epochs_sp
                params["num_views"] = num_views
                params["fusion_type"] = fusion_type

                # Create multiview training data with camera list
                params["training_data"] = [
                    [demopack_name, demo, multiview_cameras]
                    for demo in selection["sp_training"]
                ]
                params["validation_data"] = [
                    [demopack_name, demo, multiview_cameras]
                    for demo in selection["sp_validation"]
                ]

                params["vit_model"] = vit_type
                params["class"] = "Vit_multiview"
                params["model"] = "VitProprioTunedRegression"
                params["freeze_feature_extractor"] = False
                params["loss"] = "MSELoss"
                params["learning_rate"] = 0.0001

                exprun = generate_sp_propriotuned_vit_multiview(
                    exprun_path=exprun_path, result_path=result_path,
                    params=params, exp_name=exp_name, run_name=run_name
                )
                exprun["latent_size"] = latent_size
                exprun["vittype"] = vit_type
                exprun["fusion"] = fusion_type
                sps.append(exprun)
                expruns.append(exprun)
    # generate the propriotuned expruns
    for cnntype in cnntypes:
        exp_name = "sensorprocessing_propriotuned_cnn"
        run_name = f"sp_{cnntype}_{latent_size}_0001"
        params = {}
        params["image_size"] = image_size
        params["latent_size"] = latent_size
        params["epochs"] = epochs_sp
        params["training_data"] = sp_training_data
        params["validation_data"] = sp_validation_data
        if cnntype == "vgg19":
            if not do_VGG:
                continue
            params["class"] = "VGG19ProprioTunedSensorProcessing"
            params["model"] = "VGG19ProprioTunedRegression"
        elif cnntype == "resnet50":
            if not do_RESNET:
                continue
            params["class"] = "ResNetProprioTunedSensorProcessing"
            params["model"] = "ResNetProprioTunedRegression"
            params["freeze_feature_extractor"] = True
            params["reductor_step_1"] = 512
            params["proprio_step_1"] = 64
            params["proprio_step_2"] = 16
        else:
            raise Exception(f"Unknown cnntype {cnntype}")
        params["loss"] = "MSELoss" # alternative L1Loss
        params["learning_rate"] = 0.001
        # alternative
        exprun = generate_sp_propriotuned_cnn(
            exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name = run_name)
        exprun["latent_size"] = latent_size
        exprun["cnntype"] = cnntype
        sps.append(exprun)
        expruns.append(exprun)

    # FIXME: add here the ViT models

# *******************************************
# generate the proprioception models
# *******************************************
vpruns = []
vpruns_latent = {128:[], 256:[]}
for spexp, sprun,latent_size in [(a["experiment"],a["run"],a["latent_size"]) for a in sps]:
    print(spexp, sprun, latent_size)
    # *** generate the vp train expruns ***
    exp_name = "visual_proprioception"
    run_name = "vp_" + sprun[3:]
    vpruns.append(run_name)
    vpruns_latent[latent_size].append(run_name)
    params = {}
    params["name"] = run_name
    params["output_size"] = 6
    params["encoding_size"] = latent_size
# ✅ NEW: Use multiview data for multiview runs
    if "multiview" in sprun.lower():
        params["training_data"] = vp_training_data_multiview
        params["validation_data"] = vp_validation_data_multiview
    else:
        params["training_data"] = vp_training_data
        params["validation_data"] = vp_validation_data

    params["regressor_hidden_size_1"] = 64
    params["regressor_hidden_size_1"] = 64
    params["loss"] = "MSE"
    params["epochs"] = epochs_vp
    params["batch_size"] = 64
    # FIXME this is hackish, should not do it this way
    if "vae" in sprun.lower():
        params["sensor_processing"] = "ConvVaeSensorProcessing"
    elif "resnet" in sprun.lower():
        params["sensor_processing"] = "ResNetProprioTunedSensorProcessing"
    elif "vgg19" in sprun.lower():
        params["sensor_processing"] = "VGG19ProprioTunedSensorProcessing"
    elif "vit" in sprun.lower():  # Added for VIT
        params["sensor_processing"] = "VitSensorProcessing"
    elif "multiview" in sprun.lower():  # Aded for multi-VIT
        params["sensor_processing"] = "MultiViewVitSensorProcessing"
    else:
        raise Exception(f"Unexpected sprun {sprun}")

    params["sp_experiment"] = spexp
    params["sp_run"] = sprun

    exprun = generate_vp_train(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
    # *** generate the vp verify expruns FIXME: not implemented yet ***
    params_verify = {}

    expruns.append(exprun)
# *******************************************
# generate the comparisons: all, for latents 128 and 256
# *******************************************
exp_name = "visual_proprioception"
# all
run_name = "vp_comp_flow_all"
params = {}
params["name"] = run_name
params["tocompare"] = vpruns
exprun = generate_vp_compare(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
expruns.append(exprun)
# by latent
for latent_size in [128, 256]:
    run_name = f"vp_comp_flow_{latent_size}"
    params = {}
    params["name"] = run_name
    params["tocompare"] = vpruns_latent[latent_size]
    exprun = generate_vp_compare(exprun_path = exprun_path, result_path = result_path, params = params, exp_name = exp_name, run_name=run_name)
    expruns.append(exprun)



# ### Generate COMPARISON experiments

# In[82]:


# -----------------------------------------
# Generate COMPARISON experiments
# -----------------------------------------
exp_name = "visual_proprioception"

# Compare ALL models
run_name = "vp_comp_flow_all"
params = {}
params["name"] = run_name
params["tocompare"] = vpruns
params["robot_exp"] = "robot_al5d"
params["robot_run"] = "position_controller_00"
exprun = generate_vp_compare(
    exprun_path=exprun_path, result_path=result_path,
    params=params, exp_name=exp_name, run_name=run_name
)
expruns.append(exprun)



# Compare by latent size
for latent_size in latent_sizes:
    run_name = f"vp_comp_flow_{latent_size}"
    params = {}
    params["name"] = run_name
    params["tocompare"] = vpruns_latent[latent_size]
    params["robot_exp"] = "robot_al5d"
    params["robot_run"] = "position_controller_00"
    exprun = generate_vp_compare(
        exprun_path=exprun_path, result_path=result_path,
        params=params, exp_name=exp_name, run_name=run_name
    )
    expruns.append(exprun)

# Compare single-view only
if vpruns_singleview:
    run_name = "vp_comp_flow_singleview"
    params = {}
    params["name"] = run_name
    params["tocompare"] = vpruns_singleview
    params["robot_exp"] = "robot_al5d"
    params["robot_run"] = "position_controller_00"
    exprun = generate_vp_compare(
        exprun_path=exprun_path, result_path=result_path,
        params=params, exp_name=exp_name, run_name=run_name
    )
    expruns.append(exprun)

# Compare multi-view only
if vpruns_multiview:
    run_name = "vp_comp_flow_multiview"
    params = {}
    params["name"] = run_name
    params["tocompare"] = vpruns_multiview
    params["robot_exp"] = "robot_al5d"
    params["robot_run"] = "position_controller_00"
    exprun = generate_vp_compare(
        exprun_path=exprun_path, result_path=result_path,
        params=params, exp_name=exp_name, run_name=run_name
    )
    expruns.append(exprun)


# ### RUN THE FLOW

# In[83]:


# =============================================================================
# RUN THE FLOW
# =============================================================================

print(f"***Starting automated running of the flow.\n The path for the output notebooks is\n{result_path}")
print(f"Total experiments to run: {len(expruns)}")

for exprun in tqdm.tqdm(expruns):
    print(f"***Automating {exprun['notebook']} :\n {exprun['experiment']}/{exprun['run']}")
    notebook_path = pathlib.Path("..", exprun["notebook"])
    output_filename = f"{notebook_path.stem}_{exprun['experiment']}_{exprun['run']}_output{notebook_path.suffix}"
    print(f"--> {output_filename}")

    # Parameters to pass to the notebook
    params = {}
    params["experiment"] = exprun["experiment"]
    params["run"] = exprun["run"]
    params["external_path"] = exprun["external_path"]
    params["data_path"] = exprun["data_path"]
    output_path = pathlib.Path(result_path, output_filename)

    try:
        papermill.execute_notebook(
            notebook_path,
            output_path.absolute(),
            cwd=notebook_path.parent,
            parameters=params
        )
    except Exception as e:
        print(f"There was an exception {e}")

print("Flow completed!")

