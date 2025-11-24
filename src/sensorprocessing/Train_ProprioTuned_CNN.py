#!/usr/bin/env python
# coding: utf-8

# # Train a proprioception-tuned CNN
# 
# We create a sensor processing model using CNN-based visual encoding finetuned with proprioception.
# 
# We create an encoding for the robot starting from a pretrained CNN model. As the feature vector of this is still large (eg 512 * 7 * 7), we reduce this to the encoding with an MLP. 
# 
# We finetune the encoding with information from proprioception.  
# 
# The sensor processing object associated with the network trained like this is in sensorprocessing/sp_propriotuned_cnn.py

# In[17]:


import sys
sys.path.append("..")

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from demonstration.demonstration import Demonstration

import sensorprocessing.sp_helper as sp_helper
from sensorprocessing.sp_propriotuned_cnn import VGG19ProprioTunedRegression, ResNetProprioTunedRegression
from robot.al5d_position_controller import RobotPosition

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# In[18]:


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

# In[19]:


# *** Initialize the variables with default values
# *** This cell should be tagged as parameters
# *** If papermill is used, some of the values will be overwritten

# If it is set to discard-old, the exprun will be recreated from scratch
creation_style = "exist-ok"

# experiment = "sensorprocessing_propriotuned_cnn"
# run = "vgg19_128"
# run = "resnet50_128"
# run = "vgg19_256"
# run = "resnet50_256"
# run = "boo"
# If not None, set the epochs to something different than the exp
epochs = None

# If not None, set an external experiment path
# external_path = None
# If not None, set an output path
# data_path = None

# Dr. Boloni's path
#external_path = pathlib.Path(Config()["experiment_external"])
# Sahara's path
# external_path = pathlib.Path("/home/sa641631/SaharaBerryPickerData/experiment_data")


# In[20]:


# create the necessary exp/run objects

if external_path:
    external_path = pathlib.Path(external_path)
    assert external_path.exists()
    Config().set_exprun_path(external_path)
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")
if data_path:
    data_path = pathlib.Path(data_path)
    assert data_path.exists()
    Config().set_results_path(data_path)

# This is an example of how to run an exprun variant
# Config().create_exprun_variant("sensorprocessing_propriotuned_cnn","resnet50_128", {"epochs": 17}, new_run_name="boo")

# The experiment/run we are going to run: the specified model will be created
exp = Config().get_experiment(experiment, run, creation_style=creation_style)
exp_robot = Config().get_experiment(exp["robot_exp"], exp["robot_run"])


# ### Create regression training data (image to proprioception)
# The training data (X, Y) is all the pictures from a demonstration with the corresponding proprioception data. 

# In[21]:


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


# ### Create a model that performs proprioception regression

# In[22]:


def train_and_save_proprioception_model(model, criterion, optimizer, modelfile, train_loader, test_loader, device="cpu", epochs=20):
    """Trains and saves the proprioception model
    """
    model = model.to(device)
    criterion = criterion.to(device)
    # Training loop
    num_epochs = epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model.forward(batch_X)
            loss = criterion(predictions, batch_y)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    # Evaluate the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    torch.save(model.state_dict(), modelfile)


# In[24]:


modelfile = pathlib.Path(
    exp["data_dir"], exp["proprioception_mlp_model_file"])
print(str(modelfile))
if modelfile.exists():
    print("*** Train-Propriotuned-CNN ***: NOT training; model already exists")
    # model.load_state_dict(torch.load(modelfile))
else:
    if exp['model'] == 'VGG19ProprioTunedRegression':
        model = VGG19ProprioTunedRegression(exp, device)
    elif exp['model'] == 'ResNetProprioTunedRegression':
        model = ResNetProprioTunedRegression(exp, device)
    else:
        raise Exception(f"Unknown model {exp['model']}")
    if exp['loss'] == 'MSELoss':
        criterion = nn.MSELoss()
    elif exp['loss'] == 'L1Loss':
        criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])

    tr = load_images_as_proprioception_training(exp, exp_robot)
    inputs_training = tr["inputs_training"]
    targets_training = tr["targets_training"]
    inputs_validation = tr["inputs_validation"]
    targets_validation = tr["targets_validation"]

    # Create DataLoaders for batching
    batch_size = exp['batch_size']
    train_dataset = TensorDataset(inputs_training, targets_training)
    test_dataset = TensorDataset(inputs_validation, targets_validation)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if not epochs:
        epochs = exp["epochs"]

    train_and_save_proprioception_model(model, criterion, optimizer, modelfile, train_loader, test_loader, device=device, epochs=epochs)


# In[ ]:




