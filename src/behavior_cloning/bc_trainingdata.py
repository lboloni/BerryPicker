"""
bc_trainingdata.py

Create training data for behavior cloning starting from demonstrations, and save it into pre-calculated files. 

The training data calculated here is always mapping latent representations [z], or sequences of latents 
[z(t-k)...z(t)] to next action a(t+1)

The passed sensor processing component is used to calculate the sp 
that calculates the latent
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
#import helper
import pathlib
#from pprint import pformat
#import numpy as np
#from sensorprocessing.sp_helper import load_picturefile_to_tensor
from robot.al5d_position_controller import RobotPosition
from sensorprocessing.sp_helper import get_transform_to_sp
from sensorprocessing.sp_factory import create_sp
from demonstration.demonstration import Demonstration


def create_trainingpair_prediction(x_seq: torch.Tensor, y_seq: torch.Tensor, sequence_length):
    """Create a supervised training pair from two sequences, where the objective is to predict the next item from y_seq from the current or a sequence from x. 
    If sequence lenght = 0, we predict y_t+1 from x_t
    Otherwise, we predit y_t+1 from [x_t-seql, ... x_t]
    x_seq and y_seq are tensors
    Returns the results as tensors
    """
    total_length = x_seq.shape[0]
    inputs_list = []
    targets_list = []
    if sequence_length > 0: # LSTM style data, sequence to next
        for i in range(total_length - sequence_length):
            # Input is a subsequence of length `sequence_length`
            input_seq = x_seq[i:i + sequence_length]
            # Shape: [sequence_length, latent_size]
            # Target is the next item after the input sequence
            target = y_seq[i + sequence_length]
            inputs_list.append(torch.tensor(input_seq))
            targets_list.append(torch.tensor(target))
    else: # just pairs of current value, next action
        # Input is a subsequence of length `sequence_length`
        for i in range(total_length - 1):
            x = x_seq[i]
            y = y_seq[i + 1]
            inputs_list.append(torch.tensor(x))
            targets_list.append(torch.tensor(y))
    # Convert lists to tensors for training
    inputs_tensor = torch.stack(inputs_list)   # Shape: [num_pairs, sequence_length, latent_size]
    targets_tensor = torch.stack(targets_list) # Shape: [num_pairs, latent_size]
    return inputs_tensor, targets_tensor


def create_trainingdata_bc(exp, exp_sp, exp_robot, device):
    """Creates training data for training and validation with the demonstrations specified in the exp/run. Caches the results into the input and target files specified in the exp/run. Remove those files to recalculate."""

    exp.start_timer("data_preparation")

    input_path = pathlib.Path(exp.data_dir(), "training_input.pth")
    target_path = pathlib.Path(exp.data_dir(), "training_target.pth")

    if not input_path.exists():
        all_demos_inputs_list = []
        all_demos_targets_list = []
        # Create the sp object described in the experiment
        sp = create_sp(exp_sp, device)
        transform = get_transform_to_sp(exp_sp)
        for val in exp["training_data"]: # for all demonstrations
            run, demo_name, camera = val
            exp_demo = Config().get_experiment("demonstration", run)
            demo = Demonstration(exp_demo, demo_name)
            # read the a and z 
            inputs_list = []
            targets_list = []
            for i in range(demo.metadata["maxsteps"]-1): # -1 because of lookahead
                sensor_readings, _ = demo.get_image(i, device=device, transform=transform, camera=camera)                
                # inputlist.append(sensor_readings[0])
                z = sp.process(sensor_readings)
                inputs_list.append(torch.from_numpy(z))
                # the action we are choosing, is the next one
                a = demo.get_action(i+1)
                rp = RobotPosition.from_vector(exp_robot, a)
                anorm = rp.to_normalized_vector(exp_robot)        
                targets_list.append(torch.from_numpy(anorm))
            inputs_tensor = torch.stack(inputs_list)
            targets_tensor = torch.stack(targets_list)
            # at this point we have two tensors that are aligned demonstrations
            all_demos_inputs_tensor, all_demos_targets_tensor = create_trainingpair_prediction(inputs_tensor, targets_tensor, sequence_length=exp["sequence_length"])
            all_demos_inputs_list.append(all_demos_inputs_tensor)
            all_demos_targets_list.append(all_demos_targets_tensor)
        all_demos_inputs_tensor = torch.cat(all_demos_inputs_list)
        all_demos_targets_tensor = torch.cat(all_demos_targets_list)
        # shuffle
        perm = torch.randperm(all_demos_inputs_tensor.size(0))
        all_demos_inputs_tensor = all_demos_inputs_tensor[perm]
        all_demos_targets_tensor = all_demos_targets_tensor[perm]
        # save
        torch.save(all_demos_inputs_tensor, input_path)
        torch.save(all_demos_targets_tensor, target_path)
    else: # just load the cached data
        all_demos_inputs_tensor = torch.load(input_path, weights_only=True)
        all_demos_targets_tensor = torch.load(target_path, weights_only=True)


    # split the training and validation data by first shuffling the pairs
    # then separating them into the 2/3 - 1/3 ratio
    rows = torch.randperm(all_demos_inputs_tensor.shape[0]) 
    shuffled_inputs = all_demos_inputs_tensor[rows]
    shuffled_targets = all_demos_targets_tensor[rows]

    training_size = int( all_demos_inputs_tensor.shape[0] * 0.67 )

    retval = {}
    retval["z_train"] = shuffled_inputs[:training_size].to(device)
    retval["a_train"] = shuffled_targets[:training_size].to(device)
    retval["z_validation"] = shuffled_inputs[training_size:].to(device)
    retval["a_validation"] = shuffled_targets[training_size:].to(device) 
    exp.end_timer("data_preparation")
    return retval