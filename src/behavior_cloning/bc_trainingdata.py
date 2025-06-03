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


def create_RNN_training_sequence_xy(x_seq, y_seq, sequence_length):
    """Create supervised training data for RNNs such as LSTM from two sequences. In this data, from a string of length sequence_length in x_seq we are predicting the next item in y_seq.
    x_seq and y_seq are tensors
    Returns the results as tensors
    """
    # Prepare training data
    total_length = x_seq.shape[0]
    #total_length = len(x_seq)
    #assert total_length == len(y_seq)
    inputs = []
    targets = []
    for i in range(total_length - sequence_length):
        # Input is a subsequence of length `sequence_length`
        input_seq = x_seq[i:i + sequence_length]
        # Shape: [sequence_length, latent_size]

        # Target is the next vector after the input sequence
        target = y_seq[i + sequence_length]
        # Shape: [output_size]

        # Append to lists
        inputs.append(torch.tensor(input_seq))
        targets.append(torch.tensor(target))

    # Convert lists to tensors for training
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets


def create_bc_training_and_validation(exp, spexp, device):
    """Creates training data for training and validation with the demonstrations specified in the exp/run. Caches the results into the input and target files specified in the exp/run. Remove those files to recalculate."""

    exp.start_timer("data_preparation")

    input_path = pathlib.Path(exp.data_dir(), "training_input.pth")
    target_path = pathlib.Path(exp.data_dir(), "training_target.pth")

    if input_path.exists():
        inputs = torch.load(input_path, weights_only=True)
        targets = torch.load(target_path, weights_only=True)
    else:
        all_demos_inputs = []
        all_demos_targets = []
        # Create the sp object described in the experiment
        sp = create_sp(spexp, device)
        transform = get_transform_to_sp(spexp)
        for val in exp["training_data"]: # for all demonstrations
            run, demo_name, camera = val
            exp_demo = Config().get_experiment("demonstration", run)
            demo = Demonstration(exp_demo, demo_name)
            # read the a and z 
            inputlist = []
            targetlist = []
            for i in range(demo.metadata["maxsteps"]-1): # -1 because of lookahead
                sensor_readings, _ = demo.get_image(i, device=device, transform=transform, camera=camera)                
                # inputlist.append(sensor_readings[0])
                z = sp.process(sensor_readings)
                inputlist.append(torch.from_numpy(z))
                # the action we are choosing, is the next one
                a = demo.get_action(i+1)
                rp = RobotPosition.from_vector(a)
                anorm = rp.to_normalized_vector()        
                targetlist.append(torch.from_numpy(anorm))
            inputlist_tensor = torch.stack(inputlist)
            targetlist_tensor = torch.stack(targetlist)
            inputs, targets = create_RNN_training_sequence_xy(inputlist_tensor, targetlist_tensor, sequence_length=exp["sequence_length"])
            all_demos_inputs.append(inputs)
            all_demos_targets.append(targets)
        inputs = torch.cat(all_demos_inputs)
        targets = torch.cat(all_demos_targets)
        torch.save(inputs, input_path)
        torch.save(targets, target_path)


    # Separate the training and validation data. 
    # We will be shuffling the demonstrations 
    # rows = torch.randperm(inputs.size(0)) 
    rows = torch.randperm(inputs.shape[0]) 
    shuffled_inputs = inputs[rows]
    shuffled_targets = targets[rows]

    training_size = int( inputs.shape[0] * 0.67 )

    retval = {}
    # FIXME: why 1:training size???
    # retval["z_train"] = shuffled_inputs[1:training_size]
    # retval["a_train"] = shuffled_targets[1:training_size]
    retval["z_train"] = shuffled_inputs[:training_size].to(device)
    retval["a_train"] = shuffled_targets[:training_size].to(device)
    retval["z_validation"] = shuffled_inputs[training_size:].to(device)
    retval["a_validation"] = shuffled_targets[training_size:].to(device) 
    exp.end_timer("data_preparation")
    return retval