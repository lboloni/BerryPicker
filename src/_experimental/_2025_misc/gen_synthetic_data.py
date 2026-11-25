"""
gen_synthetic_data.py

Generate training data for experiments with sequences such as LSTM and transformers.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def generate_training_sequence(total_length, latent_size, y_size=5): 
    """
    Generates two sequences of specified length and width with some of the cells being predictable (being based on sine and cosine values etc). 
    The x_seq is going to be used for simple prediction of an evolving stream. The y_seq is going to be used to check whether a different shaped stream that is fully determined by the x_seq can be predicted. 
    """
    assert latent_size > 6
    # the x sequence - initialized with random
    x_seq = torch.rand(latent_size, total_length)
    # x_2 sin(i* pi/10)  
    x_seq[2] = torch.linspace(0, 2 * torch.pi * total_length / 10.0, steps = total_length)
    x_seq[2].sin_()

    # x_3 cos(i* pi/10)
    x_seq[3] = torch.linspace(0, 2 * torch.pi * total_length / 10.0, steps = total_length)
    x_seq[3].cos_()

    # x_4 sin(i* pi/20)
    x_seq[4] = torch.linspace(0, 2 * torch.pi * total_length / 20.0, steps = total_length)
    x_seq[4].sin_()

    # x_5 cos(i* pi/20)
    x_seq[5] = torch.linspace(0, 2 * torch.pi * total_length / 20.0, steps = total_length)
    x_seq[5].cos_()

    # initialize the x_6 with some randomness with 1 or -1
    for i in range(total_length):
        val = x_seq[6, i].item()
        x_seq[6, i] = 0
        if val < 0.1:
            x_seq[6, i] = -1
        if val > 0.9:
            x_seq[6, i] = 1

    # the y_sequence
    y_seq = torch.rand(y_size, total_length)
    # y_1 sin(i-shift)/10
    shift = -0.2 * 2 * torch.pi 
    y_seq[1] = torch.linspace(shift, shift+ 2 * torch.pi * total_length / 10.0, steps = total_length)
    y_seq[1].cos_()

    acc2 = 0
    acc5 = 0
    val4 = 0
    
    for i in range(total_length):
        #y_seq[0, i] = x_seq[0, i]
        #y_seq[1, i] = x_seq[1, i]
 
        # y_2 sum of x_2
        acc2 += x_seq[2, i]
        y_seq[2, i] = acc2

        # y_3 sum of x_5
        acc5 += x_seq[5, i]
        y_seq[3, i] = acc5
        
        # y_4 1 or -1 depending on whether the previous non-zero x_6 was 1 or -1
        if x_seq[6, i] < 0.1:
            val4 = -1
        if x_seq[6, i] > 0.1:
            val4 = 1
        y_seq[4, i] = val4

    # transpose them such that the first index gets the value at a 
    # certain timecode. 
    # FIXME: maybe this could have been initialized like this, but it is ok
    x_seq.transpose_(0, 1)
    y_seq.transpose_(0, 1)
    return x_seq, y_seq


def create_training_sequence_prediction(x_seq, sequence_length):
    """Create supervised training data from the single long sequence. In this data, from a string of length sequence_length we are predicting the next item."""
    # Prepare training data
    total_length = x_seq.shape[0]
    inputs = []
    targets = []
    for i in range(total_length - sequence_length):
        # Input is a subsequence of length `sequence_length`
        input_seq = x_seq[i:i + sequence_length]  # Shape: [sequence_length, latent_size]
        
        # Target is the next vector after the input sequence
        target = x_seq[i + sequence_length]       # Shape: [latent_size]
        
        # Append to lists
        inputs.append(input_seq)
        targets.append(target)

    # Convert lists to tensors for training
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets


def create_training_sequence_xy(x_seq, y_seq, sequence_length):
    """Create supervised training data from two sequences. In this data, from a string of length sequence_length we are predicting the next item in the other sequence"""
    # Prepare training data
    total_length = x_seq.shape[0]
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
        inputs.append(input_seq)
        targets.append(target)

    # Convert lists to tensors for training
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets