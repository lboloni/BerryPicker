"""
bc_LSTM.py

LSTM-based models for behavior cloning. They are mapping visual encoding to action. 

FIXME: add (1) models that also take into account tasks (2) models that predict sequences. 
"""

import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn


class LSTMXYPredictor(nn.Module):
    """
    This is the architecture created by chatgpt. 
    Uses an input of the size of the latent encoding and the output of the size of the action space (normally 6).
    The output is 
    """
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(LSTMXYPredictor, self).__init__()
        self.state = None
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, sequence_length, latent_size]
        out, _ = self.lstm(x)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        return out  # Predicted next vector

    def forward_keep_state(self, x):
        """Forward, while keeping state"""
        # x: [batch_size, sequence_length, latent_size]
        out, self.state = self.lstm(x, self.state)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        return out  # Predicted next vector

class LSTMResidualController(nn.Module):
    """
    LSTM w/ 3 layers and skip connections.
    This is an attempt to recreate the LSTM model from the Rouhollah 2020 paper. 
    
    FIXME: 
    * In its current form, this is sequence prediction, this needs to be changed to cover stuff. 
    * In its current form, it does not have an MDM at the end. 
    """
    def __init__(self, latent_size, hidden_size, output_size):
        super(LSTMResidualController, self).__init__()
        self.lstm_1 = nn.LSTM(latent_size, hidden_size, num_layers=1, batch_first=True)

        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.lstm_3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, sequence_length, latent_size]
        out_1, _ = self.lstm_1(x)
        residual = out_1
        out_2, _ = self.lstm_2(out_1)
        out_2 = out_2 + residual
        residual = out_2
        out_3, _ = self.lstm_3(out_2)
        out_3 = out_3 + residual

        # LSTM output shape: [batch_size, sequence_length, hidden_size]
        out = self.fc(out_3[:, -1, :])  # Take last time step output and pass through the fully connected layer
        return out  # Predicted next vector
    
