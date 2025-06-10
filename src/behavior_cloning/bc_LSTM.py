"""
bc_LSTM.py

LSTM-based models for behavior cloning. 
They are mapping sequences of latent encodings to the next action. 
"""

import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn


class bc_LSTM(nn.Module):
    """
    Simple LSTM-based bahavior cloning module. Mostly specified through the exp/spexp
    """
    def __init__(self, exp, spexp):
        super().__init__()
        self.input_size = spexp["latent_size"]
        self.output_size = exp["control_size"]  # deg. of freedom
        self.num_layers = exp["num_layers"]
        self.hidden_size = exp["hidden_size"]
        self.state = None
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

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

class bc_LSTM_Residual(nn.Module):
    """
    LSTM w/ 3 layers and skip connections (residuals), followed by a fully connected layer. 
    This is an attempt to recreate the LSTM model from the Rouhollah 2020 paper, without the MDN.
    """
    def __init__(self, exp, spexp):
        super().__init__()

        self.input_size = spexp["latent_size"]
        self.output_size = exp["control_size"]  # deg. of freedom
        self.hidden_size = exp["hidden_size"]

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """The residual nature of the LSTM is implemented through the ways this forward function adds the values forward up the stack."""
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
    

