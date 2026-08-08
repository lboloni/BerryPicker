"""
rcco_lstm.py

An rcco implementing an LSTM architecture
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

import pathlib
import torch

class RCCO_RobotDrive_LSTM_Network(nn.Module):
    """
    Simple LSTM-based bahavior cloning module. Mostly specified through the exp/exp_sp
    """
    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super().__init__()
        self.state = None
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, self.output_size)

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
    

class RCCO_RobotDrive_LSTM(AbstractRCComponent):
    """An rcco that implements an LSTM based controller that directly drives the robot. The input is a latent encoding $z$, while the output is a robot control $a$.

    This is a version that is forming a sequence internally of a maximum length, but it does not use the LSTM state
    """
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["z"] = None
        self.outputs["a"] = None
        self.model = RCCO_RobotDrive_LSTM_Network(
            input_size=self.exp["z_size"],
            output_size=self.exp["a_size"],
            num_layers=self.exp["num_layers"],
            hidden_size=self.exp["hidden_size"])
        model_path = pathlib.Path(self.exp.data_dir(), self.exp["model_file"])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))   

    def forward(self, x):
        """FIXME: taken from the previous code, figure it out cleanly"""
        # x: [batch_size, sequence_length, latent_size]
        out, _ = self.lstm(x)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        return out  # Predicted next vector

    def forward_keep_state(self, x):
        """FIXME: taken from the previous code, figure it out cleanly"""
        # x: [batch_size, sequence_length, latent_size]
        out, self.state = self.lstm(x, self.state)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        return out  # Predicted next vector

    def propagate(self):
        """Processes the LSTM"""
        self.forward(self.inputs["z"])
