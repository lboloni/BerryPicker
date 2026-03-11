"""
rcco_lstm.py

An rcco implementing an LSTM architecture
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent
import torch

class RCCO_LSTM(AbstractRCComponent):
    """An rcco that implements an LSTM based controller. The input is a latent encoding $z$, while the output is a robot control $a$
    
    NOTE: this was copied from bc_LSTM
    """
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["z"] = None
        self.outputs["a"] = None
        self.stochastic = False
        self.input_size = self.exp["z_size"]
        self.output_size = self.exp["a_size"]  # deg. of freedom
        self.num_layers = self.exp["num_layers"]
        self.hidden_size = self.exp["hidden_size"]
        self.keep_state = self.exp["keep_state"]
        self.state = None
        ## create the neural network
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size,self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

    def propagate(self):
        if not self.keep_state:
            # x: [batch_size, sequence_length, latent_size]
            out, _ = self.lstm(x)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
            out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        else: # keep state
            out, self.state = self.lstm(x, self.state)  # LSTM output shape: [batch_size, sequence_length, hidden_size]
            out = self.fc(out[:, -1, :])  # Take last time step output and pass through the fully connected layer
        ## create the outputs
        self.outputs["a"] = torch.squeeze(out).cpu().numpy()
        self.outputs["state"] = torch.squeeze(self.state).cpu().numpy()        
