"""
bc_LSTM_MDN.py

This model implements the configuration for the paper

R. Rahmatizadeh, P. Abolghasemi, L. Bölöni, and S. Levine. Vision-Based Multi-Task Manipulation for Inexpensive Robots Using End-To-End Learning from Demonstration. In Proc. of International Conference on Robotics and Automation (ICRA-2018), May 2018

It combines LSTM with skip connections and an MDN-based output.

"""

import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn
from mdn import MDN, mdn_loss


class bc_LSTM_MDN(nn.Module):
    """
    LSTM w/ 3 layers and skip connections (residuals), followed by an MDN layer.
    """
    def __init__(self, exp, exp_sp):
        super().__init__()
        self.stochastic = True

        self.input_size = exp_sp["latent_size"]
        self.output_size = exp["control_size"]  # deg. of freedom
        self.hidden_size = exp["hidden_size"]

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        exp_mdn = Config().get_experiment(exp["exp_mdn"], exp["run_mdn"])
        self.mdn = MDN(exp_mdn)

    def forward(self, x):
        """The residual nature of the LSTM is implemented through the ways this forward function adds the values forward up the stack. The output of this goes into the mdn. This function returns a mixed density probability function"""
        # x: [batch_size, sequence_length, latent_size]
        out_1, _ = self.lstm_1(x)
        residual = out_1
        out_2, _ = self.lstm_2(out_1)
        out_2 = out_2 + residual
        residual = out_2
        out_3, _ = self.lstm_3(out_2)
        out_3 = out_3 + residual
        # LSTM output shape: [batch_size, sequence_length, hidden_size]
        # out = self.fc(out_3[:, -1, :])  # Take last time step output and pass through the fully connected layer
        out = out_3[:, -1, :]
        mu, sigma, pi = self.mdn(out)
        return mu, sigma, pi  # Predicted next vector
    
    def forward_and_sample(self, x):
        """Forwards through the model, and then performs a sample from the output, returning a single value. 
        The random seed is the torch, one can set it with torch.manual_seed
        """
        mu, sigma, pi = self.forward(x)
        samples = self.mdn.sample(1, mu, sigma, pi)
        #print(samples.shape)
        #return samples[0]
        return torch.tensor(samples[0]).to(x.device)