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
    def __init__(self, exp, spexp):
        super().__init__()

        self.input_size = spexp["latent_size"]
        self.output_size = exp["control_size"]  # deg. of freedom
        self.hidden_size = exp["hidden_size"]

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.lstm_3 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        experiment = exp["mdn_experiment"]
        run = exp["mdn_run"]
        exp_mdn = Config().get_experiment(experiment, run)
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

        mu, sigma, pi = self.mdn(out_3)

        # LSTM output shape: [batch_size, sequence_length, hidden_size]
        # out = self.fc(out_3[:, -1, :])  # Take last time step output and pass through the fully connected layer

        # FIXME: we need the probabilistic output here, but we also need a sampling function. 

        return mu, sigma, pi  # Predicted next vector
    
    def forward_and_sample(self, x):
        """Forwards through the model, and then performs a sample from the output, returning a single value. 
        FIXME: we need some way to control the random seed.
        """
        mu, sigma, pi = self.mdn(x)
        samples = self.mdn.sample(1, mu, sigma, pi)
        print(samples.shape)
        return samples[0]
