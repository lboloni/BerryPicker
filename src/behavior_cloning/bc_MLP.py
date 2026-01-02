"""
bc_MLP.py

MLP-based models for behavior cloning. 
"""

import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn

class bc_MLP(nn.Module):
    """An MLP model for behavior cloning. Mostly specified through the exp/run"""

    def __init__(self, exp, exp_sp):
        super().__init__()
        self.stochastic = False

        self.input_size = exp_sp["latent_size"]
        self.output_size = exp["control_size"]  # deg. of freedom
        self.hidden_layers = exp["hidden_layers"]
        self.model = nn.Sequential()
        for i in range(self.hidden_layers+1):
            prev = exp[f"hidden_layer_{i}"] if i > 0 else self.input_size
            current = exp[f"hidden_layer_{i+1}"] if i < self.hidden_layers else self.output_size
            name = f"hidden_layer_{i+1}" if i < self.hidden_layers else "output_layer"
            self.model.add_module(name, nn.Linear(prev, current))
            if i != self.hidden_layers - 1:
                self.model.add_module(f"relu-{i}", nn.ReLU())

    def forward(self, x):
        return self.model(x)