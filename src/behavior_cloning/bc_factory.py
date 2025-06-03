"""
bc_factory.py

Creating different models for behavior cloning based on the specification in the exp/run
"""

import torch.nn as nn
import torch.optim as optim
from bc_LSTM import LSTMXYPredictor, LSTMResidualController

def create_bc_model(exp, spexp, device):
    latent_size = spexp["latent_size"]
    output_size = exp["control_size"]  # degrees of freedom in the robot

    if exp["controller"] == "LSTMXYPredictor":

        num_layers = exp["controller_num_layers"]
        hidden_size = exp["controller_hidden_size"] 

        # Instantiate model, loss function, and optimizer
        model = LSTMXYPredictor(latent_size=latent_size, hidden_size=hidden_size, output_size = output_size, num_layers=num_layers)
        model = model.to(device)
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, criterion, optimizer

    if exp["controller"] == "LSTMResidualController":
        hidden_size = exp["controller_hidden_size"] 
        # Instantiate model, loss function, and optimizer
        model = LSTMResidualController(latent_size=latent_size, hidden_size=hidden_size, output_size = output_size)
        model = model.to(device)
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, criterion, optimizer

    raise Exception(f"Unknown controller specified {exp['controller']}")
