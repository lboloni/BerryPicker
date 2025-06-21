"""
bc_factory.py

Creating different models for behavior cloning based on the specification in the exp/run
"""

import torch.nn as nn
import torch.optim as optim
from bc_MLP import bc_MLP
from bc_LSTM import bc_LSTM, bc_LSTM_Residual
from bc_LSTM_MDN import bc_LSTM_MDN, mdn_loss

def create_bc_model(exp, spexp, device):
    if exp["controller"] == "bc_MLP":
        model = bc_MLP(exp, spexp)
    elif exp["controller"] == "bc_LSTM":
        model = bc_LSTM(exp, spexp)
    elif exp["controller"] == "bc_LSTM_Residual":
        model = bc_LSTM_Residual(exp, spexp)
    elif exp["controller"] == "bc_LSTM_MDN":
        model = bc_LSTM_MDN(exp, spexp)
    else:
        raise Exception(f"Unknown controller specified {exp['controller']}")    
    model.to(device)
    criterion = create_criterion(exp, device)
    optimizer = create_optimizer(exp, model)
    return model, criterion, optimizer


def create_criterion(exp, device):
    if exp["loss"] == "MSELoss":
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = criterion.to(device)
    elif exp["loss"] == "MDNLoss":
        criterion == mdn_loss() 
        criterion = criterion.to(device)
        # Note that this is a bit different in parameters
    else:
        raise Exception(f"Loss function {exp['loss']} not implemented yet")
    return criterion

def create_optimizer(exp, model):
    if exp["optimizer"] == "Adam":
        lr = exp["optimizer_lr"]
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer {exp['optimizer']} not implemented yet")
    return optimizer