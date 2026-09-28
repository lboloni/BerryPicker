"""
bc_factory.py

Creating different models for behavior cloning based on the specification in the exp/run
"""

import socket
import torch.nn as nn
import torch.optim as optim
from bc_MLP import bc_MLP
from bc_LSTM import bc_LSTM, bc_LSTM_Residual
from bc_LSTM_MDN import bc_LSTM_MDN, mdn_loss

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"


def create_bc_model(exp, exp_sp):
    if exp["controller"] == "bc_MLP":
        model = bc_MLP(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM":
        model = bc_LSTM(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM_Residual":
        model = bc_LSTM_Residual(exp, exp_sp)
    elif exp["controller"] == "bc_LSTM_MDN":
        model = bc_LSTM_MDN(exp, exp_sp)
    else:
        raise Exception(f"Unknown controller specified {exp['controller']}")    
    model.to(Config().runtime["device"])
    criterion = create_criterion(exp)
    optimizer = create_optimizer(exp, model)
    return model, criterion, optimizer


def create_criterion(exp):
    if exp["loss"] == "MSELoss":
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = criterion.to(Config().runtime["device"])
    elif exp["loss"] == "MDNLoss":
        criterion = mdn_loss 
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

