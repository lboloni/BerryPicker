# Future refactoring: save and restore all RNG states with each checkpoint to
# make interrupted-and-resumed runs perfectly reproducible.

"""Shared deterministic-run configuration for notebooks and scripts."""

import os
import random

import numpy as np
import torch


def configure_deterministic_run(seed=777):
    """Configure the process RNGs and PyTorch for deterministic execution."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    return seed
