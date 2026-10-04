# Future refactoring: save and restore all RNG states with each checkpoint to
# make interrupted-and-resumed runs perfectly reproducible.

"""Shared run configuration for notebooks and scripts.

Only the RNGs are seeded; deterministic algorithms are not enforced, as they
cost speed on CUDA and are not supported for all operations on MPS.
"""

import random

import numpy as np
import torch


def configure_deterministic_run(seed=777):
    """Seed the process RNGs and let cuDNN pick the fastest algorithms."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    return seed
