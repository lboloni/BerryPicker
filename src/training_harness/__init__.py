"""
__init__.py

Public API for shared model-training and checkpoint lifecycle helpers.
"""

from .checkpoints import (
    CheckpointStore,
    TrainingState,
    find_latest_checkpoint,
    model_available,
)
from .runner import (
    load_or_train,
    train_model,
    train_with_checkpoints,
)
from .steps import (
    evaluate_loss,
    make_epoch_steps,
    move_batch_to_device,
)

__all__ = [
    "CheckpointStore",
    "TrainingState",
    "evaluate_loss",
    "find_latest_checkpoint",
    "load_or_train",
    "make_epoch_steps",
    "model_available",
    "move_batch_to_device",
    "train_model",
    "train_with_checkpoints",
]
