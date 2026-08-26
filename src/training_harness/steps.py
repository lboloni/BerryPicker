"""
steps.py

Epoch-step factories for the shared training harness.

``make_epoch_steps`` returns the ``(model_training_step, model_eval_step)``
pair expected by :func:`training_harness.load_or_train` /
:func:`training_harness.train_model`. It handles both single-view batches
(``(tensor, target)``) and multi-view batches (``(list_of_view_tensors,
target)``, see ``helper_training_data.collate_multiview``), moves everything
to the configured device, and optionally clips gradients.
"""

import torch

from exp_run_config import Config

Config.PROJECTNAME = "BerryPicker"


def move_batch_to_device(batch_inputs, device):
    """Move a tensor, or a list/tuple of view tensors, to ``device``."""
    if isinstance(batch_inputs, (list, tuple)):
        return [view.to(device, non_blocking=True) for view in batch_inputs]
    return batch_inputs.to(device, non_blocking=True)


def make_epoch_steps(criterion, train_loader, validation_loader, *, grad_clip_norm=None):
    """Return ``(model_training_step, model_eval_step)`` closures.

    Each closure runs one full epoch over its loader and returns the mean
    batch loss, which is the contract of the training harness.
    """

    def model_training_step(model, optimizer):
        device = Config().runtime["device"]
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = move_batch_to_device(batch_inputs, device)
            batch_targets = batch_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def model_eval_step(model):
        device = Config().runtime["device"]
        model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in validation_loader:
                batch_inputs = move_batch_to_device(batch_inputs, device)
                batch_targets = batch_targets.to(device, non_blocking=True)
                total_loss += criterion(model(batch_inputs), batch_targets).item()
                batch_count += 1
        return total_loss / max(batch_count, 1)

    return model_training_step, model_eval_step


def evaluate_loss(model, criterion, loader):
    """Mean loss of ``model`` over ``loader`` (single- or multi-view batches)."""
    _, eval_step = make_epoch_steps(criterion, loader, loader)
    return eval_step(model)
