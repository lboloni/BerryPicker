"""
checkpoints.py

Checkpoint storage and restoration for the shared training harness.
"""

from dataclasses import dataclass
import re
from pathlib import Path

import torch


def _model_file(exp):
    return Path(exp["data_dir"]) / exp["proprioception_mlp_model_file"]


def model_available(exp):
    """Return whether the final model file configured by ``exp`` exists."""
    return _model_file(exp).is_file()


def find_latest_checkpoint(model_dir):
    """Return the most recent epoch checkpoint and its epoch number."""
    checkpoint_dir = Path(model_dir) / "checkpoints"
    epoch_checkpoints = []

    for checkpoint_file in checkpoint_dir.glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", checkpoint_file.name)
        if match:
            epoch_checkpoints.append((int(match.group(1)), checkpoint_file))

    if not epoch_checkpoints:
        return None, 0

    latest_epoch, latest_file = max(epoch_checkpoints)
    return latest_file, latest_epoch


@dataclass
class TrainingState:
    """The state needed to continue a training run."""

    next_epoch: int = 0
    best_loss: float = float("inf")


class CheckpointStore:
    """Persist and restore model checkpoints without owning training logic.

    ``checkpoint_epoch_offset`` preserves a caller's existing checkpoint
    convention. The original visual-proprioception notebook stores zero-based
    epoch numbers, while the generic runner stores one-based counts.
    """

    def __init__(
        self,
        model_dir,
        model_filename,
        *,
        keep_checkpoints=2,
        checkpoint_epoch_offset=0,
    ):
        if keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be at least 1")

        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_filename
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_epoch_offset = checkpoint_epoch_offset

    @property
    def best_path(self):
        return self.checkpoint_dir / "best_model.pth"

    def _load(self, path, device):
        return torch.load(path, map_location=device, weights_only=True)

    @staticmethod
    def _model_state(payload):
        return payload.get("model_state_dict", payload)

    @staticmethod
    def _best_loss(payload):
        return payload.get(
            "best_val_loss", payload.get("best_loss", float("inf"))
        )

    def _checkpoint_payload(
        self, epoch, state, model, optimizer, train_loss, validation_loss,
        scheduler=None,
    ):
        stored_epoch = epoch + 1 - self.checkpoint_epoch_offset
        payload = {
            "epoch": stored_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": state.best_loss,
            "best_val_loss": state.best_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "val_loss": validation_loss,
            "test_loss": validation_loss,
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        return payload

    def load_completed_model(self, model, device):
        """Load the exported final model, returning whether one existed."""
        if not self.model_path.is_file():
            return False
        model.load_state_dict(self._load(self.model_path, device))
        print(f"Loading existing model from {self.model_path}")
        return True

    def resume_latest(self, model, optimizer, device, scheduler=None):
        """Load the newest epoch checkpoint and return its continuation state."""
        checkpoint_path, _ = find_latest_checkpoint(self.model_dir)
        if checkpoint_path is None:
            return TrainingState()

        checkpoint = self._load(checkpoint_path, device)
        model.load_state_dict(self._model_state(checkpoint))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        state = TrainingState(
            next_epoch=checkpoint["epoch"] + self.checkpoint_epoch_offset,
            best_loss=self._best_loss(checkpoint),
        )
        print(
            f"Resuming training from {checkpoint_path} at epoch "
            f"{state.next_epoch} with best loss {state.best_loss:.4f}"
        )
        return state

    def restart_from_best(self, model, optimizer, device, scheduler=None):
        """Restore the best checkpoint as an explicit training restart point."""
        if not self.best_path.is_file():
            return None

        checkpoint = self._load(self.best_path, device)
        model.load_state_dict(self._model_state(checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        state = TrainingState(
            next_epoch=checkpoint["epoch"] + self.checkpoint_epoch_offset,
            best_loss=self._best_loss(checkpoint),
        )
        print(
            f"Restarting from best checkpoint {self.best_path} at epoch "
            f"{state.next_epoch} with loss {state.best_loss:.4f}"
        )
        return state

    def save_epoch(
        self, epoch, state, model, optimizer, train_loss, validation_loss,
        scheduler=None,
    ):
        """Save a resumable epoch checkpoint and retain only recent epochs."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stored_epoch = epoch + 1 - self.checkpoint_epoch_offset
        checkpoint_path = self.checkpoint_dir / f"epoch_{stored_epoch:06d}.pth"
        torch.save(
            self._checkpoint_payload(
                epoch, state, model, optimizer, train_loss, validation_loss,
                scheduler,
            ),
            checkpoint_path,
        )
        self.trim_epoch_checkpoints()
        print(f"Checkpoint saved: {checkpoint_path}")

    def save_best(
        self, epoch, state, model, optimizer, train_loss, validation_loss,
        scheduler=None,
    ):
        """Save a full checkpoint that can be restored independently."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._checkpoint_payload(
                epoch, state, model, optimizer, train_loss, validation_loss,
                scheduler,
            ),
            self.best_path,
        )
        print(f"New best model saved with validation loss: {state.best_loss:.4f}")

    def save_emergency(
        self, epoch, batch, state, model, optimizer, partial_loss,
        scheduler=None,
    ):
        """Save a diagnostic checkpoint after a recoverable batch failure."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / (
            f"emergency_epoch_{epoch:06d}_batch_{batch:06d}.pth"
        )
        payload = self._checkpoint_payload(
            epoch, state, model, optimizer, partial_loss, float("nan"), scheduler
        )
        payload["batch"] = batch
        torch.save(payload, path)
        print(f"Emergency checkpoint saved to {path}")

    def restore_best_for_export(self, model, device):
        """Restore the best checkpoint's model state before final export."""
        if not self.best_path.is_file():
            return False
        model.load_state_dict(self._model_state(self._load(self.best_path, device)))
        return True

    def save_completed_model(self, model):
        """Export the final model state in the configured final-model format."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_path)
        print(f"Final model saved to {self.model_path}")

    def trim_epoch_checkpoints(self):
        """Delete old epoch checkpoints without affecting the best checkpoint."""
        checkpoints = []
        for checkpoint_path in self.checkpoint_dir.glob("epoch_*.pth"):
            match = re.fullmatch(r"epoch_(\d+)\.pth", checkpoint_path.name)
            if match:
                checkpoints.append((int(match.group(1)), checkpoint_path))

        checkpoints.sort()
        for _, checkpoint_path in checkpoints[:-self.keep_checkpoints]:
            checkpoint_path.unlink()
            print(f"Deleted old checkpoint: {checkpoint_path}")

