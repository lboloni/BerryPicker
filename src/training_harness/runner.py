"""
runner.py

Training-loop orchestration for the shared training harness.
"""

import re

import torch

from exp_run_config import Config
from .checkpoints import (
    CheckpointStore,
    TrainingState,
    _model_file,
    find_latest_checkpoint,
    model_available,
)

Config.PROJECTNAME = "BerryPicker"


def train_with_checkpoints(
    exp,
    model,
    optimizer,
    train_epoch,
    evaluate_epoch,
    *,
    checkpoint_store=None,
    scheduler=None,
    progress=None,
    restart_from_best=False,
    on_completed_model_loaded=None,
):
    """Run model-specific epoch callbacks under a shared checkpoint harness.

    ``train_epoch`` receives ``(model, optimizer, epoch, state,
    on_batch_error)`` and returns the mean training loss. ``evaluate_epoch``
    receives ``model`` and returns the mean validation loss.
    """
    device = Config().runtime["device"]
    model = model.to(device)
    store = checkpoint_store or CheckpointStore(
        exp["data_dir"], exp["proprioception_mlp_model_file"]
    )

    if (
        not restart_from_best
        and exp.get("reload_existing_model", True)
        and store.load_completed_model(model, device)
    ):
        if on_completed_model_loaded is not None:
            on_completed_model_loaded(model)
        return model

    if restart_from_best:
        state = store.restart_from_best(model, optimizer, device, scheduler)
        if state is None:
            print("No best checkpoint found; starting a new training run")
            state = TrainingState()
    else:
        state = store.resume_latest(model, optimizer, device, scheduler)
        if state.next_epoch == 0:
            print(f"Starting new training for {exp['epochs']} epochs")

    epochs = range(state.next_epoch, exp["epochs"])
    if progress is not None:
        epochs = progress(epochs)

    for epoch in epochs:
        def on_batch_error(batch, partial_loss, error):
            print(f"Error in batch {batch}: {error}")
            store.save_emergency(
                epoch, batch, state, model, optimizer, partial_loss, scheduler
            )

        train_loss = train_epoch(model, optimizer, epoch, state, on_batch_error)
        validation_loss = evaluate_epoch(model)
        improved = validation_loss < state.best_loss
        if improved:
            state.best_loss = validation_loss

        if scheduler is not None:
            scheduler.step(validation_loss)

        store.save_epoch(
            epoch, state, model, optimizer, train_loss, validation_loss, scheduler
        )
        if improved:
            store.save_best(
                epoch, state, model, optimizer, train_loss, validation_loss,
                scheduler,
            )

        print(
            f"Epoch [{epoch + 1}/{exp['epochs']}], Train Loss: "
            f"{train_loss:.4f}, Validation Loss: {validation_loss:.4f}"
        )

    if store.restore_best_for_export(model, device):
        print(f"Training complete. Best validation loss: {state.best_loss:.4f}")
    else:
        print("Training complete without a best checkpoint")
    store.save_completed_model(model)
    return model


def train_model(
    exp,
    model,
    optimizer,
    model_training_step,
    model_eval_step,
    epochs=None,
    scheduler=None,
    log_interval=1,
    start_epoch=0,
    best_val_loss=float("inf"),
    keep_checkpoints=2,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    """Train and save a model using model-specific epoch callbacks.

    The final model path and the default epoch count are read from ``exp``.
    Passing ``epochs`` overrides ``exp["epochs"]``, which is useful for
    parameterized notebook runs.

    ``model_training_step(model, optimizer)`` and ``model_eval_step(model)``
    must each run one complete epoch and return its average loss.

    The best model (by validation loss) is saved to the configured final model
    file and to ``checkpoints/best_model.pth``. Epoch checkpoints contain the
    state needed to resume training, and only the most recent
    ``keep_checkpoints`` are kept.

    When ``early_stopping_patience`` is set, training stops once the
    validation loss has not improved by more than ``early_stopping_min_delta``
    for that many consecutive epochs (the counter is stored in the epoch
    checkpoints so a resumed run continues where it left off).
    """
    if keep_checkpoints < 1:
        raise ValueError("keep_checkpoints must be at least 1")
    epochs_without_improvement = 0

    modelfile = _model_file(exp)
    if epochs is None:
        epochs = exp.get("epochs", 20)

    checkpoint_dir = modelfile.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"

    def checkpoint_epoch(path):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        return int(match.group(1)) if match else -1

    saved_checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pth"), key=checkpoint_epoch
    )
    while len(saved_checkpoints) > keep_checkpoints:
        oldest_checkpoint = saved_checkpoints.pop(0)
        oldest_checkpoint.unlink()
        print(f"Deleted old checkpoint: {oldest_checkpoint}")

    latest_checkpoint, _ = find_latest_checkpoint(modelfile.parent)
    if latest_checkpoint is not None and start_epoch > 0:
        resumed = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
        epochs_without_improvement = resumed.get("epochs_without_improvement", 0)

    for epoch in range(start_epoch, epochs):
        avg_train_loss = model_training_step(model, optimizer)
        avg_val_loss = model_eval_step(model)

        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                "  New best model saved with validation loss: "
                f"{best_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        checkpoint_file = checkpoint_dir / f"epoch_{epoch + 1:06d}.pth"
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")
        saved_checkpoints = [
            path for path in saved_checkpoints if path != checkpoint_file
        ]
        saved_checkpoints.append(checkpoint_file)
        saved_checkpoints.sort(key=checkpoint_epoch)

        while len(saved_checkpoints) > keep_checkpoints:
            oldest_checkpoint = saved_checkpoints.pop(0)
            oldest_checkpoint.unlink()
            print(f"Deleted old checkpoint: {oldest_checkpoint}")

        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    if best_model_path.exists():
        best_model_state = torch.load(
            best_model_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, modelfile)
        print(f"Best model saved to {modelfile}")
    else:
        torch.save(model.state_dict(), modelfile)
        print(f"Final model saved to {modelfile}")

    return model


def load_or_train(
    exp,
    model,
    optimizer,
    model_training_step,
    model_eval_step,
    epochs=None,
    scheduler=None,
    log_interval=1,
    keep_checkpoints=2,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    """Load a completed model, resume a checkpoint, or train from scratch.

    ``early_stopping_patience`` / ``early_stopping_min_delta`` are forwarded to
    :func:`train_model`; when they are ``None`` the values are read from
    ``exp["early_stopping_patience"]`` / ``exp["early_stopping_min_delta"]``
    (both optional).
    """
    if early_stopping_patience is None:
        early_stopping_patience = exp.get("early_stopping_patience")
    if not early_stopping_min_delta:
        early_stopping_min_delta = exp.get("early_stopping_min_delta", 0.0)
    device = Config().runtime["device"]
    model = model.to(device)
    modelfile = _model_file(exp)
    num_epochs = epochs if epochs is not None else exp.get("epochs", 20)

    if model_available(exp) and exp.get("reload_existing_model", True):
        print(f"Loading existing model from {modelfile}")
        model_state = torch.load(
            modelfile, map_location=device, weights_only=True
        )
        model.load_state_dict(model_state)
        return model

    latest_checkpoint, start_epoch = find_latest_checkpoint(modelfile.parent)
    best_val_loss = float("inf")

    if latest_checkpoint is not None:
        print(
            f"Resuming training from checkpoint: {latest_checkpoint} "
            f"(Epoch {start_epoch})"
        )
        checkpoint = torch.load(
            latest_checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Previous best validation loss: {best_val_loss:.4f}")
        print(
            "Continuing training for "
            f"{max(0, num_epochs - start_epoch)} more epochs"
        )
    else:
        print(f"Training new model for {num_epochs} epochs")

    return train_model(
        exp,
        model,
        optimizer,
        model_training_step,
        model_eval_step,
        epochs=num_epochs,
        scheduler=scheduler,
        log_interval=log_interval,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        keep_checkpoints=keep_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
