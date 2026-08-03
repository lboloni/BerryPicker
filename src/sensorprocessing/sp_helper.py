"""
sp_helper.py

Helper functions for sensor processing. Loading image files or video captures into formats that are suitable to be fed into the video models.
"""
import re
import sys
from pathlib import Path
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from PIL import Image
import torch
from torchvision import transforms


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


def train_model(exp, model, optimizer, model_training_step,
                model_eval_step, epochs=None, scheduler=None, log_interval=1,
                start_epoch=0, best_val_loss=float("inf"),
                keep_checkpoints=2):
    """Train and save a model using model-specific epoch callbacks.

    The final model path and the default epoch count are read from ``exp``.
    Passing ``epochs`` overrides ``exp["epochs"]``, which is useful for
    parameterized notebook runs.

    ``model_training_step(model, optimizer)`` and ``model_eval_step(model)``
    must each run one complete epoch and return its average loss.

    The best model (by validation loss) is saved to ``modelfile`` and to
    ``checkpoints/best_model.pth``. Epoch checkpoints contain the state needed
    to resume training, and only the most recent ``keep_checkpoints`` are kept.
    """
    if keep_checkpoints < 1:
        raise ValueError("keep_checkpoints must be at least 1")

    modelfile = (
        Path(exp["data_dir"]) / exp["proprioception_mlp_model_file"]
    )
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

    for epoch in range(start_epoch, epochs):
        avg_train_loss = model_training_step(model, optimizer)
        avg_val_loss = model_eval_step(model)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                "  New best model saved with validation loss: "
                f"{best_val_loss:.4f}"
            )

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


def load_or_train(exp, model, optimizer, model_training_step,
                  model_eval_step, epochs=None, scheduler=None,
                  log_interval=1, keep_checkpoints=2):
    """Load a completed model, resume a checkpoint, or train from scratch.

    A completed model is returned immediately when it exists and
    ``exp["reload_existing_model"]`` is true (the default). Otherwise, the
    latest epoch checkpoint is restored before training continues. If neither
    is available, training starts with the supplied model and optimizer.
    """
    device = Config().runtime["device"]
    model = model.to(device)
    modelfile = (
        Path(exp["data_dir"]) / exp["proprioception_mlp_model_file"]
    )
    num_epochs = epochs if epochs is not None else exp.get("epochs", 20)

    if modelfile.exists() and exp.get("reload_existing_model", True):
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
            f"Continuing training for "
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
    )


def get_transform_to_sp(exp):
    """Creates a transform object that transforms a figure into the right size tensor that is suitable for the sensor processing model."""
    image_size = exp["image_size"][0]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return transform

def load_image_to_batch(image, transform):
    """Takes a PIL image and transforms it into a tensor representing a single element batch, moved to the appropriate device.
    This format should be approprate for feeding into the sensor processing model."""
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)
    image_tensor_for_pic = image_tensor.permute(1, 2, 0)
    image_batch = image_tensor.unsqueeze(0)
    image_batch_device = image_batch.to(Config().runtime["device"])
    return image_batch_device, image_tensor_for_pic

def load_picturefile_to_tensor(picture_file, transform):
    """Loads an image from a file using PIL and returns it as a single image batch and the image in a displayable format.
    """
    image = Image.open(picture_file)
    return load_image_to_batch(image, transform)

def load_capture_to_tensor(image_from_camera, transform):
    """Gets an image as returned by the camera controller and transforms it into a single element batch, moved to the appropriate device. Returns the batch and the image in a displayable format.
    """
    image = Image.fromarray(image_from_camera)
    return load_image_to_batch(image, transform)
