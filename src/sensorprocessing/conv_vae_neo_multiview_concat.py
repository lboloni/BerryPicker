"""A joint Conv-VAE-Neo over horizontally concatenated camera views."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from exp_run_config import Config
from sensorprocessing.conv_vae_neo import (
    ConvVAENeo,
    ConvVAENeoLoss,
    seed_everything,
)
from sensorprocessing.multiview_data import (
    configured_cameras,
    make_multiview_dataloaders,
)
from training_harness import load_or_train


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exp_values(exp):
    values = getattr(exp, "values", exp)
    if callable(values):
        values = exp
    if not hasattr(values, "copy"):
        raise TypeError("exp must provide copyable configuration values")
    return values.copy()


class ConvVAENeoMultiViewConcatModel(nn.Module):
    """Encode a fixed ordered camera set as one full-width RGB observation."""

    def __init__(self, exp):
        super().__init__()
        self.num_views = _positive_int(exp["num_views"], "num_views")
        self.cameras = configured_cameras(exp)
        stack_mode = exp.get("stack_mode", "width")
        if stack_mode != "width":
            raise ValueError(
                "Conv-VAE-Neo MultiView Concat requires stack_mode='width'"
            )
        size = exp["image_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("image_size must contain [height, width]")
        self.view_image_size = (
            _positive_int(size[0], "image_size height"),
            _positive_int(size[1], "image_size width"),
        )
        self.composite_image_size = (
            self.view_image_size[0],
            self.view_image_size[1] * self.num_views,
        )
        self.latent_size = _positive_int(exp["latent_size"], "latent_size")

        vae_exp = _exp_values(exp)
        vae_exp["image_size"] = list(self.composite_image_size)
        self.vae = ConvVAENeo(vae_exp)

    def compose_views(self, views):
        if not isinstance(views, (list, tuple)):
            raise TypeError("views must be an ordered list or tuple of tensors")
        if len(views) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views)}")
        expected = (3, *self.view_image_size)
        batch_size = None
        for index, view in enumerate(views):
            if not isinstance(view, torch.Tensor):
                raise TypeError(f"View {index} must be a torch.Tensor")
            if view.ndim != 4 or tuple(view.shape[1:]) != expected:
                raise ValueError(
                    f"View {index} must have shape [batch, {expected[0]}, "
                    f"{expected[1]}, {expected[2]}], got {tuple(view.shape)}"
                )
            if not view.is_floating_point():
                raise TypeError(f"View {index} must be floating point")
            if batch_size is None:
                batch_size = view.size(0)
            elif view.size(0) != batch_size:
                raise ValueError(
                    f"View {index} has batch size {view.size(0)}; expected "
                    f"{batch_size}"
                )
        composite = torch.cat(list(views), dim=3)
        expected_composite = (batch_size, 3, *self.composite_image_size)
        if tuple(composite.shape) != expected_composite:
            raise RuntimeError(
                f"Composite has shape {tuple(composite.shape)}; expected "
                f"{expected_composite}"
            )
        return composite

    def split_composite(self, composite):
        expected = (3, *self.composite_image_size)
        if not isinstance(composite, torch.Tensor):
            raise TypeError("composite must be a torch.Tensor")
        if composite.ndim != 4 or tuple(composite.shape[1:]) != expected:
            raise ValueError(
                f"Composite must have shape [batch, {expected[0]}, "
                f"{expected[1]}, {expected[2]}], got {tuple(composite.shape)}"
            )
        views = list(torch.split(composite, self.view_image_size[1], dim=3))
        if len(views) != self.num_views:
            raise RuntimeError(
                f"Composite split produced {len(views)} views; expected "
                f"{self.num_views}"
            )
        return views

    def encode_views(self, views):
        return self.vae.encode(self.compose_views(views))

    def encode(self, views):
        return self.encode_views(views)

    def decode(self, latent):
        return self.vae.decode(latent)

    def sample(self, count):
        return self.vae.sample(count)

    def forward(self, views):
        return self.vae(self.compose_views(views))


def make_dataloaders(exp):
    """Create lazy synchronized loaders for joint-VAE training."""
    return make_multiview_dataloaders(exp)


def make_vae_epoch_steps(
    loss_function, training_loader, validation_loader, *, grad_clip_norm=None
):
    """Create Concat VAE callbacks for the shared training harness."""
    if grad_clip_norm is not None:
        if (
            not isinstance(grad_clip_norm, (int, float))
            or not math.isfinite(grad_clip_norm)
            or grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be a positive finite number")
        grad_clip_norm = float(grad_clip_norm)

    def training_step(model, optimizer):
        model.train()
        total_loss = 0.0
        batches = 0
        device = Config().runtime["device"]
        for views in training_loader:
            views = [view.to(device, non_blocking=True) for view in views]
            target = model.compose_views(views)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(views), target)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if batches == 0:
            raise ValueError("MultiView Concat training loader produced no batches")
        return total_loss / batches

    def validation_step(model):
        model.eval()
        total_loss = 0.0
        batches = 0
        device = Config().runtime["device"]
        with torch.no_grad():
            for views in validation_loader:
                views = [view.to(device, non_blocking=True) for view in views]
                target = model.compose_views(views)
                total_loss += loss_function(model(views), target).item()
                batches += 1
        if batches == 0:
            raise ValueError("MultiView Concat validation loader produced no batches")
        return total_loss / batches

    return training_step, validation_step


def train(exp, *, epochs=None):
    """Load, resume, or train a joint MultiView Concat Conv-VAE-Neo."""
    seed_everything(exp.get("random_seed", 0))
    training_loader, validation_loader = make_dataloaders(exp)
    model = ConvVAENeoMultiViewConcatModel(exp).to(Config().runtime["device"])
    loss_function = ConvVAENeoLoss(exp)
    optimizer_name = exp.get("optimizer", "adam")
    if not isinstance(optimizer_name, str):
        raise ValueError("optimizer must be a string")
    learning_rate = exp["learning_rate"]
    if (
        not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number")
    weight_decay = exp.get("weight_decay", 0.0)
    if (
        not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be a finite nonnegative number")
    arguments = {"lr": float(learning_rate), "weight_decay": float(weight_decay)}
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **arguments)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **arguments)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    training_step, validation_step = make_vae_epoch_steps(
        loss_function,
        training_loader,
        validation_loader,
        grad_clip_norm=exp.get("grad_clip_norm"),
    )
    return load_or_train(
        exp,
        model,
        optimizer,
        training_step,
        validation_step,
        epochs=epochs,
        keep_checkpoints=exp.get("keep_checkpoints", 2),
    )
