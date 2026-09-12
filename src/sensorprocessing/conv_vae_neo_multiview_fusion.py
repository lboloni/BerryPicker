"""Supervised multiview fusion initialized from a trained Conv-VAE-Neo."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from exp_run_config import Config
from sensorprocessing.conv_vae_neo import ConvVAENeo, seed_everything
from sensorprocessing.multiview_backbones import (
    ViewBackbones,
    describe_multiview_model,
)
from sensorprocessing.multiview_data import (
    configured_cameras,
    make_multiview_dataloaders,
)
from sensorprocessing.multiview_fusion import fusion_from_exp
from training_harness import (
    find_latest_checkpoint,
    load_or_train,
    make_epoch_steps,
    model_available,
)
from training_harness.checkpoints import model_file


_NEO_DEFAULTS = {
    "architecture_version": 1,
    "input_channels": 3,
    "base_channels": 32,
    "max_channels": 512,
    "bottleneck_max_size": 8,
    "group_norm_groups": 8,
}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unwrap_model_state(payload):
    if not isinstance(payload, dict):
        raise TypeError("A Conv-VAE-Neo checkpoint must contain a state dictionary")
    return payload.get("model_state_dict", payload)


class ConvVAENeoMeanEncoder(nn.Module):
    """The deterministic encoder and mean projection extracted from Neo."""

    def __init__(self, exp):
        super().__init__()
        vae = ConvVAENeo(exp)
        self.image_size = vae.image_size
        self.input_channels = vae.input_channels
        self.flatten_size = vae.flatten_size
        self.latent_size = vae.latent_size
        self.encoder = vae.encoder
        self.fc_mu = vae.fc_mu

    def forward(self, images):
        expected = (self.input_channels, *self.image_size)
        if not isinstance(images, torch.Tensor):
            raise TypeError("Neo view input must be a torch.Tensor")
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"Neo view encoder expected [batch, {expected[0]}, "
                f"{expected[1]}, {expected[2]}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("Neo view input must be floating point")
        features = self.encoder(images).flatten(1)
        if features.size(1) != self.flatten_size:
            raise RuntimeError(
                f"Neo view encoder produced {features.size(1)} features; "
                f"expected {self.flatten_size}"
            )
        return self.fc_mu(features)

    def load_vae_state_dict(self, vae_state):
        """Strictly load the encoder and mean projection from a full Neo VAE."""
        expected = set(self.state_dict())
        selected = {
            key: value for key, value in vae_state.items() if key in expected
        }
        missing = sorted(expected - set(selected))
        if missing:
            raise KeyError(
                "Source Conv-VAE-Neo checkpoint is missing encoder weights: "
                f"{missing}"
            )
        self.load_state_dict(selected, strict=True)


class ConvVAENeoMultiViewFusionModel(nn.Module):
    """Neo mean encoders, deterministic fusion, and proprioception head."""

    def __init__(self, exp):
        super().__init__()
        self.num_views = _positive_int(exp["num_views"], "num_views")
        self.latent_size = _positive_int(exp["latent_size"], "latent_size")
        self.output_size = _positive_int(exp["output_size"], "output_size")
        self.cameras = configured_cameras(exp)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.backbones = ViewBackbones(
            lambda: ConvVAENeoMeanEncoder(exp),
            self.num_views,
            shared=exp.get("shared_backbone", True),
            freeze=exp.get("freeze_vae_encoder", True),
            batched=exp.get("batched_backbone", True),
        )
        self.fusion = fusion_from_exp(
            exp, self.latent_size, self.num_views, self.latent_size
        )
        step_1 = _positive_int(exp.get("proprio_step_1", 64), "proprio_step_1")
        step_2 = _positive_int(exp.get("proprio_step_2", 32), "proprio_step_2")
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, step_1),
            nn.ReLU(),
            nn.Linear(step_1, step_2),
            nn.ReLU(),
            nn.Linear(step_2, self.output_size),
        )
        describe_multiview_model(self, exp, type(self).__name__)

    def _view_encoders(self):
        if self.backbones.shared:
            return [self.backbones.backbone]
        return list(self.backbones.backbones)

    def load_vae_state_dict(self, state_dict):
        for encoder in self._view_encoders():
            encoder.load_vae_state_dict(state_dict)

    def extract_features(self, views):
        return self.backbones(views)

    def encode_views(self, views):
        if not isinstance(views, (list, tuple)):
            raise TypeError("views must be an ordered list or tuple of tensors")
        views = list(views)
        if len(views) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views)}")
        return self.fusion(self.extract_features(views))

    def encode(self, views):
        return self.encode_views(views)

    def forward(self, views):
        return self.proprioceptor(self.encode_views(views))


def _validate_source_architecture(exp, source_exp):
    fields = ("image_size", "latent_size", *_NEO_DEFAULTS)
    mismatches = []
    for field in fields:
        default = _NEO_DEFAULTS.get(field)
        target = exp.get(field, default)
        source = source_exp.get(field, default)
        if target != source:
            mismatches.append(f"{field}: fusion={target!r}, source={source!r}")
    if mismatches:
        raise ValueError(
            "Source Conv-VAE-Neo architecture does not match Fusion model: "
            + "; ".join(mismatches)
        )


def initialize_from_source_vae(exp, model):
    """Initialize every view encoder from the configured completed Neo VAE."""
    source_experiment = exp["source_vae_experiment"]
    source_run = exp["source_vae_run"]
    source_exp = Config().get_experiment(source_experiment, source_run)
    _validate_source_architecture(exp, source_exp)
    source_path = model_file(source_exp)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source Conv-VAE-Neo model does not exist: {source_path}"
        )
    payload = torch.load(
        source_path,
        map_location=Config().runtime["device"],
        weights_only=True,
    )
    model.load_vae_state_dict(_unwrap_model_state(payload))
    print(f"Initialized Fusion view encoder from {source_path}")
    return source_path


def _criterion(exp):
    loss_name = exp.get("loss", "MSELoss")
    if loss_name in {"MSELoss", "MSE"}:
        return nn.MSELoss()
    if loss_name in {"L1Loss", "L1"}:
        return nn.L1Loss()
    raise ValueError(f"Unsupported Fusion loss: {loss_name}")


def train(exp, *, epochs=None):
    """Load, resume, or train a supervised MultiView Fusion model."""
    seed_everything(exp.get("random_seed", 0))
    robot_exp = Config().get_experiment(exp["robot_exp"], exp["robot_run"])
    training_loader, validation_loader = make_multiview_dataloaders(
        exp, robot_exp=robot_exp
    )
    model = ConvVAENeoMultiViewFusionModel(exp).to(Config().runtime["device"])

    final_will_load = model_available(exp) and exp.get(
        "reload_existing_model", True
    )
    latest_checkpoint, _ = find_latest_checkpoint(Path(exp["data_dir"]))
    if not final_will_load and latest_checkpoint is None:
        initialize_from_source_vae(exp, model)

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Fusion model has no trainable parameters")
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
    optimizer_name = exp.get("optimizer", "adam")
    if not isinstance(optimizer_name, str):
        raise ValueError("optimizer must be a string")
    optimizer_name = optimizer_name.lower()
    arguments = {"lr": float(learning_rate), "weight_decay": float(weight_decay)}
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, **arguments)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, **arguments)
    else:
        raise ValueError(f"Unsupported Fusion optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=exp.get("lr_factor", 0.5),
        patience=exp.get("lr_patience", 10),
    )
    training_step, validation_step = make_epoch_steps(
        _criterion(exp),
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
        scheduler=scheduler,
        keep_checkpoints=exp.get("keep_checkpoints", 2),
    )
