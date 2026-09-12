"""Local convolutional variational autoencoder and training support.

This module intentionally has no dependency on the legacy external Conv-VAE
checkout.  Image geometry and all training parameters come from the BerryPicker
experiment/run.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from demonstration.demonstration import Demonstration
from exp_run_config import Config
from sensorprocessing.sp_helper import get_transform_to_sp
from training_harness import load_or_train


ARCHITECTURE_VERSION = 1


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _image_size(exp):
    size = exp["image_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("image_size must contain [height, width]")
    height = _positive_int(size[0], "image_size height")
    width = _positive_int(size[1], "image_size width")
    return height, width


def _group_count(channels, maximum_groups):
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError(f"Could not select GroupNorm groups for {channels} channels")


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels, group_norm_groups):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(
                _group_count(output_channels, group_norm_groups), output_channels
            ),
            nn.SiLU(inplace=True),
        )


class _UpsampleBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels, group_norm_groups):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(
                _group_count(output_channels, group_norm_groups), output_channels
            ),
            nn.SiLU(inplace=True),
        )


class ConvVAENeo(nn.Module):
    """A resolution-scalable convolutional VAE implemented inside BerryPicker."""

    def __init__(self, exp):
        super().__init__()
        version = exp.get("architecture_version", ARCHITECTURE_VERSION)
        if version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported Conv-VAE-Neo architecture_version: {version}"
            )
        self.image_size = _image_size(exp)
        self.input_channels = _positive_int(
            exp.get("input_channels", 3), "input_channels"
        )
        if self.input_channels != 3:
            raise ValueError("Conv-VAE-Neo currently requires RGB input_channels=3")
        self.latent_size = _positive_int(exp["latent_size"], "latent_size")
        base_channels = _positive_int(
            exp.get("base_channels", 32), "base_channels"
        )
        max_channels = _positive_int(
            exp.get("max_channels", 512), "max_channels"
        )
        if max_channels < base_channels:
            raise ValueError("max_channels must not be smaller than base_channels")
        bottleneck_max_size = _positive_int(
            exp.get("bottleneck_max_size", 8), "bottleneck_max_size"
        )
        group_norm_groups = _positive_int(
            exp.get("group_norm_groups", 8), "group_norm_groups"
        )

        height, width = self.image_size
        encoded_height, encoded_width = height, width
        channel_widths = []
        while max(encoded_height, encoded_width) > bottleneck_max_size:
            width_for_stage = min(
                base_channels * (2 ** len(channel_widths)), max_channels
            )
            channel_widths.append(width_for_stage)
            encoded_height = (encoded_height + 1) // 2
            encoded_width = (encoded_width + 1) // 2
        if not channel_widths:
            channel_widths.append(base_channels)
            encoded_height = (encoded_height + 1) // 2
            encoded_width = (encoded_width + 1) // 2
        if encoded_height < 1 or encoded_width < 1:
            raise ValueError("image_size is too small for the configured encoder")

        self.channel_widths = tuple(channel_widths)
        self.encoded_size = (encoded_height, encoded_width)
        encoder_blocks = []
        previous_channels = self.input_channels
        for output_channels in self.channel_widths:
            encoder_blocks.append(
                _ConvBlock(previous_channels, output_channels, group_norm_groups)
            )
            previous_channels = output_channels
        self.encoder = nn.Sequential(*encoder_blocks)

        self.flatten_size = (
            self.channel_widths[-1] * encoded_height * encoded_width
        )
        self.fc_mu = nn.Linear(self.flatten_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.flatten_size, self.latent_size)
        self.fc_decode = nn.Linear(self.latent_size, self.flatten_size)

        decoder_blocks = []
        previous_channels = self.channel_widths[-1]
        for output_channels in reversed(self.channel_widths[:-1]):
            decoder_blocks.append(
                _UpsampleBlock(previous_channels, output_channels, group_norm_groups)
            )
            previous_channels = output_channels
        self.decoder = nn.Sequential(*decoder_blocks)
        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(previous_channels, self.input_channels, 3, padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_images(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Conv-VAE-Neo input must be a torch.Tensor")
        expected = (self.input_channels, *self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                "Conv-VAE-Neo expected input shape [batch, "
                f"{expected[0]}, {expected[1]}, {expected[2]}], got "
                f"{tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("Conv-VAE-Neo input tensor must be floating point")

    def encode_distribution(self, images):
        """Return the mean and log-variance of ``q(z | images)``."""
        self._validate_images(images)
        features = self.encoder(images).flatten(1)
        if features.size(1) != self.flatten_size:
            raise RuntimeError(
                f"Encoder produced {features.size(1)} features; expected "
                f"{self.flatten_size}"
            )
        return self.fc_mu(features), self.fc_logvar(features)

    def encode(self, images):
        """Return the deterministic latent mean used by sensor processing."""
        mu, _ = self.encode_distribution(images)
        return mu

    @staticmethod
    def reparameterize(mu, logvar):
        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent):
        """Decode a ``[batch, latent_size]`` tensor to RGB images in ``[0, 1]``."""
        if not isinstance(latent, torch.Tensor):
            raise TypeError("Conv-VAE-Neo latent must be a torch.Tensor")
        if latent.ndim != 2 or latent.size(1) != self.latent_size:
            raise ValueError(
                f"Expected latent shape [batch, {self.latent_size}], got "
                f"{tuple(latent.shape)}"
            )
        decoded = self.fc_decode(latent).view(
            latent.size(0), self.channel_widths[-1], *self.encoded_size
        )
        decoded = self.output_layer(self.decoder(decoded))
        if tuple(decoded.shape[-2:]) != self.image_size:
            decoded = F.interpolate(
                decoded,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(decoded)

    def forward(self, images):
        mu, logvar = self.encode_distribution(images)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(latent), mu, logvar

    def sample(self, count):
        count = _positive_int(count, "sample count")
        parameter = next(self.parameters())
        latent = torch.randn(
            count,
            self.latent_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.decode(latent)


class ConvVAENeoLoss(nn.Module):
    """Configured ELBO objective with separately inspectable components."""

    def __init__(self, exp):
        super().__init__()
        self.reconstruction_loss = exp.get("reconstruction_loss", "mse")
        if self.reconstruction_loss not in {"mse", "bce"}:
            raise ValueError(
                "Conv-VAE-Neo reconstruction_loss must be 'mse' or 'bce'"
            )
        self.kl_weight = _finite_nonnegative(
            exp.get("kl_weight", 0.001), "kl_weight"
        )

    def components(self, output, target):
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError("Conv-VAE-Neo output must be (reconstruction, mu, logvar)")
        reconstruction, mu, logvar = output
        if reconstruction.shape != target.shape:
            raise ValueError("Reconstruction and target must have identical shapes")
        if mu.shape != logvar.shape or mu.ndim != 2:
            raise ValueError("mu and logvar must be matching rank-two tensors")
        if self.reconstruction_loss == "mse":
            reconstruction_value = F.mse_loss(reconstruction, target)
        else:
            reconstruction_value = F.binary_cross_entropy(reconstruction, target)
        kl_value = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )
        total = reconstruction_value + self.kl_weight * kl_value
        if not torch.isfinite(total):
            raise FloatingPointError("Conv-VAE-Neo loss is not finite")
        return {
            "loss": total,
            "reconstruction": reconstruction_value,
            "kl": kl_value,
        }

    def forward(self, output, target):
        return self.components(output, target)["loss"]


class DemonstrationImageDataset(Dataset):
    """Lazy single-camera frame access for unsupervised VAE training."""

    def __init__(
        self,
        exp,
        dataset_name,
        *,
        demonstration_factory=Demonstration,
        experiment_loader=None,
    ):
        entries = exp[dataset_name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"exp['{dataset_name}'] must be a nonempty list")
        self.transform = get_transform_to_sp(exp)
        self.image_size = _image_size(exp)
        self.frame_stride = _positive_int(
            exp.get("frame_stride", 1), "frame_stride"
        )
        self.sources = []
        self.samples = []
        self._video_captures = {}
        load_experiment = experiment_loader or Config().get_experiment

        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError(
                    f"exp['{dataset_name}'] entries must be "
                    "[demonstration_run, demonstration_name, camera]"
                )
            run, demonstration_name, camera = entry
            demo_exp = load_experiment("demonstration", run)
            demonstration = demonstration_factory(demo_exp, demonstration_name)
            cameras = demonstration.metadata.get("cameras", [])
            if cameras and camera not in cameras:
                raise ValueError(
                    f"Demonstration {demonstration_name} does not contain camera "
                    f"{camera}"
                )
            maxsteps = demonstration.metadata.get("maxsteps", 0)
            if type(maxsteps) is not int or maxsteps <= 0:
                raise ValueError(
                    f"Demonstration {demonstration_name} contains no image frames"
                )
            if not demonstration.metadata.get("stored_as_images", False) \
                    and not demonstration.metadata.get("stored_as_video", False):
                raise ValueError(
                    f"Demonstration {demonstration_name} has no image storage"
                )
            source_index = len(self.sources)
            self.sources.append((demonstration, camera))
            self.samples.extend(
                (source_index, frame)
                for frame in range(0, maxsteps, self.frame_stride)
            )

    def __len__(self):
        return len(self.samples)

    def _read_video_frame(self, source_index, demonstration, camera, frame):
        capture = self._video_captures.get(source_index)
        if capture is None:
            video_path = demonstration.get_video_path(camera)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open demonstration video: {video_path}")
            self._video_captures[source_index] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            raise ValueError(
                f"Could not read frame {frame} for camera {camera} from "
                f"demonstration {demonstration.demo}"
            )
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def __getitem__(self, index):
        source_index, frame = self.samples[index]
        demonstration, camera = self.sources[source_index]
        if demonstration.metadata.get("stored_as_images", False):
            image_path = demonstration.get_image_path(frame, camera)
            if not image_path.is_file():
                raise FileNotFoundError(f"Demonstration image does not exist: {image_path}")
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        else:
            image = self._read_video_frame(
                source_index, demonstration, camera, frame
            )
            tensor = self.transform(image)
        expected = (3, *self.image_size)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Preprocessed demonstration image has shape "
                f"{getattr(tensor, 'shape', None)}; expected {expected}"
            )
        return tensor

    def close(self):
        for capture in getattr(self, "_video_captures", {}).values():
            capture.release()
        self._video_captures = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_captures"] = {}
        return state

    def __del__(self):
        self.close()


def make_dataloaders(exp):
    """Create lazy training and validation loaders from demonstration entries."""
    training_entries = {tuple(entry) for entry in exp["training_data"]}
    validation_entries = {tuple(entry) for entry in exp["validation_data"]}
    overlap = training_entries & validation_entries
    if overlap:
        raise ValueError(
            f"training_data and validation_data overlap: {sorted(overlap)}"
        )
    training_dataset = DemonstrationImageDataset(exp, "training_data")
    validation_dataset = DemonstrationImageDataset(exp, "validation_data")
    batch_size = _positive_int(exp["batch_size"], "batch_size")
    num_workers = exp.get("num_workers", 0)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    pin_memory = exp.get("pin_memory", False)
    if type(pin_memory) is not bool:
        raise ValueError("pin_memory must be boolean")
    seed = exp.get("random_seed", 0)
    if type(seed) is not int:
        raise ValueError("random_seed must be an integer")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return (
        DataLoader(
            training_dataset, shuffle=True, generator=generator, **common
        ),
        DataLoader(validation_dataset, shuffle=False, **common),
    )


def make_vae_epoch_steps(
    loss_function, training_loader, validation_loader, *, grad_clip_norm=None
):
    """Return model-specific callbacks accepted by the shared training harness."""
    if grad_clip_norm is not None:
        grad_clip_norm = _finite_nonnegative(grad_clip_norm, "grad_clip_norm")
        if grad_clip_norm == 0:
            raise ValueError("grad_clip_norm must be positive when configured")

    def training_step(model, optimizer):
        model.train()
        total_loss = 0.0
        batches = 0
        device = Config().runtime["device"]
        for images in training_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), images)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if batches == 0:
            raise ValueError("Conv-VAE-Neo training loader produced no batches")
        return total_loss / batches

    def validation_step(model):
        model.eval()
        total_loss = 0.0
        batches = 0
        device = Config().runtime["device"]
        with torch.no_grad():
            for images in validation_loader:
                images = images.to(device, non_blocking=True)
                total_loss += loss_function(model(images), images).item()
                batches += 1
        if batches == 0:
            raise ValueError("Conv-VAE-Neo validation loader produced no batches")
        return total_loss / batches

    return training_step, validation_step


def seed_everything(seed):
    if type(seed) is not int:
        raise ValueError("random_seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(exp, *, epochs=None):
    """Load, resume, or train a configured Conv-VAE-Neo model."""
    seed_everything(exp.get("random_seed", 0))
    training_loader, validation_loader = make_dataloaders(exp)
    model = ConvVAENeo(exp).to(Config().runtime["device"])
    loss_function = ConvVAENeoLoss(exp)
    optimizer_name = exp.get("optimizer", "adam")
    if not isinstance(optimizer_name, str):
        raise ValueError("Conv-VAE-Neo optimizer must be a string")
    optimizer_name = optimizer_name.lower()
    learning_rate = exp["learning_rate"]
    if (
        not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number")
    weight_decay = _finite_nonnegative(
        exp.get("weight_decay", 0.0), "weight_decay"
    )
    optimizer_arguments = {
        "lr": float(learning_rate),
        "weight_decay": weight_decay,
    }
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_arguments)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_arguments)
    else:
        raise ValueError(f"Unsupported Conv-VAE-Neo optimizer: {optimizer_name}")
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
