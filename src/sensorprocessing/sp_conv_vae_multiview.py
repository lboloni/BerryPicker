"""
sp_conv_vae_multiview.py

Multi-view sensor processing with a small convolutional encoder trained from
scratch on proprioception (no pretrained backbone).

NOTE ON THE NAME: this is the multi-view counterpart of the Conv-VAE sensor
processing and it reuses the Conv-VAE *encoder architecture* (four stride-2
conv blocks followed by a linear layer to the latent size), but it is trained
supervised on the robot position like the propriotuned CNN and ViT models.
There is no decoder and no KL term, so it is a "conv encoder", not a VAE.
Keep this in mind when comparing it against the single-view Conv-VAE, which
is trained unsupervised.

Architecture: shared conv encoder per view -> per-view latent of size
``latent_size`` -> :class:`sensorprocessing.multiview_fusion.MultiViewFusion`
-> fused latent (``latent_size``) -> proprioceptor (training only).

Relevant exp/run keys (see ``experiment_configs/sensorprocessing_conv_vae_multiview``):

- ``image_size`` (default [64, 64]); any size divisible by 16 works, the
  flattened width is derived from it
- ``latent_size``, ``num_views``, ``cameras``
- ``shared_backbone`` (default true), ``batched_backbone`` (default true)
- ``encoder_channels`` (default [32, 64, 128, 256])
- ``fusion_type`` and the other fusion keys (see multiview_fusion)
- ``proprio_step_1`` / ``proprio_step_2`` (defaults 64 / 32)
"""

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn

from .sensor_processing import MultiViewEncoderSensorProcessing
from .multiview_backbones import ViewBackbones, describe_multiview_model
from .multiview_fusion import fusion_from_exp


def _image_size(exp):
    size = exp.get("image_size", [64, 64])
    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


class ConvEncoder(nn.Module):
    """Conv-VAE style encoder: stride-2 conv blocks, then a linear layer to ``latent_size``.

    Returns the latent mean only (no log-variance, no sampling).
    """

    def __init__(self, latent_size, image_size, channels=(32, 64, 128, 256), in_channels=3):
        super().__init__()
        blocks = []
        previous = in_channels
        for width in channels:
            blocks.extend(
                [
                    nn.Conv2d(previous, width, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*blocks)

        with torch.no_grad():
            self.flatten_size = self.encoder(torch.zeros(1, in_channels, *image_size)).numel()
        self.fc_mu = nn.Linear(self.flatten_size, latent_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc_mu(features.flatten(1))


class MultiViewConvVAEModel(nn.Module):
    """Shared conv encoder per view + fusion head + proprioceptor."""

    def __init__(self, exp):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp.get("latent_size", 128)
        self.output_size = exp.get("output_size", 6)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None
        self.image_size = _image_size(exp)
        channels = tuple(exp.get("encoder_channels", (32, 64, 128, 256)))

        self.backbones = ViewBackbones(
            lambda: ConvEncoder(self.latent_size, self.image_size, channels),
            self.num_views,
            shared=exp.get("shared_backbone", True),
            freeze=False,  # trained from scratch, never frozen
            batched=exp.get("batched_backbone", True),
        )
        # Each view is encoded to latent_size, so the fusion feature width is latent_size.
        self.d_view = self.latent_size
        self.fusion = fusion_from_exp(exp, self.d_view, self.num_views, self.latent_size)

        step_1 = exp.get("proprio_step_1", 64)
        step_2 = exp.get("proprio_step_2", 32)
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, step_1),
            nn.ReLU(),
            nn.Linear(step_1, step_2),
            nn.ReLU(),
            nn.Linear(step_2, self.output_size),
        )

        self.to(Config().runtime["device"])
        describe_multiview_model(self, exp, "MultiViewConvVAEModel")

    def extract_features(self, views_list):
        """Per-view encoder latents of shape ``[batch, latent_size]``."""
        return self.backbones(views_list)

    def encode_views(self, views_list):
        views_list = list(views_list)
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")
        return self.fusion(self.extract_features(views_list))

    def encode(self, views_list):
        return self.encode_views(views_list)

    def forward(self, views_list):
        return self.proprioceptor(self.encode_views(views_list))


class MultiViewConvVAESensorProcessing(MultiViewEncoderSensorProcessing):
    """Runtime wrapper: ordered camera views -> fused latent (no regression)."""

    encoder_method = "encode_views"

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = exp.get("num_views", 2)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None

        print("Initializing Multi-View Conv encoder Sensor Processing:")
        print(f"  Number of views: {self.num_views}")
        print(f"  Fusion type: {self.fusion_type}")
        print(f"  Latent dimension: {exp['latent_size']}")
        print(f"  Image size: {exp.get('image_size', [64, 64])}")
        print(f"  Camera order: {self.cameras}")

        self.enc = MultiViewConvVAEModel(exp)
        self.load_encoder_checkpoint(required=False, label="Multi-View Conv encoder")


# Backwards compatible alias used in older configs/notebooks.
ConvVaeSensorProcessing_multiview = MultiViewConvVAESensorProcessing
