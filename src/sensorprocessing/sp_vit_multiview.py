"""
sp_vit_multiview.py

Multi-view sensor processing using a Vision Transformer (ViT) backbone.

Every camera view is encoded by a pretrained ViT (by default one backbone
shared by all views), the per-view features are combined by a
:class:`sensorprocessing.multiview_fusion.MultiViewFusion` head, and a small
proprioceptor MLP maps the fused latent to the robot position during training.

Relevant exp/run keys (see ``experiment_configs/sensorprocessing_propriotuned_Vit_multiview``):

- ``vit_model`` / ``vit_weights`` / ``vit_output_dim``: backbone (see vit_helper)
- ``num_views``: number of camera views
- ``cameras``: the ordered camera list the model is trained with (the training
  data entries carry the order; this is used to warn about mismatches)
- ``shared_backbone`` (default true): one ViT for all views, or one per view
- ``batched_backbone`` (default true): run all views through a shared backbone
  in one batched call
- ``fusion_type`` and the other fusion keys (see multiview_fusion)
- ``freeze_feature_extractor``: freeze the ViT weights
- ``proprio_step_1`` / ``proprio_step_2``: proprioceptor widths
"""

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch.nn as nn

from .sensor_processing import MultiViewEncoderSensorProcessing
from .multiview_backbones import ViewBackbones, describe_multiview_model
from .multiview_fusion import fusion_from_exp
from .vit_helper import (
    create_image_preprocessing,
    create_proprioceptor,
    create_vit_backbone,
    normalize_if_unit_interval,
    resize_if_needed,
)


class MultiViewViTEncoder(nn.Module):
    """ViT backbone(s) + fusion head + proprioceptor for multi-view proprioception."""

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.num_views = exp.get("num_views", 2)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None

        # The first backbone tells us the feature width; the remaining ones
        # (if the backbone is not shared) are built by the container.
        first_backbone, vit_output_dim = create_vit_backbone(exp)
        self.vit_output_dim = vit_output_dim
        pending = [first_backbone]

        def make_backbone():
            if pending:
                return pending.pop()
            return create_vit_backbone(exp)[0]

        self.backbones = ViewBackbones(
            make_backbone,
            self.num_views,
            shared=exp.get("shared_backbone", True),
            freeze=exp.get("freeze_feature_extractor", False),
            batched=exp.get("batched_backbone", True),
        )
        print(f"Using {exp['vit_model']} backbone(s) with output dimension {vit_output_dim}")

        self.fusion = fusion_from_exp(exp, vit_output_dim, self.num_views, self.latent_size)
        self.proprioceptor = create_proprioceptor(self.latent_size, self.output_size, exp)
        print(
            f"Created proprioceptor: {self.latent_size} -> "
            f"{exp.get('proprio_step_1', 128)} -> {exp.get('proprio_step_2', 64)} -> {self.output_size}"
        )

        self.normalize, self.resize = create_image_preprocessing(exp["image_size"])

        self.to(Config().runtime["device"])
        describe_multiview_model(self, exp, "MultiViewViTEncoder")

    # -- image preprocessing ------------------------------------------------
    def _prepare_view(self, x):
        """Resize to the ViT input size and apply ImageNet normalization."""
        x = resize_if_needed(x, self.resize)
        return normalize_if_unit_interval(x, self.normalize)

    # -- encoding -----------------------------------------------------------
    def extract_features(self, views_list):
        """Return the per-view ViT feature vectors (before fusion)."""
        return self.backbones([self._prepare_view(view) for view in views_list])

    def encode_single_view(self, x, view_idx=0):
        """Encode one view with the backbone that would process view ``view_idx``."""
        if self.backbones.shared:
            backbone = self.backbones.backbone
        else:
            backbone = self.backbones.backbones[view_idx]
        return backbone(self._prepare_view(x))

    def encode(self, views_list):
        """Fused latent of shape ``[batch, latent_size]`` for an ordered view list."""
        views_list = list(views_list)
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")
        return self.fusion(self.extract_features(views_list))

    # ``encode_views`` is the name used by the CNN and Conv encoders; provide
    # both so training and evaluation code can be written once.
    def encode_views(self, views_list):
        return self.encode(views_list)

    def forward(self, views_list):
        """Robot position prediction; used only during sensor-processing training."""
        return self.proprioceptor(self.encode(views_list))


class MultiViewVitSensorProcessing(MultiViewEncoderSensorProcessing):
    """Runtime wrapper: ordered camera views -> fused latent (no regression)."""

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = exp.get("num_views", 2)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None

        print("Initializing Multi-View ViT Sensor Processing:")
        print(f"  Model: {exp['vit_model']}")
        print(f"  Number of views: {self.num_views}")
        print(f"  Fusion type: {self.fusion_type}")
        print(f"  Latent dimension: {exp['latent_size']}")
        print(f"  Image size: {exp['image_size']}")
        print(f"  Camera order: {self.cameras}")

        self.enc = MultiViewViTEncoder(exp)
        self.load_encoder_checkpoint(required=False, label="Multi-View ViT encoder")
