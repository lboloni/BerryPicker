"""
sp_propriotuned_cnn_multiview.py

Multi-view sensor processing using pretrained CNN backbones (VGG19, ResNet50).

Each camera view goes through the CNN backbone (shared by default), the
flattened features are reduced by a per-view projector, and the projected
view features are combined by a
:class:`sensorprocessing.multiview_fusion.MultiViewFusion` head (all five
fusion types are supported, the same heads as the ViT multi-view model). A
proprioceptor MLP maps the fused latent to the robot position during training.

Relevant exp/run keys (see ``experiment_configs/sensorprocessing_propriotuned_cnn_multiview``):

- ``model``: ``MultiViewVGG19Model`` or ``MultiViewResNetModel`` (legacy names
  ``VGG19ProprioTunedRegression_multiview`` / ``ResNetProprioTunedRegression_multiview``
  are accepted too)
- ``image_size``: the backbone feature width is derived from it, so any size works
- ``num_views``, ``cameras``, ``shared_backbone``, ``batched_backbone``
- ``view_projection_hidden_dim`` (default 512) and ``view_projection_dim``
  (default 256): the per-view projector ``features -> hidden -> d_view``
- ``fusion_type`` and the other fusion keys (see multiview_fusion)
- ``freeze_feature_extractor`` (default true)
- ``proprio_step_1`` / ``proprio_step_2``: proprioceptor widths
"""

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn
from torchvision import models

from .sensor_processing import MultiViewEncoderSensorProcessing
from .multiview_backbones import ViewBackbones, describe_multiview_model
from .multiview_fusion import fusion_from_exp
from .vit_helper import create_proprioceptor


def _image_size(exp):
    size = exp.get("image_size", [256, 256])
    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


class _MultiViewCNNModel(nn.Module):
    """Common implementation: CNN backbone(s) -> per-view projector -> fusion -> proprioceptor."""

    backbone_name = None

    def __init__(self, exp):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp.get("output_size", 6)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None
        self.image_size = _image_size(exp)

        self.backbones = ViewBackbones(
            self.create_feature_extractor,
            self.num_views,
            shared=exp.get("shared_backbone", True),
            freeze=exp.get("freeze_feature_extractor", True),
            batched=exp.get("batched_backbone", True),
        )
        self.flatten = nn.Flatten()
        self.feature_size = self._infer_feature_size()

        # Per-view projector (shared across views) that reduces the wide CNN
        # features to d_view before fusion.
        view_hidden = exp.get("view_projection_hidden_dim", 512)
        self.d_view = exp.get("view_projection_dim", 256)
        self.view_projector = nn.Sequential(
            nn.Linear(self.feature_size, view_hidden),
            nn.BatchNorm1d(view_hidden),
            nn.ReLU(),
            nn.Dropout(exp.get("fusion_dropout", 0.1)),
            nn.Linear(view_hidden, self.d_view),
            nn.BatchNorm1d(self.d_view),
            nn.ReLU(),
        )
        print(
            f"Created per-view projector: {self.feature_size} -> {view_hidden} -> {self.d_view}"
        )

        self.fusion = fusion_from_exp(exp, self.d_view, self.num_views, self.latent_size)
        self.proprioceptor = create_proprioceptor(self.latent_size, self.output_size, exp)

        self.to(Config().runtime["device"])
        describe_multiview_model(self, exp, type(self).__name__)

    # -- backbone -----------------------------------------------------------
    def create_feature_extractor(self):
        raise NotImplementedError

    def _infer_feature_size(self):
        """Run one dummy image through the backbone to learn the flattened width."""
        backbone = self.backbones.backbone if self.backbones.shared else self.backbones.backbones[0]
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.image_size, device=next(backbone.parameters()).device)
            feature_size = self.flatten(backbone(dummy)).size(1)
        backbone.train(was_training and not self.backbones.freeze)
        return feature_size

    # -- encoding -----------------------------------------------------------
    def extract_features(self, views_list):
        """Per-view projected features of shape ``[batch, d_view]``."""
        raw = self.backbones(views_list)
        return [self.view_projector(self.flatten(features)) for features in raw]

    def encode_views(self, views_list):
        """Fused latent of shape ``[batch, latent_size]`` for an ordered view list."""
        views_list = list(views_list)
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")
        return self.fusion(self.extract_features(views_list))

    def encode(self, views_list):
        return self.encode_views(views_list)

    def forward(self, views_list):
        return self.proprioceptor(self.encode_views(views_list))


class MultiViewVGG19Model(_MultiViewCNNModel):
    """Multi-view VGG19 encoder."""

    backbone_name = "vgg19"

    def create_feature_extractor(self):
        return models.vgg19(weights=models.VGG19_Weights.DEFAULT).features


class MultiViewResNetModel(_MultiViewCNNModel):
    """Multi-view ResNet50 encoder."""

    backbone_name = "resnet50"

    def create_feature_extractor(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        return nn.Sequential(*list(resnet.children())[:-1])


_MODEL_CLASSES = {
    "MultiViewVGG19Model": MultiViewVGG19Model,
    "MultiViewResNetModel": MultiViewResNetModel,
    # legacy run names
    "VGG19ProprioTunedRegression_multiview": MultiViewVGG19Model,
    "ResNetProprioTunedRegression_multiview": MultiViewResNetModel,
}


def create_multiview_cnn_model(exp):
    """Instantiate the multi-view CNN model named by ``exp["model"]``."""
    model_name = exp.get("model", "MultiViewVGG19Model")
    try:
        model_class = _MODEL_CLASSES[model_name]
    except KeyError as error:
        available = ", ".join(_MODEL_CLASSES)
        raise ValueError(
            f"Unknown multi-view CNN model: {model_name!r}. Available models: {available}"
        ) from error
    return model_class(exp)


class MultiViewCNNSensorProcessing(MultiViewEncoderSensorProcessing):
    """Runtime wrapper: ordered camera views -> fused latent (no regression)."""

    encoder_method = "encode_views"

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = exp.get("num_views", 2)
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.cameras = list(exp.get("cameras", []))[: self.num_views] or None

        print("Initializing Multi-View CNN Sensor Processing:")
        print(f"  Model: {exp.get('model', 'MultiViewVGG19Model')}")
        print(f"  Number of views: {self.num_views}")
        print(f"  Fusion type: {self.fusion_type}")
        print(f"  Latent dimension: {exp['latent_size']}")
        print(f"  Camera order: {self.cameras}")

        self.enc = create_multiview_cnn_model(exp)
        self.load_encoder_checkpoint(required=False, label="Multi-View CNN encoder")
