"""
Sensor processing using pretrained CNN, with multi-view support
"""
import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import MultiViewEncoderSensorProcessing

import torch
import torch.nn as nn
from torchvision import models


class _MultiViewCNNModel(nn.Module):
    """Common implementation for per-view CNN backbones and regression heads."""

    feature_size = None
    default_reductor_size = None

    def __init__(self, exp):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.feature_extractors = nn.ModuleList(
            self._create_feature_extractor(exp) for _ in range(self.num_views)
        )
        self.flatten = nn.Flatten()
        reductor_size = exp.get("reductor_step_1", self.default_reductor_size)
        self.reductor = nn.Sequential(
            nn.Linear(self.feature_size * self.num_views, reductor_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(reductor_size, self.latent_size),
        )
        proprio_step_1 = exp.get("proprio_step_1", 128)
        proprio_step_2 = exp.get("proprio_step_2", 64)
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, proprio_step_1),
            nn.ReLU(),
            nn.Linear(proprio_step_1, proprio_step_2),
            nn.ReLU(),
            nn.Linear(proprio_step_2, self.output_size),
        )
        self.to(Config().runtime["device"])

    def _create_feature_extractor(self, exp):
        extractor = self.create_feature_extractor()
        if exp.get("freeze_feature_extractor", True):
            for parameter in extractor.parameters():
                parameter.requires_grad = False
        return extractor

    def create_feature_extractor(self):
        raise NotImplementedError

    def encode_views(self, views_list):
        features = [
            self.flatten(self.feature_extractors[index](view))
            for index, view in enumerate(views_list)
        ]
        return self.reductor(torch.cat(features, dim=1))

    def forward(self, views_list):
        return self.proprioceptor(self.encode_views(views_list))


class MultiViewVGG19Model(_MultiViewCNNModel):
    """Multi-view VGG19 encoder with a shared regression-head implementation."""

    feature_size = 512 * 8 * 8
    default_reductor_size = 512

    def create_feature_extractor(self):
        return models.vgg19(pretrained=True).features


class MultiViewResNetModel(_MultiViewCNNModel):
    """Multi-view ResNet50 encoder with a shared regression-head implementation."""

    feature_size = 2048
    default_reductor_size = 1024

    def create_feature_extractor(self):
        resnet = models.resnet50(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-1])

class MultiViewCNNSensorProcessing(MultiViewEncoderSensorProcessing):
    """
    Sensor processing class that handles multiple camera views using CNN encoders.

    This class manages the processing of multiple camera views, maintaining
    a cache of previously seen views to ensure complete processing even when
    only one view is updated at a time.
    """

    encoder_method = "encode_views"

    def __init__(self, exp):
        """
        Initialize the multi-view CNN sensor processing

        Args:
            exp: Experiment configuration
        """
        super().__init__(exp)

        # Log configuration details
        print(f"Initializing Multi-View CNN Sensor Processing:")
        print(f"  Model: {exp['model']}")
        print(f"  Number of views: {exp.get('num_views', 2)}")
        print(f"  Latent dimension: {exp['latent_size']}")

        # Create the encoder model based on configuration
        if exp['model'] == 'MultiViewVGG19Model':
            self.enc = MultiViewVGG19Model(exp)
        elif exp['model'] == 'MultiViewResNetModel':
            self.enc = MultiViewResNetModel(exp)
        else:
            raise ValueError(f"Unknown model type: {exp['model']}")

        self.load_encoder_checkpoint(required=False, label="Multi-View CNN encoder")
