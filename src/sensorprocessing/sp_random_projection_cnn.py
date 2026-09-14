"""Training-free CNN sensor processing with a fixed random projection."""

import math
import sys

sys.path.append("..")

from exp_run_config import Config

Config.PROJECTNAME = "BerryPicker"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .sensor_processing import SingleViewEncoderSensorProcessing


class _RandomProjectionCNN(nn.Module):
    """Frozen CNN features followed by a deterministic random projection."""

    weights_enum = None

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.feature_normalization = exp["feature_normalization"]

        weights = self._configured_weights(exp["backbone_weights"])
        self.feature_extractor = self.create_feature_extractor(weights)
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()
        self.pool = nn.AdaptiveAvgPool2d(tuple(exp["feature_pool_size"]))
        self.flatten = nn.Flatten()

        if weights is None:
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            transforms = weights.transforms()
            mean, std = transforms.mean, transforms.std
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1))

        self.feature_size = self._feature_size(exp["image_size"])
        projection = self._create_projection(
            exp["projection"], exp["projection_seed"]
        )
        self.register_buffer("projection", projection)

        self.to(Config().runtime["device"])
        self.eval()

    def _configured_weights(self, name):
        if name == "NONE":
            return None
        return self.weights_enum[name]

    def create_feature_extractor(self, weights):
        raise NotImplementedError

    def _feature_size(self, image_size):
        with torch.no_grad():
            sample = torch.zeros(1, 3, *image_size)
            features = self.pool(self.feature_extractor(sample))
        return self.flatten(features).shape[1]

    def _create_projection(self, distribution, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.empty(self.latent_size, self.feature_size)

        if distribution == "rademacher":
            projection.bernoulli_(0.5, generator=generator)
            projection.mul_(2).sub_(1)
        elif distribution == "gaussian":
            projection.normal_(generator=generator)
        else:
            raise ValueError(f"Unknown projection distribution: {distribution}")
        return projection / math.sqrt(self.latent_size)

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def encode(self, images):
        images = (images - self.image_mean) / self.image_std
        features = self.flatten(self.pool(self.feature_extractor(images)))
        if self.feature_normalization == "l2":
            features = F.normalize(features, p=2, dim=1)
        elif self.feature_normalization != "none":
            raise ValueError(
                f"Unknown feature normalization: {self.feature_normalization}"
            )
        return F.linear(features, self.projection)

    def forward(self, images):
        return self.encode(images)


class ResNet50RandomProjection(_RandomProjectionCNN):
    """Random projection of pooled ResNet50 convolutional features."""

    weights_enum = models.ResNet50_Weights

    def create_feature_extractor(self, weights):
        resnet = models.resnet50(weights=weights)
        return nn.Sequential(*list(resnet.children())[:-2])


class VGG19RandomProjection(_RandomProjectionCNN):
    """Random projection of optionally pooled VGG19 convolutional features."""

    weights_enum = models.VGG19_Weights

    def create_feature_extractor(self, weights):
        return models.vgg19(weights=weights).features


class RandomProjectionCNNSensorProcessing(SingleViewEncoderSensorProcessing):
    """Runtime wrapper for a configured random-projection CNN."""

    encoder_classes = {
        "ResNet50RandomProjection": ResNet50RandomProjection,
        "VGG19RandomProjection": VGG19RandomProjection,
    }

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = self.encoder_classes[exp["model"]](exp).to(
            Config().runtime["device"]
        )
