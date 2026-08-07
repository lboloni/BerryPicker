"""Sensor processing using pretrained, proprioception-tuned CNNs."""

import sys

sys.path.append("..")

from exp_run_config import Config

Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import SingleViewEncoderSensorProcessing

import torch.nn as nn
from torchvision import models


class _ProprioTunedCNNRegression(nn.Module):
    """Shared backbone-to-latent-to-proprioception CNN regression pipeline."""

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.feature_extractor = self.create_feature_extractor()
        self.flatten = nn.Flatten()
        self.create_heads(exp)

        if exp.get("freeze_feature_extractor", True):
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        self.to(Config().runtime["device"])

    def create_feature_extractor(self):
        raise NotImplementedError

    def create_heads(self, exp):
        raise NotImplementedError

    def encode_flat_features(self, flat_features):
        raise NotImplementedError

    def predict_from_latent(self, latent):
        raise NotImplementedError

    def encode(self, x):
        features = self.feature_extractor(x)
        return self.encode_flat_features(self.flatten(features))

    def forward(self, x):
        return self.predict_from_latent(self.encode(x))


class VGG19ProprioTunedRegression(_ProprioTunedCNNRegression):
    """VGG19 backbone with the original VGG regression head."""

    def create_feature_extractor(self):
        return models.vgg19(pretrained=True).features

    def create_heads(self, _exp):
        self.model = nn.Sequential(
            nn.Linear(512 * 8 * 8, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.output_size),
        )

    def encode_flat_features(self, flat_features):
        return self.model[:3](flat_features)

    def predict_from_latent(self, latent):
        return self.model[3:](latent)


class ResNetProprioTunedRegression(_ProprioTunedCNNRegression):
    """ResNet50 backbone with a reductor and proprioception head."""

    def create_feature_extractor(self):
        resnet = models.resnet50(pretrained=True)
        self.feature_size = resnet.fc.in_features
        return nn.Sequential(*list(resnet.children())[:-1])

    def create_heads(self, exp):
        self.reductor = nn.Sequential(
            nn.Linear(self.feature_size, exp["reductor_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["reductor_step_1"], self.latent_size),
        )
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, exp["proprio_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_1"], exp["proprio_step_2"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_2"], self.output_size),
        )

    def encode_flat_features(self, flat_features):
        return self.reductor(flat_features)

    def predict_from_latent(self, latent):
        return self.proprioceptor(latent)


class ProprioTunedCNNSensorProcessing(SingleViewEncoderSensorProcessing):
    """Runtime wrapper for a configured proprioception-tuned CNN encoder."""

    encoder_classes = {
        "VGG19ProprioTunedRegression": VGG19ProprioTunedRegression,
        "ResNetProprioTunedRegression": ResNetProprioTunedRegression,
    }

    def __init__(self, exp):
        super().__init__(exp)
        try:
            encoder_class = self.encoder_classes[exp["model"]]
        except KeyError as error:
            available = ", ".join(self.encoder_classes)
            raise ValueError(
                f"Unknown proprioception-tuned CNN model: {exp.get('model')!r}. "
                f"Available models: {available}"
            ) from error

        self.enc = encoder_class(exp).to(Config().runtime["device"])
        self.load_encoder_checkpoint(required=True, label=encoder_class.__name__)
