"""Single-view VAE-GAN with a Neo VAE and learned reconstruction features."""

import torch
from torch import nn

from .conv_vae_neo import ConvVAENeo, _positive_int, _group_count


class VAEGANDiscriminator(nn.Module):
    """Resolution-dependent discriminator; return logits and spatial features."""

    def __init__(self, exp):
        super().__init__()
        base = _positive_int(exp.get("discriminator_base_channels", 32), "discriminator_base_channels")
        maximum = _positive_int(exp.get("discriminator_max_channels", 256), "discriminator_max_channels")
        if maximum < base:
            raise ValueError("discriminator_max_channels must be at least discriminator_base_channels")
        size = max(exp["image_size"])
        blocks, previous = [], 3
        while size > 4 or not blocks:
            width = min(base * 2 ** len(blocks), maximum)
            blocks.append(nn.Sequential(
                nn.Conv2d(previous, width, 3, stride=2, padding=1),
                nn.GroupNorm(_group_count(width, 8), width),
                nn.LeakyReLU(0.2),
            ))
            previous, size = width, (size + 1) // 2
        self.blocks = nn.ModuleList(blocks)
        self.feature_layer = exp.get("discriminator_feature_layer", -2)
        if type(self.feature_layer) is not int or not -len(blocks) <= self.feature_layer < len(blocks):
            raise ValueError("discriminator_feature_layer is outside the discriminator")
        self.feature_layer %= len(blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(previous, 1))

    def forward(self, images):
        features = None
        for index, block in enumerate(self.blocks):
            images = block(images)
            if index == self.feature_layer:
                features = images
        return self.classifier(images), features


class VAEGAN(nn.Module):
    """Training model. Only ``vae.state_dict()`` is exported for inference."""

    def __init__(self, exp):
        super().__init__()
        self.vae = ConvVAENeo(exp)
        self.discriminator = VAEGANDiscriminator(exp)

    def encode(self, images):
        return self.vae.encode(images)

    def encode_distribution(self, images):
        return self.vae.encode_distribution(images)

    def decode(self, latent):
        return self.vae.decode(latent)

    def sample(self, count):
        return self.vae.sample(count)

    def forward(self, images):
        return self.vae(images)


def kl_loss(mu, logvar):
    """Mean per-latent KL, matching ConvVAENeoLoss's normalization."""
    return -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
