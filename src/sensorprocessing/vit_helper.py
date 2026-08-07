"""Shared building blocks for the single- and multi-view ViT encoders."""

import torch.nn as nn
from torchvision import transforms


_VIT_VARIANTS = {
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights", 768),
    "vit_l_16": ("vit_l_16", "ViT_L_16_Weights", 1024),
    "vit_h_14": ("vit_h_14", "ViT_H_14_Weights", 1280),
}


def create_vit_backbone(exp):
    """Create a pretrained ViT with its classification head removed."""
    model_name = exp["vit_model"]
    try:
        constructor_name, weights_name, default_output_dim = _VIT_VARIANTS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported ViT model type: {model_name}") from error

    from torchvision import models

    constructor = getattr(models, constructor_name)
    weights_class = getattr(models, weights_name)
    model = constructor(weights=getattr(weights_class, exp["vit_weights"]))
    model.heads = nn.Identity()
    return model, exp.get("vit_output_dim", default_output_dim)


def create_projection(input_size, latent_size, exp):
    """Build the standard ViT-feature-to-latent projection head."""
    hidden_size = exp.get("projection_hidden_dim", input_size // 2)
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.BatchNorm1d(hidden_size // 2),
        nn.ReLU(),
        nn.Linear(hidden_size // 2, latent_size),
    ), hidden_size


def create_proprioceptor(latent_size, output_size, exp):
    """Build the common latent-to-proprioception regression head."""
    first_hidden = exp.get("proprio_step_1", 128)
    second_hidden = exp.get("proprio_step_2", 64)
    return nn.Sequential(
        nn.Linear(latent_size, first_hidden),
        nn.ReLU(),
        nn.Linear(first_hidden, second_hidden),
        nn.ReLU(),
        nn.Linear(second_hidden, output_size),
    )


def create_image_preprocessing(image_size):
    """Return ImageNet normalization and a resize transform for a configured size."""
    target_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    return (
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(target_size, antialias=True),
    )


def resize_if_needed(x, resize):
    """Resize a batch only when its spatial dimensions differ from the target."""
    return resize(x) if tuple(x.shape[-2:]) != tuple(resize.size) else x


def normalize_if_unit_interval(x, normalize):
    """Normalize raw [0, 1] images, leaving pre-normalized tensors unchanged."""
    return normalize(x) if x.min() >= 0 and x.max() <= 1 else x


def freeze_feature_extractor(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
