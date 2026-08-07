# """
# Concatenated Image Vision Transformer (ViT) model for sensor processing
# """

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import MultiViewEncoderSensorProcessing
import torch
import torch.nn as nn
from torchvision.transforms import functional
from .vit_helper import (
    create_image_preprocessing,
    create_projection,
    create_proprioceptor,
    create_vit_backbone,
    freeze_feature_extractor,
    normalize_if_unit_interval,
)


class ConcatImageViTEncoder(nn.Module):
    """Neural network that concatenates multiple camera views before processing with ViT.

    The model horizontally concatenates images from different camera views, then
    processes the combined image through a single Vision Transformer to create a 128d
    latent representation.
    """

    def __init__(self, exp):
        super().__init__()
        # All values from config
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.num_views = exp.get("num_views", 2)  # Default to 2 views

        self.vit_model, vit_output_dim = create_vit_backbone(exp)
        projection, projection_hidden_dim = create_projection(
            vit_output_dim, self.latent_size, exp
        )
        self.projection = projection
        self.proprioceptor = create_proprioceptor(
            self.latent_size, self.output_size, exp
        )

        print(f"Using {exp['vit_model']} with output dimension {vit_output_dim}")
        print(f"Created projection network: {vit_output_dim} → {projection_hidden_dim} → {projection_hidden_dim//2} → {self.latent_size}")

        print(f"Created proprioceptor: {self.latent_size} → {exp.get('proprio_step_1', 128)} → {exp.get('proprio_step_2', 64)} → {self.output_size}")

        self.normalize, self.resize = create_image_preprocessing(exp["image_size"])
        self.image_size = tuple(self.resize.size)

        # Freeze the feature extractor if specified
        if exp.get("freeze_feature_extractor", False):
            freeze_feature_extractor(self.vit_model)
            print("Feature extractor frozen. Projection and proprioceptor layers are trainable.")

        # Move to device
        self.to(Config().runtime["device"])

    def concatenate_images(self, views_list):
        """Horizontally concatenate multiple image views into a single image.

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            concat_image: Single concatenated image tensor
        """
        # Make sure all images are the same size before concatenation
        resized_views = [
            self.resize(view) if tuple(view.shape[-2:]) != self.image_size else view
            for view in views_list
        ]

        # Concatenate along width dimension (dim=3)
        concat_image = torch.cat(resized_views, dim=3)

        # Resize the concatenated image to fit ViT input requirements
        # The height stays the same but width is num_views*width
        if self.vit_model.__class__.__name__.startswith("VisionTransformer"):
            # For standard ViTs, we need to resize to a square
            concat_image = functional.resize(
                concat_image,
                self.image_size,
                antialias=True
            )

        return concat_image

    def encode(self, views_list):
        """Extract 128d latent representation from concatenated views.

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            latent: 128-dimensional latent representation
        """
        # Make sure we have the right number of views
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")

        # Concatenate views
        concat_image = self.concatenate_images(views_list)

        # Normalize for ViT
        concat_image = self._normalize_input(concat_image)

        # Forward through ViT
        features = self.vit_model(concat_image)

        # Project to latent space
        latent = self.projection(features)

        return latent

    def _normalize_input(self, x):
        """Normalize input images to ImageNet statistics."""
        return normalize_if_unit_interval(x, self.normalize)

    def forward(self, views_list):
        """Forward pass to generate latent representation and then proprioceptor (6d)
        This forward function is used during training.
        For inference, call the encode function.

        Args:
            views_list: List of image tensors from different camera views
        """
        return self.proprioceptor(self.encode(views_list))


class ConcatImageVitSensorProcessing(MultiViewEncoderSensorProcessing):
    """Sensor processing that concatenates multiple images before processing with ViT.

    This class handles image preprocessing by concatenating multiple camera views
    into a single image, which is then processed through a single ViT model.
    """

    def __init__(self, exp):
        """Create the sensor model

        Args:
            exp (dict): Experiment configuration dictionary
        """
        super().__init__(exp)

        # Log configuration details
        print(f"Initializing Concatenated Image ViT Sensor Processing:")
        print(f"  Model: {exp['vit_model']}")
        print(f"  Number of views: {exp.get('num_views', 2)}")
        print(f"  Latent dimension: {exp['latent_size']}")
        print(f"  Image size: {exp['image_size']}x{exp['image_size']}")

        # Create the encoder model
        self.enc = ConcatImageViTEncoder(exp)

        self.load_encoder_checkpoint(
            required=False, label="Concatenated Image ViT encoder"
        )
