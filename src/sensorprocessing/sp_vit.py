# """
# Sensor processing using Vision Transformer (ViT) model
# """

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import SingleViewEncoderSensorProcessing
import torch.nn as nn
from .vit_helper import (
    create_image_preprocessing,
    create_projection,
    create_proprioceptor,
    create_vit_backbone,
    freeze_feature_extractor,
    normalize_if_unit_interval,
    resize_if_needed,
)


class ViTEncoder(nn.Module):
    """Neural network used to create our 128 d latent embedding using Vision Transformer architecture.

    The model extracts features using a pretrained ViT and projects them to our 128 latent.
    """

    def __init__(self, exp):
        super().__init__()
        # All values from config
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]

        self.vit_model, vit_output_dim = create_vit_backbone(exp)
        projection, projection_hidden_dim = create_projection(
            vit_output_dim, self.latent_size, exp
        )
        self.projection = projection
        self.proprioceptor = create_proprioceptor(
            self.latent_size, self.output_size, exp
        )

        print(f"Using {exp['vit_model']} with output dimension {vit_output_dim}")
        print(f"Created projection network: {vit_output_dim} → {projection_hidden_dim} → {projection_hidden_dim // 2} → {self.latent_size}")
        print(f"Created latent representation: {vit_output_dim} → {projection_hidden_dim} → {self.latent_size}")
        print(f"Created proprioceptor: {self.latent_size} → {exp.get('proprio_step_1', 128)} → {exp.get('proprio_step_2', 64)} → {self.output_size}")

        self.normalize, self.resize = create_image_preprocessing(exp["image_size"])

        # Freeze the feature extractor if specified
        if exp.get("freeze_feature_extractor", False):
            freeze_feature_extractor(self.vit_model)
            print("Feature extractor frozen. Projection and proprioceptor layers are trainable.")


        # Move to device
        self.to(Config().runtime["device"])

    def encode(self, x):
        """Extract 128 d latent representation without 6d proprioceptor.

        This is used during inference to get the 128-dimensional embedding.

        Returns:
            latent: 128-dimensional latent representation
        """
        x = resize_if_needed(x, self.resize)
        return self.projection(self.vit_model(self._normalize_input(x)))

    def _normalize_input(self, x):
        """Normalize input images to ImageNet statistics."""
        return normalize_if_unit_interval(x, self.normalize)

    def forward(self, x):
        """Forward pass to generate latent representation and then proprioceptor (6d)
            This forward function is only used during training
            for inference I will call the encode function

        """
        return self.proprioceptor(self.encode(x))


class VitSensorProcessing(SingleViewEncoderSensorProcessing):
    """Sensor processing using Vision Transformer (ViT) architecture.

    This class handles image processing using a ViT model to extract our 128 embeddings .
    It only does the encoding step, not the
    regression to robot positions.
    """

    def __init__(self, exp):
        """Create the sensor model

        Args:
            exp (dict): Experiment configuration dictionary
        """
        super().__init__(exp)

        # Log configuration details
        print(f"Initializing ViT Sensor Processing:")
        print(f"  Model: {exp['vit_model']}")
        print(f"  Latent dimension: {exp['latent_size']}")
        # print(f"  Image size: {exp['image_size'][0]}x{exp['image_size'][1]}")
        print(f"  Image size: {exp['image_size']}")


        # Create the ViT encoder model
        self.enc = ViTEncoder(exp)

        self.load_encoder_checkpoint(required=False, label="ViT encoder")
