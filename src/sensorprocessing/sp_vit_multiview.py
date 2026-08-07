# """
# Multi-view Sensor processing using Vision Transformer (ViT) model
# """

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import MultiViewEncoderSensorProcessing
import torch
import torch.nn as nn
from .vit_helper import (
    create_image_preprocessing,
    create_proprioceptor,
    create_vit_backbone,
    freeze_feature_extractor,
    normalize_if_unit_interval,
    resize_if_needed,
)


class MultiViewViTEncoder(nn.Module):
    """Neural network used to create our 128d latent embedding using multiple Vision Transformer architectures.

    The model extracts features from multiple camera views using pretrained ViTs and fuses them
    into a single 128d latent representation.
    """

    def __init__(self, exp):
        super().__init__()
        # All values from config
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.num_views = exp.get("num_views", 2)  # Default to 2 views
        self.fusion_type = exp.get("fusion_type", "concat_proj")  # Default fusion method

        first_model, vit_output_dim = create_vit_backbone(exp)
        self.vit_models = nn.ModuleList(
            [first_model] + [create_vit_backbone(exp)[0] for _ in range(self.num_views - 1)]
        )

        print(f"Using {self.num_views} x {exp['vit_model']} with output dimension {vit_output_dim}")

        # Determine projection architecture based on fusion type
        if "projection_hidden_dim" in exp:
            projection_hidden_dim = exp["projection_hidden_dim"]
        else:
            # Default to a reasonable size based on input dimension
            projection_hidden_dim = vit_output_dim // 2

        # Define different fusion strategies
        if self.fusion_type == "concat_proj":
            # Concatenate features then project
            self.projection = nn.Sequential(
                nn.Linear(vit_output_dim * self.num_views, projection_hidden_dim),
                nn.BatchNorm1d(projection_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_hidden_dim, projection_hidden_dim // 2),
                nn.BatchNorm1d(projection_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim // 2, self.latent_size),
            )
            print(f"Created fusion network (concat_proj): {vit_output_dim*self.num_views} → {projection_hidden_dim} → {projection_hidden_dim//2} → {self.latent_size}")

        elif self.fusion_type == "indiv_proj":
            self.view_projections = self._create_view_projections(
                vit_output_dim, projection_hidden_dim
            )

            # Fusion layer to combine individual projections
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.latent_size * self.num_views, self.latent_size),
                nn.BatchNorm1d(self.latent_size),
                nn.ReLU()
            )
            print(f"Created individual projections: {vit_output_dim} → {projection_hidden_dim} → {self.latent_size}")
            print(f"Created fusion layer: {self.latent_size*self.num_views} → {self.latent_size}")

        elif self.fusion_type == "attention":
            # Cross-attention fusion
            self.query_proj = nn.Linear(vit_output_dim, projection_hidden_dim)
            self.key_proj = nn.Linear(vit_output_dim, projection_hidden_dim)
            self.value_proj = nn.Linear(vit_output_dim, projection_hidden_dim)

            self.attention = nn.MultiheadAttention(
                embed_dim=projection_hidden_dim,
                num_heads=4,
                batch_first=True
            )

            self.final_proj = nn.Sequential(
                nn.Linear(projection_hidden_dim, projection_hidden_dim // 2),
                nn.BatchNorm1d(projection_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim // 2, self.latent_size),
            )
            print(f"Created attention fusion: {vit_output_dim} → {projection_hidden_dim} → {projection_hidden_dim//2} → {self.latent_size}")

        elif self.fusion_type == "weighted_sum":
            self.view_projections = self._create_view_projections(
                vit_output_dim, projection_hidden_dim
            )

            # Learnable weights for each view
            self.view_weights = nn.Parameter(torch.ones(self.num_views) / self.num_views)
            print(f"Created weighted sum fusion with learnable weights")

        elif self.fusion_type == "gated":
            self.view_projections = self._create_view_projections(
                vit_output_dim, projection_hidden_dim, dropout=False
            )

            # Gate network to determine importance of each view
            self.gate_network = nn.Sequential(
                nn.Linear(vit_output_dim * self.num_views, self.num_views),
                nn.Softmax(dim=1)
            )
            print(f"Created gated fusion network")
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        self.proprioceptor = create_proprioceptor(
            self.latent_size, self.output_size, exp
        )

        print(f"Created proprioceptor: {self.latent_size} → {exp.get('proprio_step_1', 128)} → {exp.get('proprio_step_2', 64)} → {self.output_size}")

        self.normalize, self.resize = create_image_preprocessing(exp["image_size"])

        # Freeze the feature extractor if specified
        if exp.get("freeze_feature_extractor", False):
            for model in self.vit_models:
                freeze_feature_extractor(model)
            print("Feature extractors frozen. Projection and proprioceptor layers are trainable.")

        # Move to device
        self.to(Config().runtime["device"])

    def _create_view_projections(self, input_size, hidden_size, dropout=True):
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_size, self.latent_size))
        return nn.ModuleList(nn.Sequential(*layers) for _ in range(self.num_views))

    def encode_single_view(self, x, view_idx=0):
        """Extract features from a single view."""
        x = resize_if_needed(x, self.resize)
        return self.vit_models[view_idx](self._normalize_input(x))

    def encode(self, views_list):
        """Extract 128d latent representation from multiple views without 6d proprioceptor.

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            latent: 128-dimensional latent representation
        """
        # Make sure we have the right number of views
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")

        features_list = [
            self.encode_single_view(view, index)
            for index, view in enumerate(views_list)
        ]

        # Apply fusion based on the chosen method
        if self.fusion_type == "concat_proj":
            # Concatenate features then project
            combined_features = torch.cat(features_list, dim=1)
            latent = self.projection(combined_features)

        elif self.fusion_type == "indiv_proj":
            # Project each view individually then fuse
            latent_views = self._project_views(features_list)
            combined_latents = torch.cat(latent_views, dim=1)
            latent = self.fusion_layer(combined_latents)

        elif self.fusion_type == "attention":
            # Reshape for attention: [batch_size, num_views, feature_dim]
            stacked_features = torch.stack(features_list, dim=1)

            # Apply projections for query, key, value
            query = self.query_proj(stacked_features)
            key = self.key_proj(stacked_features)
            value = self.value_proj(stacked_features)

            # Apply attention mechanism
            attn_output, _ = self.attention(query, key, value)

            # Take mean across views dimension to get a single vector per batch
            fused_features = torch.mean(attn_output, dim=1)

            # Final projection to latent size
            latent = self.final_proj(fused_features)

        elif self.fusion_type == "weighted_sum":
            # Project each view to latent space
            latent_views = self._project_views(features_list)

            # Apply learnable weights
            weights = torch.softmax(self.view_weights, dim=0)
            latent = torch.zeros_like(latent_views[0])
            for i, view_latent in enumerate(latent_views):
                latent += weights[i] * view_latent

        elif self.fusion_type == "gated":
            # Project each view to latent space
            latent_views = self._project_views(features_list)

            # Concatenate features for gate determination
            combined_features = torch.cat(features_list, dim=1)
            gates = self.gate_network(combined_features)

            # Apply gates to each view's latent
            latent = torch.zeros_like(latent_views[0])
            for i, view_latent in enumerate(latent_views):
                latent += gates[:, i:i+1] * view_latent

        return latent

    def _project_views(self, features_list):
        return [
            self.view_projections[index](features)
            for index, features in enumerate(features_list)
        ]

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


class MultiViewVitSensorProcessing(MultiViewEncoderSensorProcessing):
    """Multi-view sensor processing using Vision Transformer (ViT) architecture.

    This class handles image processing using multiple ViT models to extract a fused 128d embedding.
    It only does the encoding step, not the regression to robot positions.
    """

    def __init__(self, exp):
        """Create the sensor model

        Args:
            exp (dict): Experiment configuration dictionary
        """
        super().__init__(exp)

        # Log configuration details
        print(f"Initializing Multi-View ViT Sensor Processing:")
        print(f"  Model: {exp['vit_model']}")
        print(f"  Number of views: {exp.get('num_views', 2)}")
        print(f"  Fusion type: {exp.get('fusion_type', 'concat_proj')}")
        print(f"  Latent dimension: {exp['latent_size']}")
        print(f"  Image size: {exp['image_size']}x{exp['image_size']}")

        # Create the multi-view ViT encoder model
        self.enc = MultiViewViTEncoder(exp)

        self.load_encoder_checkpoint(required=False, label="Multi-View ViT encoder")
