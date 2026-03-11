# """
# Sensor processing using Vision Transformer (ViT) model
# """

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_sp
import pathlib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from torchvision import transforms


class ViTEncoder(nn.Module):
    """Neural network used to create our 128 d latent embedding using Vision Transformer architecture.

    The model extracts features using a pretrained ViT and projects them to our 128 latent.
    """

    def __init__(self, exp):
        super().__init__()
        # All values from config
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]

        # Load the ViT model based on configuration
        vit_model_name = exp["vit_model"]
        vit_weights_name = exp["vit_weights"]

        # Handle the model and weights imports
        if vit_model_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = getattr(ViT_B_16_Weights, vit_weights_name)
            self.vit_model = vit_b_16(weights=weights)
            vit_output_dim = exp["vit_output_dim"]  # ViT-B has 768 hidden dim
        elif vit_model_name == "vit_l_16":
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            weights = getattr(ViT_L_16_Weights, vit_weights_name)
            self.vit_model = vit_l_16(weights=weights)
            vit_output_dim = exp["vit_output_dim"]  # ViT-L has 1024 hidden dim
        elif vit_model_name == "vit_h_14":
            from torchvision.models import vit_h_14, ViT_H_14_Weights
            weights = getattr(ViT_H_14_Weights, vit_weights_name)
            self.vit_model = vit_h_14(weights=weights)
            vit_output_dim = exp["vit_output_dim"]   # ViT-H has 1280 hidden dim
        else:
            raise ValueError(f"Unsupported ViT model type: {vit_model_name}")

        # Override with config value if provided
        if "vit_output_dim" in exp:
            vit_output_dim = exp["vit_output_dim"]

        ## ;ets see if the putput dimention matches my VIT model

        print(f"Using {vit_model_name} with output dimension {vit_output_dim}")

        # Replace the head with our custom projection to get better regression
        if "projection_hidden_dim" in exp:
            projection_hidden_dim = exp["projection_hidden_dim"]
        else:
            # Default to a reasonable size based on input dimension
            projection_hidden_dim = vit_output_dim // 2

        # More sophisticated projection network with multiple layers
        self.projection = nn.Sequential(
            nn.Linear(vit_output_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden_dim, projection_hidden_dim // 2),
            nn.BatchNorm1d(projection_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim // 2, self.latent_size),
        )



        print(f"Created projection network: {vit_output_dim} → {projection_hidden_dim} → {projection_hidden_dim // 2} → {self.latent_size}")

        # Replace the ViT's head with our projection
        self.vit_model.heads = nn.Identity()  # Remove original classification head

        # Add proprioceptor for end-to-end training (not used for inference)
        # This maps from latent space to 6d
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, exp["proprio_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_1"], exp["proprio_step_2"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_2"], self.output_size)
        )

        print(f"Created latent representation: {vit_output_dim} → {projection_hidden_dim} → {self.latent_size}")
        print(f"Created proprioceptor: {self.latent_size} → {exp.get('proprio_step_1', 128)} → {exp.get('proprio_step_2', 64)} → {self.output_size}")

        # some papers said this is a good idea for VIT we can try without Normalize as well
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Create a resize transform to the required vit input size
        # input_size = (exp["image_size"], exp["image_size"])
        input_size = (exp["image_size"])

        self.resize = transforms.Resize(input_size, antialias=True)

        # Freeze the feature extractor if specified
        if exp["freeze_feature_extractor"]:
            # Freeze all parameters except the head
            for name, param in self.vit_model.named_parameters():
                if "heads" not in name:
                    param.requires_grad = False
            print("Feature extractor frozen. Projection and proprioceptor layers are trainable.")


        # Move to device
        self.to(Config().runtime["device"])

    def encode(self, x):
        """Extract 128 d latent representation without 6d proprioceptor.

        This is used during inference to get the 128-dimensional embedding.

        Returns:
            latent: 128-dimensional latent representation
        """
        # Resize and normalize input if needed
        if x.size(2) != x.size(3) or x.size(2) != self.resize.size[0]:
            x = self.resize(x)

        x = self._normalize_input(x)

        # Forward through base ViT (without its head)
        features = self.vit_model(x)

        # Project to latent space only (no proprioceptor)
        latent = self.projection(features)

        return latent

    def _normalize_input(self, x):
        """Normalize input images to ImageNet statistics."""
        # Check if already normalized
        if x.min() >= 0 and x.max() <= 1:
            # Convert [0,1] to normalized range matching ImageNet stats
            return self.normalize(x)
        return x

    def forward(self, x):
        """Forward pass to generate latent representation and then proprioceptor (6d)
            This forward function is only used during training
            for inference I will call the encode function

        """
        # Resize the input image to 224x224 as expected by ViT
        x = self.resize(x)

        # Use the full ViT model with our custom head
        latent = self.vit_model(x)

        # Resize and normalize input
        if x.size(2) != x.size(3) or x.size(2) != self.resize.size[0]:
            x = self.resize(x)

        x = self._normalize_input(x)

        # Forward through base ViT (without its head)
        features = self.vit_model(x)

        # Project to latent space
        latent = self.projection(features)

        # Map to robot position prediction via proprioceptor
        output = self.proprioceptor(latent)

        return output


class VitSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using Vision Transformer (ViT) architecture.

    This class handles image processing using a ViT model to extract our 128 embeddings .
    It only does the encoding step, not the
    regression to robot positions.
    """

    def __init__(self, exp):
        """Create the sensor model

        Args:
            exp (dict): Experiment configuration dictionary
            device (str, optional): Device to run the model on. Defaults to "cpu".
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

        # Load weights if model file exists
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            print(f"Loading ViT encoder weights from {modelfile}")
            self.enc.load_state_dict(torch.load(modelfile, map_location=Config().runtime["device"]))
        else:
            print(f"Warning: Model file {modelfile} does not exist. Using untrained model.")

        # Set model to evaluation mode
        self.enc.eval()

    def process(self, sensor_readings):
        """Process sensor readings (images) to produce embeddings.

        Args:
            sensor_readings: Image tensor prepared into a batch

        Returns:
            Embedding vector as numpy array with dimensions batch x 128
        """
        self.enc.eval()
        with torch.no_grad():
            # Use the encode func method which returns just the latent representation
            # without passing through the proprioceptor
            z = self.enc.encode(sensor_readings)
        z = torch.squeeze(z)
        return z.cpu().numpy()


