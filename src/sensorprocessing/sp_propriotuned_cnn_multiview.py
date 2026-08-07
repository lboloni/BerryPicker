"""
Sensor processing using pretrained CNN, with multi-view support
"""
import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import (
    MultiViewSensorProcessing,
)

import pathlib
import torch
import torch.nn as nn
from torchvision import models


# Multi-view CNN models
class MultiViewVGG19Model(nn.Module):
    """
    Neural network that processes multiple camera views using VGG19 encoders.

    The model processes each view separately through a VGG19 backbone,
    then concatenates the feature vectors before passing them through
    a regression head for proprioception prediction.
    """

    def __init__(self, exp):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]

        # Create separate VGG19 feature extractors for each view
        self.feature_extractors = nn.ModuleList()
        for _ in range(self.num_views):
            vgg19 = models.vgg19(pretrained=True)
            extractor = vgg19.features
            # Freeze the parameters of the feature extractor if specified
            if exp.get("freeze_feature_extractor", True):
                for param in extractor.parameters():
                    param.requires_grad = False
            self.feature_extractors.append(extractor)

        self.flatten = nn.Flatten()  # Flatten the output for the fully connected layer

        # Calculate the size of the concatenated feature vector
        # VGG19 features output size is 512 * 8 * 8 for each view
        concat_size = 512 * 8 * 8 * self.num_views

        # Dimension reduction network
        self.reductor = nn.Sequential(
            nn.Linear(concat_size, exp.get("reductor_step_1", 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(exp.get("reductor_step_1", 512), self.latent_size)
        )

        # Proprioception head for predicting robot position
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, exp.get("proprio_step_1", 128)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_1", 128), exp.get("proprio_step_2", 64)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_2", 64), self.output_size)
        )

        # Move the model to the specified device
        self.to(Config().runtime["device"])

    def encode_views(self, views_list):
        """
        Extract features from each view and concatenate them

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            latent: The latent representation of the concatenated views
        """
        # Process each view through its respective feature extractor
        features_list = []
        for i, view in enumerate(views_list):
            features = self.feature_extractors[i](view)
            flat_features = self.flatten(features)
            features_list.append(flat_features)

        # Concatenate the flattened features
        concat_features = torch.cat(features_list, dim=1)

        # Reduce dimensions to latent size
        latent = self.reductor(concat_features)

        return latent

    def forward(self, views_list):
        """
        Forward pass through the network

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            output: Predicted robot position
        """
        latent = self.encode_views(views_list)
        output = self.proprioceptor(latent)
        return output

class MultiViewResNetModel(nn.Module):
    """
    Neural network that processes multiple camera views using ResNet50 encoders.

    The model processes each view separately through a ResNet50 backbone,
    then concatenates the feature vectors before passing them through
    a regression head for proprioception prediction.
    """

    def __init__(self, exp):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]

        # Create separate ResNet feature extractors for each view
        self.feature_extractors = nn.ModuleList()
        for _ in range(self.num_views):
            resnet = models.resnet50(pretrained=True)
            # Create feature extractor by removing the last fully connected layer
            extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
            # Freeze the parameters of the feature extractor if specified
            if exp.get("freeze_feature_extractor", True):
                for param in extractor.parameters():
                    param.requires_grad = False
            self.feature_extractors.append(extractor)

        self.flatten = nn.Flatten()

        # ResNet50 features size is 2048 per view
        concat_size = 2048 * self.num_views

        # Dimension reduction network
        self.reductor = nn.Sequential(
            nn.Linear(concat_size, exp.get("reductor_step_1", 1024)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(exp.get("reductor_step_1", 1024), self.latent_size)
        )

        # Proprioception head for predicting robot position
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, exp.get("proprio_step_1", 128)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_1", 128), exp.get("proprio_step_2", 64)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_2", 64), self.output_size)
        )

        # Move the model to the specified device
        self.to(Config().runtime["device"])

    def encode_views(self, views_list):
        """
        Extract features from each view and concatenate them

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            latent: The latent representation of the concatenated views
        """
        # Process each view through its respective feature extractor
        features_list = []
        for i, view in enumerate(views_list):
            features = self.feature_extractors[i](view)
            flat_features = self.flatten(features)
            features_list.append(flat_features)

        # Concatenate the flattened features
        concat_features = torch.cat(features_list, dim=1)

        # Reduce dimensions to latent size
        latent = self.reductor(concat_features)

        return latent

    def forward(self, views_list):
        """
        Forward pass through the network

        Args:
            views_list: List of image tensors from different camera views

        Returns:
            output: Predicted robot position
        """
        latent = self.encode_views(views_list)
        output = self.proprioceptor(latent)
        return output

class MultiViewCNNSensorProcessing(MultiViewSensorProcessing):
    """
    Sensor processing class that handles multiple camera views using CNN encoders.

    This class manages the processing of multiple camera views, maintaining
    a cache of previously seen views to ensure complete processing even when
    only one view is updated at a time.
    """

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

        # Load weights if model file exists
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            print(f"Loading Multi-View CNN encoder weights from {modelfile}")
            self.enc.load_state_dict(torch.load(modelfile, map_location=Config().runtime["device"]))
        else:
            print(f"Warning: Model file {modelfile} does not exist. Using untrained model.")

        # Set model to evaluation mode
        self.enc.eval()

    def process(self, sensor_readings_list):
        """
        Process multiple sensor readings (images) to produce a single embedding.

        Args:
            sensor_readings_list: List of image tensors from different camera views

        Returns:
            Embedding vector as numpy array with dimensions batch x latent_size
        """
        self.enc.eval()
        with torch.no_grad():
            # Use the encode_views function to get the latent representation
            z = self.enc.encode_views(sensor_readings_list)
        z = torch.squeeze(z)
        return z.cpu().numpy()

class MultiViewVGG19SensorProcessing(MultiViewCNNSensorProcessing):
    """Convenience class for VGG19-based multi-view sensor processing"""

    def __init__(self, exp):
        # Ensure the model is set to VGG19
        exp_copy = exp.copy()
        exp_copy['model'] = 'MultiViewVGG19Model'
        super().__init__(exp_copy)

class MultiViewResNetSensorProcessing(MultiViewCNNSensorProcessing):
    """Convenience class for ResNet-based multi-view sensor processing"""

    def __init__(self, exp):
        # Ensure the model is set to ResNet
        exp_copy = exp.copy()
        exp_copy['model'] = 'MultiViewResNetModel'
        super().__init__(exp_copy)
