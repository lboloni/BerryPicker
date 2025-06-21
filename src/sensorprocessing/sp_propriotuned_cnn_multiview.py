"""
Sensor processing using pretrained CNN, with multi-view support
"""
import sys
sys.path.append("..")
from settings import Config

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_robot, load_picturefile_to_tensor

import pathlib
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import numpy as np


class VGG19ProprioTunedRegression(nn.Module):
    """Neural network used to create a latent embedding. Starts with a VGG19 neural network, without the classification head. The features are flattened, and fed into a regression MLP trained on visual proprioception.

    When used for encoding, the processing happens only to an internal layer in the MLP.
    """

    def __init__(self, exp, device):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        # Pretrained vgg19
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = vgg19.features
        self.flatten = nn.Flatten()  # Flatten the output for the fully connected layer
        self.model = nn.Sequential(
            # The internal size seem to depend on the external size.
            # the original with 7 * 7 corresponded to the 224 x 224 inputs
            #nn.Linear(512 * 7 * 7, hidden_size),
            nn.Linear(512 * 8 * 8, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.output_size)
        )
        # freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            # Move the whole thing to the GPU if available
        self.feature_extractor.to(device)
        self.model.to(device)


    def forward(self, x):
        """Forward the input image through the complete network for the purposes of training using the proprioception. Return a vector of output_size which """
        features = self.feature_extractor(x)
        #print(features.shape)
        flatfeatures = self.flatten(features)
        #print(flatfeatures.shape)
        output = self.model(flatfeatures)
        # print(output.device)
        return output

    def encode(self, x):
        """Performs an encoding of the input image, by forwarding though the encoding and first three layers."""
        features = self.feature_extractor(x)
        flatfeatures = self.flatten(features)
        h1 = self.model[0](flatfeatures)
        h2 = self.model[1](h1)
        h3 = self.model[2](h2)
        return h3


class ResNetProprioTunedRegression(nn.Module):
    """Neural network used to create a latent embedding. Starts with a ResNet neural network, without the classification head. The features are flattened, and fed into a regression MLP trained on visual proprioception.

    When used for encoding, the processing happens only to an internal layer in the MLP.
    """

    def __init__(self, exp, device):
        super(ResNetProprioTunedRegression, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Create the feature extractor by removing the last fully
        # connected layer of the resnet (fc)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # freeze the parameters of the feature extractor
        if exp["freeze_feature_extractor"]:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.flatten = nn.Flatten()  # Flatten the output for the fully
        # the reductor component
        self.reductor = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, exp["reductor_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["reductor_step_1"], exp["latent_size"])
        )

        # the proprioceptor auxiliary training component
        # not used in inference
        self.proprioceptor = nn.Sequential(
            nn.Linear(exp["latent_size"], exp["proprio_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_1"], exp["proprio_step_2"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_2"], exp["output_size"])
        )

        # Move the whole thing to the GPU if available
        self.feature_extractor.to(device)
        self.reductor.to(device)
        self.proprioceptor.to(device)

    def forward(self, x):
        """Forward the input image through the complete network for the purposes of training using the proprioception. Return a vector of output_size which """
        features = self.feature_extractor(x)
        # print(f"Features shape {features.shape}")
        flatfeatures = self.flatten(features)
        latent = self.reductor(flatfeatures)
        output = self.proprioceptor(latent)
        # print(output.device)
        return output

    def encode(self, x):
        """Performs an encoding of the input image, by forwarding though the encoding and first three layers."""
        features = self.feature_extractor(x)
        flatfeatures = self.flatten(features)
        latent = self.reductor(flatfeatures)
        return latent

# Multi-view CNN models
class MultiViewVGG19Model(nn.Module):
    """
    Neural network that processes multiple camera views using VGG19 encoders.

    The model processes each view separately through a VGG19 backbone,
    then concatenates the feature vectors before passing them through
    a regression head for proprioception prediction.
    """

    def __init__(self, exp, device):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.device = device

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
        self.to(device)

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

    def __init__(self, exp, device):
        super().__init__()
        self.num_views = exp.get("num_views", 2)
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        self.device = device

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
        self.to(device)

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

# FIXME: these are identical, differ only in the regression component,
# maybe can be merged together somehow.

class ResNetProprioTunedSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using a pre-trained architecture from above."""

    def __init__(self, exp, device="cpu"):
        """Create the sensormodel """
        super().__init__(exp, device)
        # self.exp = exp
        self.enc = ResNetProprioTunedRegression(exp, device)
        modelfile = pathlib.Path(exp["data_dir"],
                                exp["proprioception_mlp_model_file"])
        assert modelfile.exists()
        self.enc.load_state_dict(torch.load(modelfile))

    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor.
        Returns the z encoding in the form of a numpy array."""
        # print(f"sensor readings shape {sensor_readings.shape}")
        with torch.no_grad():
            z = self.enc.encode(sensor_readings)
        z = torch.squeeze(z)
        return z.cpu().numpy()

class VGG19ProprioTunedSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using a pre-trained VGG19 architecture from above."""

    def __init__(self, exp, device="cpu"):
        """Create the sensormodel """
        super().__init__(exp, device)
        self.enc = VGG19ProprioTunedRegression(exp, device)
        self.enc = self.enc.to(device)
        modelfile = pathlib.Path(exp["data_dir"],
                                exp["proprioception_mlp_model_file"])
        assert modelfile.exists()
        self.enc.load_state_dict(torch.load(modelfile))

    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor.
        Returns the z encoding in the form of a numpy array."""
        # print(f"sensor readings shape {sensor_readings.shape}")
        with torch.no_grad():
            z = self.enc.encode(sensor_readings)
        z = torch.squeeze(z)
        return z.cpu().numpy()

class MultiViewCNNSensorProcessing(AbstractSensorProcessing):
    """
    Sensor processing class that handles multiple camera views using CNN encoders.

    This class manages the processing of multiple camera views, maintaining
    a cache of previously seen views to ensure complete processing even when
    only one view is updated at a time.
    """

    def __init__(self, exp, device="cpu"):
        """
        Initialize the multi-view CNN sensor processing

        Args:
            exp: Experiment configuration
            device: Device to run the model on (cpu/cuda)
        """
        super().__init__(exp, device)

        # Log configuration details
        print(f"Initializing Multi-View CNN Sensor Processing:")
        print(f"  Model: {exp['model']}")
        print(f"  Number of views: {exp.get('num_views', 2)}")
        print(f"  Latent dimension: {exp['latent_size']}")

        # Create the encoder model based on configuration
        if exp['model'] == 'MultiViewVGG19Model':
            self.enc = MultiViewVGG19Model(exp, device)
        elif exp['model'] == 'MultiViewResNetModel':
            self.enc = MultiViewResNetModel(exp, device)
        else:
            raise ValueError(f"Unknown model type: {exp['model']}")

        # Load weights if model file exists
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            print(f"Loading Multi-View CNN encoder weights from {modelfile}")
            self.enc.load_state_dict(torch.load(modelfile, map_location=device))
        else:
            print(f"Warning: Model file {modelfile} does not exist. Using untrained model.")

        # Set model to evaluation mode
        self.enc.eval()

        # Initialize view cache
        self._view_cache = {}
        self._timestep_cache = {}
        self._current_timestep = None

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

    def process_multiple_files(self, file_paths, camera_ids=None):
        """
        Process multiple image files to produce a single embedding.

        Args:
            file_paths: List of paths to image files
            camera_ids: Optional list of camera identifiers corresponding to each file

        Returns:
            Embedding vector as numpy array with dimensions latent_size
        """
        # Default camera IDs if none provided
        if camera_ids is None:
            camera_ids = [f"camera_{i}" for i in range(len(file_paths))]

        # Ensure we have the right number of files
        if len(file_paths) != self.enc.num_views:
            raise ValueError(f"Expected {self.enc.num_views} view files, got {len(file_paths)}")

        # Load all images - pass None as transform to use default ToTensor transform
        views_list = []
        for file_path in file_paths:
            sensor_readings, _ = load_picturefile_to_tensor(file_path, transform=None)
            views_list.append(sensor_readings)

        # Process all views
        return self.process(views_list)

    def process_file(self, file_path, camera_id=None):
        """
        Process a single image file to produce an embedding.
        This method maintains a cache of previously seen views and combines them
        with the current view to generate a complete multi-view embedding.

        Args:
            file_path: Path to the image file
            camera_id: Camera identifier (required for proper view caching)

        Returns:
            Embedding vector as numpy array with dimensions latent_size
        """
        # Ensure we have a camera ID
        if camera_id is None:
            # Try to extract camera ID from filename (format: 00001_camera.jpg)
            try:
                camera_id = pathlib.Path(file_path).stem.split('_')[1]
                print(f"Extracted camera ID '{camera_id}' from filename")
            except (IndexError, ValueError):
                camera_id = "default_camera"
                print(f"No camera ID provided or extracted, using '{camera_id}'")

        # Load the image
        sensor_readings, _ = load_picturefile_to_tensor(file_path, transform=None)

        # Try to extract timestep from filename (format: 00001_camera.jpg)
        try:
            timestep = int(pathlib.Path(file_path).stem.split('_')[0])
            self._current_timestep = timestep

            # Initialize timestep in cache if needed
            if timestep not in self._timestep_cache:
                self._timestep_cache[timestep] = {}

            # Store this view in the timestep cache
            self._timestep_cache[timestep][camera_id] = sensor_readings

        except (ValueError, IndexError):
            # If we can't extract a timestep, just use the global cache
            pass

        # Update the global view cache with this view
        self._view_cache[camera_id] = sensor_readings

        # Prepare views for processing
        views_list = []
        required_cameras = self.enc.num_views

        # First try to get views from the current timestep cache
        if self._current_timestep is not None and self._current_timestep in self._timestep_cache:
            timestep_views = self._timestep_cache[self._current_timestep]

            # If we have all views for this timestep, use only those
            if len(timestep_views) == required_cameras:
                for camera in sorted(timestep_views.keys())[:required_cameras]:
                    views_list.append(timestep_views[camera])
                print(f"Using complete set of {required_cameras} views from timestep {self._current_timestep}")
                return self.process(views_list)

        # We don't have all views for the current timestep, so use the global cache
        # Start with the current view
        views_list = [sensor_readings]

        # Add other views from the cache, avoiding the current camera
        other_cameras = [cam for cam in sorted(self._view_cache.keys()) if cam != camera_id]

        # Add views until we reach the required number
        while len(views_list) < required_cameras and other_cameras:
            camera = other_cameras.pop(0)
            views_list.append(self._view_cache[camera])

        # If we still don't have enough views, duplicate the current view
        while len(views_list) < required_cameras:
            views_list.append(sensor_readings)

        print(f"Using {len(set(self._view_cache.keys()))} unique views (with {required_cameras - len(set(self._view_cache.keys()))} duplicated)")

        # Process the views
        return self.process(views_list)

class MultiViewVGG19SensorProcessing(MultiViewCNNSensorProcessing):
    """Convenience class for VGG19-based multi-view sensor processing"""

    def __init__(self, exp, device="cpu"):
        # Ensure the model is set to VGG19
        exp_copy = exp.copy()
        exp_copy['model'] = 'MultiViewVGG19Model'
        super().__init__(exp_copy, device)

class MultiViewResNetSensorProcessing(MultiViewCNNSensorProcessing):
    """Convenience class for ResNet-based multi-view sensor processing"""

    def __init__(self, exp, device="cpu"):
        # Ensure the model is set to ResNet
        exp_copy = exp.copy()
        exp_copy['model'] = 'MultiViewResNetModel'
        super().__init__(exp_copy, device)