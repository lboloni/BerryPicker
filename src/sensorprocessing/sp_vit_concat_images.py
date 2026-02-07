# """
# Concatenated Image Vision Transformer (ViT) model for sensor processing
# """

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_sp, load_picturefile_to_tensor
import pathlib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np


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

        # Load the ViT model based on configuration
        vit_model_name = exp["vit_model"]
        vit_weights_name = exp["vit_weights"]

        # Handle the model and weights imports
        if vit_model_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = getattr(ViT_B_16_Weights, vit_weights_name)
            self.vit_model = vit_b_16(weights=weights)
            vit_output_dim = exp.get("vit_output_dim", 768)  # ViT-B has 768 hidden dim
        elif vit_model_name == "vit_l_16":
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            weights = getattr(ViT_L_16_Weights, vit_weights_name)
            self.vit_model = vit_l_16(weights=weights)
            vit_output_dim = exp.get("vit_output_dim", 1024)  # ViT-L has 1024 hidden dim
        elif vit_model_name == "vit_h_14":
            from torchvision.models import vit_h_14, ViT_H_14_Weights
            weights = getattr(ViT_H_14_Weights, vit_weights_name)
            self.vit_model = vit_h_14(weights=weights)
            vit_output_dim = exp.get("vit_output_dim", 1280)  # ViT-H has 1280 hidden dim
        else:
            raise ValueError(f"Unsupported ViT model type: {vit_model_name}")

        # Replace the head with identity to get features
        self.vit_model.heads = nn.Identity()

        # Override with config value if provided
        if "vit_output_dim" in exp:
            vit_output_dim = exp["vit_output_dim"]

        print(f"Using {vit_model_name} with output dimension {vit_output_dim}")

        # Determine projection architecture
        if "projection_hidden_dim" in exp:
            projection_hidden_dim = exp["projection_hidden_dim"]
        else:
            # Default to a reasonable size based on input dimension
            projection_hidden_dim = vit_output_dim // 2

        # Projection from ViT features to latent space
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
        print(f"Created projection network: {vit_output_dim} → {projection_hidden_dim} → {projection_hidden_dim//2} → {self.latent_size}")

        # Add proprioceptor for end-to-end training (not used for inference)
        # This maps from latent space to 6d robot position
        self.proprioceptor = nn.Sequential(
            nn.Linear(self.latent_size, exp.get("proprio_step_1", 128)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_1", 128), exp.get("proprio_step_2", 64)),
            nn.ReLU(),
            nn.Linear(exp.get("proprio_step_2", 64), self.output_size)
        )

        print(f"Created proprioceptor: {self.latent_size} → {exp.get('proprio_step_1', 128)} → {exp.get('proprio_step_2', 64)} → {self.output_size}")

        # Image normalization for pre-trained ViT
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Create a resize transform to the required vit input size
        self.image_size = exp["image_size"]
        self.resize = transforms.Resize((self.image_size, self.image_size), antialias=True)

        # Freeze the feature extractor if specified
        if exp.get("freeze_feature_extractor", False):
            # Freeze all ViT parameters
            for param in self.vit_model.parameters():
                param.requires_grad = False
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
        resized_views = []
        for view in views_list:
            if view.size(2) != self.image_size or view.size(3) != self.image_size:
                view = self.resize(view)
            resized_views.append(view)

        # Concatenate along width dimension (dim=3)
        concat_image = torch.cat(resized_views, dim=3)

        # Resize the concatenated image to fit ViT input requirements
        # The height stays the same but width is num_views*width
        if self.vit_model.__class__.__name__.startswith("VisionTransformer"):
            # For standard ViTs, we need to resize to a square
            concat_image = transforms.functional.resize(
                concat_image,
                (self.image_size, self.image_size),
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
        # Check if already normalized
        if x.min() >= 0 and x.max() <= 1:
            # Convert [0,1] to normalized range matching ImageNet stats
            return self.normalize(x)
        return x

    def forward(self, views_list):
        """Forward pass to generate latent representation and then proprioceptor (6d)
        This forward function is used during training.
        For inference, call the encode function.

        Args:
            views_list: List of image tensors from different camera views
        """
        # Get latent representation
        latent = self.encode(views_list)

        # Map to robot position prediction via proprioceptor
        output = self.proprioceptor(latent)

        return output


class ConcatImageVitSensorProcessing(AbstractSensorProcessing):
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

        # Load weights if model file exists
        modelfile = pathlib.Path(exp["data_dir"], exp["proprioception_mlp_model_file"])
        if modelfile.exists():
            print(f"Loading Concatenated Image ViT encoder weights from {modelfile}")
            self.enc.load_state_dict(torch.load(modelfile, map_location=Config().runtime["device"]))
        else:
            print(f"Warning: Model file {modelfile} does not exist. Using untrained model.")

        # Set model to evaluation mode
        self.enc.eval()

    def process(self, sensor_readings_list):
        """Process multiple sensor readings (images) to produce a single embedding.

        Args:
            sensor_readings_list: List of image tensors from different camera views

        Returns:
            Embedding vector as numpy array with dimensions batch x 128
        """
        self.enc.eval()
        with torch.no_grad():
            # Use the encode function which returns just the latent representation
            # without passing through the proprioceptor
            z = self.enc.encode(sensor_readings_list)
        z = torch.squeeze(z)
        return z.cpu().numpy()

    def process_multiple_files(self, file_paths, camera_ids=None):
        """Process multiple image files to produce a single embedding.

        Args:
            file_paths: List of paths to image files
            camera_ids: Optional list of camera identifiers corresponding to each file

        Returns:
            Embedding vector as numpy array with dimensions 128
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
        """Process a single image file to produce an embedding.
        This method maintains a cache of previously seen views and combines them
        with the current view to generate a complete multi-view embedding.

        The cache persists across timesteps, allowing processing of files where
        some views might be missing entirely.

        Args:
            file_path: Path to the image file
            camera_id: Camera identifier (required for proper view caching)

        Returns:
            Embedding vector as numpy array with dimensions 128
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

        # Load the image - pass None as transform to use default ToTensor transform
        sensor_readings, _ = load_picturefile_to_tensor(file_path, transform=None)

        # Initialize view cache if it doesn't exist
        if not hasattr(self, '_view_cache'):
            self._view_cache = {}
            self._timestep_cache = {}  # Track views by timestep
            self._current_timestep = None

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