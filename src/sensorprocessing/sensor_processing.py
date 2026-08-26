import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from .sp_helper import SensorPreprocessor


class AbstractSensorProcessing(ABC):
    """The ancestor of all the classes that perform a sensor processing. We make the assumption that all these classes are configured by an experiment/run, and take in an image"""

    def __init__(self, exp):
        self.exp = exp
        self.preprocessor = SensorPreprocessor(exp)
        self.latent_size = exp["latent_size"]

    @abstractmethod
    def process(self, sensor_image):
        """Processes the sensor_image (which is assumed to be an image) and returns the latent encoding. Returns zero here, it must be overwritten in inherited models. 
        This is intended to be used during real-time deployment"""
        return np.zeros(self.latent_size)

    def process_file(self, sensor_image_file):
        """Processes the sensor image from a file. This probably does not need to be overwritten. 
        """
        return self.process(self.preprocessor.from_file(sensor_image_file))


class MultiViewDemonstrationProcessing:
    """Demonstration-backed access for processors that encode camera views.

    Image and video access belongs to :class:`Demonstration`.  Multi-view
    processors receive the resulting tensors through :meth:`process`, or use
    :meth:`process_demonstration` when encoding one demonstration timestep.
    """

    # ``AbstractSensorProcessing.process_file`` is meaningful only for a
    # single image.  A multi-view processor must receive a complete, ordered
    # camera set, so it deliberately has no file-oriented API.
    process_file = None

    def _expected_view_count(self):
        expected_views = getattr(self, "num_views", None)
        if expected_views is None:
            expected_views = getattr(
                getattr(self, "enc", None), "num_views", self.exp.get("num_views", 1)
            )
        return expected_views

    def _validate_view_count(self, views, description):
        if isinstance(views, str):
            raise TypeError(f"{description} must be an ordered sequence")
        expected_views = self._expected_view_count()
        if len(views) != expected_views:
            raise ValueError(
                f"Expected {expected_views} {description}, got {len(views)}"
            )

    def _warn_on_camera_order(self, cameras):
        """Warn when the requested camera order differs from the trained one.

        Multi-view models are trained with an ordered camera list (stored in
        ``exp["cameras"]``); feeding the views in another order silently
        degrades the latent.
        """
        trained_order = getattr(self, "cameras", None)
        if trained_order and list(cameras) != list(trained_order):
            print(
                f"WARNING: camera order {list(cameras)} differs from the order "
                f"this model was trained with {list(trained_order)}"
            )

    def process_demonstration(self, demonstration, timestep, cameras, transform=None):
        """Encode ordered camera views from one ``Demonstration`` timestep.

        ``cameras`` must be in the same order used to train the model.  The
        demonstration object supplies the image tensors, so this works for
        demonstrations stored either as image files or as video.
        """
        self._validate_view_count(cameras, "camera views")
        self._warn_on_camera_order(cameras)

        if transform is None:
            transform = self.preprocessor.transform

        views = []
        for camera in cameras:
            sensor_readings, _ = demonstration.get_image(
                timestep, camera=camera, transform=transform
            )
            if sensor_readings is None:
                raise ValueError(
                    f"Could not load timestep {timestep} from camera {camera}"
                )
            views.append(sensor_readings)
        return self.process(views)

    def process_captures(self, captures):
        """Encode an ordered sequence of RGB camera frames.

        Each frame is preprocessed with this processor's shared inference
        preprocessor before it is passed to the multi-view model.
        """
        self._validate_view_count(captures, "camera captures")
        views = [
            self.preprocessor.from_capture(capture) for capture in captures
        ]
        return self.process(views)


class MultiViewSensorProcessing(
    MultiViewDemonstrationProcessing, AbstractSensorProcessing
):
    """Base class for multi-view processors built on AbstractSensorProcessing."""


class EncoderSensorProcessing:
    """Shared inference and checkpoint lifecycle for tensor encoders.

    Subclasses create ``self.enc`` and optionally set ``encoder_method`` when
    their encoder uses a name other than ``encode``.
    """

    encoder_method = "encode"

    def load_encoder_checkpoint(self, *, required=False, label="encoder"):
        """Load the configured encoder state dictionary and enter eval mode."""
        checkpoint_path = Path(
            self.exp["data_dir"], self.exp["proprioception_mlp_model_file"]
        )
        if not checkpoint_path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required {label} model file does not exist: {checkpoint_path}"
                )
            print(
                f"Warning: {label} model file {checkpoint_path} does not exist. "
                "Using untrained model."
            )
            self.enc.eval()
            return None

        print(f"Loading {label} weights from {checkpoint_path}")
        state_dict = torch.load(
            checkpoint_path, map_location=Config().runtime["device"]
        )
        # Full training checkpoints wrap the weights in "model_state_dict".
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.enc.load_state_dict(state_dict)
        self.enc.eval()
        return checkpoint_path

    def process(self, sensor_readings):
        """Encode sensor tensors and return a squeezed NumPy representation."""
        self.enc.eval()
        with torch.no_grad():
            encode = getattr(self.enc, self.encoder_method)
            encoding = encode(sensor_readings)
        return torch.squeeze(encoding).cpu().numpy()


class SingleViewEncoderSensorProcessing(
    EncoderSensorProcessing, AbstractSensorProcessing
):
    """Base class for single-view processors backed by a tensor encoder."""


class MultiViewEncoderSensorProcessing(
    EncoderSensorProcessing, MultiViewSensorProcessing
):
    """Base class for multi-view processors backed by a tensor encoder."""
