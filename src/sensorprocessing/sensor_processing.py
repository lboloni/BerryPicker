import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from abc import ABC, abstractmethod
import numpy as np
from torchvision import transforms
from .sp_helper import get_transform_to_sp, load_picturefile_to_tensor


class AbstractSensorProcessing(ABC):
    """The ancestor of all the classes that perform a sensor processing. We make the assumption that all these classes are configured by an experiment/run, and take in an image"""

    def __init__(self, exp):
        self.exp = exp
        self.transform = transforms.Compose([
          transforms.ToTensor(),
        ])
        self.latent_size = exp["latent_size"]

    @abstractmethod
    def process(self, sensor_image):
        """Processes the sensor_image (which is assumed to be an image) and returns the latent encoding. Returns zero here, it must be overwritten in inherited models. 
        This is intended to be used during real-time deployment"""
        return np.zeros(self.latent_size)

    def process_file(self, sensor_image_file):
        """Processes the sensor image from a file. This probably does not need to be overwritten. 
        """
        sensor_readings, _ = load_picturefile_to_tensor(sensor_image_file, self.transform)
        output = self.process(sensor_readings)
        return output


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

    def process_demonstration(self, demonstration, timestep, cameras, transform=None):
        """Encode ordered camera views from one ``Demonstration`` timestep.

        ``cameras`` must be in the same order used to train the model.  The
        demonstration object supplies the image tensors, so this works for
        demonstrations stored either as image files or as video.
        """
        if isinstance(cameras, str):
            raise TypeError("cameras must be an ordered sequence of camera IDs")

        expected_views = getattr(self, "num_views", None)
        if expected_views is None:
            expected_views = getattr(
                getattr(self, "enc", None), "num_views", self.exp.get("num_views", 1)
            )
        if len(cameras) != expected_views:
            raise ValueError(
                f"Expected {expected_views} camera views, got {len(cameras)}"
            )

        if transform is None:
            transform = get_transform_to_sp(self.exp)

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


class MultiViewSensorProcessing(
    MultiViewDemonstrationProcessing, AbstractSensorProcessing
):
    """Base class for multi-view processors built on AbstractSensorProcessing."""
