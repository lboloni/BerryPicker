"""
sp_aruco.py

Sensor processing using Aruco markers
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_sp

import numpy as np
import cv2


class ArucoSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using a pre-trained VGG19 architecture from above."""

    def __init__(self, exp):
        """Create the sensormodel """
        super().__init__(exp)
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.MARKER_COUNT = exp["MARKER_COUNT"]
        if self.latent_size < self.MARKER_COUNT * (8+1):
            raise Exception(f"Latent size {self.latent_size} too small for {self.MARKER_COUNT} markers!")


    def process(self, sensor_image):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor."""
        print(sensor_image.shape)
        # Convert to NumPy and rearrange dimensions
        numpy_image = sensor_image[0].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)

        self.NORMALIZER = np.tile([numpy_image.shape[0], numpy_image.shape[1]], 4)
        print(self.NORMALIZER)


        # Convert from float [0,1] to uint8 [0,255]
        numpy_image = (numpy_image * 255).astype(np.uint8)

        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
                numpy_image, self.arucoDict, 
                parameters=self.arucoParams)

        print(marker_corners)

        z = np.ones(self.latent_size) * -1.0
        if marker_ids is not None:
            for id, corners in zip(marker_ids, marker_corners):
                detection = corners[0].flatten() / self.NORMALIZER
                idn = id[0]
                z[idn * (8+1):(idn+1) * (8+1)-1] = detection
                z[(idn+1) * (8+1)-1] = 1.0 # mark the fact that it is present
        return z

