import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import numpy as np
from torchvision import transforms
from .sp_helper import load_picturefile_to_tensor


class AbstractSensorProcessing:
    """The ancestor of all the classes that perform a sensor processing. We make the assumption that all these classes are configured by an experiment/run, and take in an image"""

    def __init__(self, exp):
        self.exp = exp
        self.transform = transforms.Compose([
          transforms.ToTensor(),
        ])
        self.latent_size = exp["latent_size"]

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


