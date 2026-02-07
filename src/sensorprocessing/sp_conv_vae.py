"""
sp_conv_vae.py

Sensor processing using the encoder part of a convolutional VAE.
"""
from .sensor_processing import AbstractSensorProcessing

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from pathlib import Path
# verifying pathlib is clean
import pathlib
print(pathlib.__file__)

vaepath = Path(Config().values["conv_vae"]["code_dir"]).expanduser()
sys.path.append(str(vaepath))

# This is some weird stuff where Torch cannot find pathlib
import os
sys.modules["pathlib._local"] = pathlib 
if os.name == 'nt':
   pathlib.PosixPath = pathlib.WindowsPath
else:
   pathlib.WindowsPath = pathlib.PosixPath


import torch

# these imports are from the Conv-VAE package
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse

# from mpl_toolkits.axes_grid1 import ImageGrid

from sensorprocessing.conv_vae import get_conv_vae_config
from .sp_helper import get_transform_to_sp
from .sensor_processing import AbstractSensorProcessing

class ConvVaeSensorProcessing (AbstractSensorProcessing):
    """Sensor processing based on a pre-trained Conv-VAE"""

    def __init__(self, exp):
        """Restore a pre-trained model based on the configuration json file and the model file"""
        super().__init__(exp)
        #model_subdir = Path(exp["data_dir"], exp["model_dir"], "models", exp["model_name"], exp["model_subdir"])
        self.conv_vae_jsonfile = Path(exp.data_dir(), "config.json")
        self.resume_model_pthfile = Path(exp.data_dir(), "model.pth")
        # self.conv_vae_jsonfile = conv_vae_jsonfile
        # self.resume_model_pthfile = resume_model_pthfile
        self.vae_config = get_conv_vae_config(
            self.conv_vae_jsonfile, 
            self.resume_model_pthfile, 
            inference_only=True)
        # build model architecture
        self.model = self.vae_config.init_obj('arch', module_arch)
        self.loss_fn = getattr(module_loss, self.vae_config['loss'])
        # loading the checkpoint, have to set weights_only false here
        self.checkpoint = torch.load(self.vae_config.resume, 
                                     weights_only = False, map_location=torch.device('cpu'))
        self.state_dict = self.checkpoint['state_dict']
        self.model.load_state_dict(self.state_dict)
        # prepare model for testing
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = get_transform_to_sp(exp)


    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor. 
        Returns the z encoding in the form of a numpy array."""
        with torch.no_grad():
            self.output, self.mu, self.logvar = self.model(sensor_readings)
        mus = torch.squeeze(self.mu)
        return mus.cpu().numpy()
    

        
        