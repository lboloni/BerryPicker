"""
rcco_vae.py

An rcco implementing a variational autoencoder
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

class RCCO_VAE(AbstractRCComponent):
    """An rcco that wraps a convolutional variational autoencoder. The input is a picture, the outputs are the z values. 
    TODO: possibly the uncertainty values. 
    TODO: implement based on the library we used. 
    TODO: implement based on scratch"""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None

