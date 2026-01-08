"""
rcco_vae.py

An rcco implementing a variational autoencoder
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

class RCCO_VAE(AbstractRCComponent):
    """An RCCO that implements an external input to the robot controller. Examples include camera inputs, sensors, remote control, proprioception etc.
    The input values are stored in an *output* called input (due to the way this fits in the graph)"""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None

