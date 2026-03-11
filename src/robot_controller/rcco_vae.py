"""
rcco_vae.py

An rcco implementing a variational autoencoder
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent
import sensorprocessing.sp_factory 
import torch

class RCCO_VAE(AbstractRCComponent):
    """An rcco that wraps an SP with a convolutional variational autoencoder. The input is a picture, the outputs are the z values = the mu and logvar values from the encoder. The outputs are on the cpu."""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None
        self.outputs["mu"] = None
        self.outputs["logvar"] = None
        self.exp_sp = Config().get_experiment(self.exp["sp_experiment"], self.exp["sp_run"])
        self.sp = sensorprocessing.sp_factory.create_sp(self.exp_sp, Config().runtime["device"])

    def propagate(self):
        """The input is a picture. The output is the z encoding which is the mu value, as well as the log of the variance. Normally only the z is used."""
        # prepare the input
        input = torch.from_numpy(self.inputs["image"]).to(Config().runtime["device"])
        self.sp.process(input)
        # perform the transfer into the outputs in the expected form
        self.outputs["z"] = torch.squeeze(self.sp.mu).cpu().numpy()
        self.outputs["logvar"] = torch.squeeze(self.sp.logvar).cpu().numpy()