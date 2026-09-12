"""
rcco_sp_cnn.py

An rcco implementing a sensor processing unit relying on a CNN model.
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from robot_controller.abstract_rcco import AbstractRCComponent
import sensorprocessing.sp_factory 
import torch

class RCCO_SP_CNN(AbstractRCComponent):
    """An rcco that wraps an SP with a CNN. For instance, it can be one of those that had been proprioception fine-tuned. We rely on sp_factory to create it, with its own exp/run.

    The input is a picture, the outputs are the z values.

    It instantiates the sp with the device from the Config().runtime.
    """
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None
        # We rely on sp_factory to create it, with its own exp/run.
        self.exp_sp = Config().get_experiment(self.exp["sp_experiment"], self.exp["sp_run"])
        self.sp = sensorprocessing.sp_factory.create_sp(self.exp_sp)
        self.output_sizes["z"] = self.exp_sp["latent_size"]

    def propagate(self):
        """Performs the processing of the rcco, by calling the process() call of the sp"""
        latent = self.sp.process(self.inputs["image"])
        self.outputs["z"] = torch.as_tensor(
            latent, device=Config().runtime["device"]
        )
        self.dirty = False
        return True
