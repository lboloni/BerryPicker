"""
rcco_lstm.py

An rcco implementing an LSTM architecture
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

class RCCO_LSTM(AbstractRCComponent):
    """An rcco that implements an LSTM based controller. The input is a latent encoding $z$, while the output is a robot control $a$"""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["z"] = None
        self.outputs["a"] = None