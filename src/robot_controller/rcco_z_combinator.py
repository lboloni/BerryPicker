"""
rcco_z_combinator.py

An rcco implementing a Z-combinator, a component that combines different latent encodings
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

class RCCO_Z_Combinator(AbstractRCComponent):
    """An RCCO implementing a component that combines different latent encodings."""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        # fixme: the list of inputs is specified in the exp
        self.inputs["z1"] = None
        self.outputs["z"] = None