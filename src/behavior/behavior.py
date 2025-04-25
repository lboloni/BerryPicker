import sys
sys.path.append("..")
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import numpy as np

class AbstractBehavior:
    """The ancestor of all the classes that enact a robot behavior"""

    def action(self, observations, task):
        """Implements a behavior, which based on observations and task returns an action"""
        return np.zeros(Config().values["robot"]["action_space_size"])