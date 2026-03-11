"""
double_demo_controller.py

Functionality to support the double control of an AL5D and WidowX, for instance for gathering data for transfer of control.
"""

import pathlib
import time
import yaml
from robot.al5d_position_controller import RobotPosition, PositionController

class DoubleDemoController():

    def __init__(self, al5d_controller: PositionController, widowx_controller, filename):
        self.al5d_controller = al5d_controller
        self.widowx_controller = widowx_controller
        self.filename = filename
        self.observations = []

    def record(self):
        print("record the state of the two robots")
        obs = {}
        # obs["time"] = time.now()
        obs["al5d_state"] = self.al5d_controller.get_position()
        # fixme add here the widowx state
        self.observations.append(obs)

    def save(self):
        with open(self.filename, "w") as file:
            yaml.dump(self.observations, file, default_flow_style=False)