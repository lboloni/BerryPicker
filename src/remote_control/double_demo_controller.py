"""
double_demo_controller.py

Functionality to support the double control of an AL5D and WidowX, for instance for gathering data for transfer of control.
"""

import pathlib
import time
import yaml
from robot.al5d_position_controller import RobotPosition, PositionController

import numpy as np
from scipy.spatial.transform import Rotation as R

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

class DoubleDemoController():

    def __init__(self, al5d_controller: PositionController, widowx_controller, filename):
        self.al5d_controller = al5d_controller
        self.widowx_controller = widowx_controller
        self.filename = filename
        self.observations = []

    def get_ee_pose_6d(self):
        """Returns the pose of the widowx in the form of a 6-dimensional python list. This is necessary, because we don't want to operate with the 4x4 matrix returned by the get_ee_pose(), and there is no direct function for this. This can be later used by set_ee_pose_components()"""
        T = self.widowx_controller.bot.arm.get_ee_pose()
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate([pos, rot]).tolist()


    def record(self):
        print("record the state of the two robots")
        obs = {}
        # obs["time"] = time.now()
        obs["al5d_state"] = self.al5d_controller.get_position().values
        # fixme add here the widowx state
        obs["widowx_state"] = self.get_ee_pose_6d()
        self.observations.append(obs)


    def save(self):
        with open(self.filename, "w") as file:
            yaml.dump(self.observations, file, default_flow_style=False,  Dumper=NoAliasDumper)