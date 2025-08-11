"""
al5d_simulated_position_controller.py

A simulated version of the position controller. It can be run without having access to the robot.
"""
from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"
from .al5d_position_controller import RobotPosition
from copy import copy

class SimulatedPositionController:
    """A controller that controls the robot in terms of the physical position of the actuator. The general idea is that this captures some of the low level calculations necessary to control the robot in an intelligent way. The idea is that this had been engineered, while what comes on top of this will be learned.
    
    """
    def __init__(self, exp: Experiment):
        self.exp = exp
        self.pos = RobotPosition(exp)
        self.move(self.pos)

    def get_position(self):
        return copy(self.pos)

    def stop_robot(self):
        print("stop_robot on the simulated robot")

    def move(self, target: RobotPosition):
        """Move to the specified target position: new version with one shot commands"""
        self.pos = target

