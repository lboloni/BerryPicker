"""
position_controller.py

A high-level position controller for the al5d robot
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"
import numpy as np
from .pulse_controller import PulseController
from .angle_controller import AngleController
from .constants import ANGLE_SERVO_COUNT, SERVO_Z, SERVO_SHOULDER, SERVO_ELBOW, SERVO_WRIST, SERVO_WRIST_ROTATION
from .position import RobotPosition
from math import sqrt, atan, acos, fabs, degrees
from copy import copy
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

#logging.basicConfig(level=logging.WARNING)

class PositionController:
    """A controller that controls the robot in terms of the physical position of the actuator. The general idea is that this captures some of the low level calculations necessary to control the robot in an intelligent way. The idea is that this had been engineered, while what comes on top of this will be learned.
    
    device = '/dev/ttyUSB0'
    """
    def __init__(self, exp: Experiment):
        self.exp = exp
        self.exp_pulse = Config().get_experiment(exp["exp_pulsecontroller"], exp["run_pulsecontroller"])
        self.exp_angle = Config().get_experiment(exp["exp_anglecontroller"], exp["run_anglecontroller"])
        self.device = self.exp_pulse["device"]
        self.pulse_controller = PulseController(self.exp_pulse)
        self.angle_controller = AngleController(self.exp_angle, self.pulse_controller)
        self.pos = RobotPosition(exp)
        self.started = False

    def start_robot(self):
        """Move the robot to its configured default position."""
        self.pulse_controller.start_robot()
        self.started = True
        try:
            self.move(self.pos)
        except Exception:
            self.started = False
            raise

    def get_position(self):
        return copy(self.pos)

    def stop_robot(self):
        print("***al5d_position_controller: Initiating the stopping of the robot")
        self.pulse_controller.stop_robot()
        self.started = False
        print("***al5d_position_controller: Robot stopped")

    @staticmethod
    def ik_shoulder_elbow_wrist(target:RobotPosition):
        """Performs the inverse kinematics necessary to the height and distance"""
        # if AL5D - a set of constants that are used in the
        A = 5.75
        B = 7.375
        # position_distance should be larger than zero
        if target["distance"] <= 0:
            raise Exception("x <= 0")
        # Get distance and check it for error
        m = sqrt((target["height"] * target["height"]) + (target["distance"] * target["distance"]))
        a1 = degrees( atan(target["height"] / target["distance"]) )
        # Get 2nd angle (radians)
        a2 = degrees( acos((A * A - B * B + m * m) / ((A * 2) * m)) )
        # Calculate elbow angle (radians)
        angle_elbow =  degrees( acos((A * A + B * B - m * m) / ((A * 2) * B)) )
        # Calculate shoulder angle (radians)
        angle_shoulder = a1 + a2
        # Check elbow/shoulder angle for error
        if (angle_elbow <= 0) or (angle_shoulder <= 0):
            raise Exception("Elbow <=0 or Shoulder <=0")
        angle_wrist = fabs(target["wrist_angle"] - angle_elbow - angle_shoulder) - 90
        # corrections compared to the system I got
        angle_elbow = 180 - int(angle_elbow) - 20         
        angle_shoulder = int(angle_shoulder)
        # It seems that this goes in the opposite direction - or the way they added it up in the calculation was incorrect and you need the elbow removed
        angle_wrist = 180 - int(angle_wrist) + 25 # zero is vertical
        return angle_shoulder, angle_elbow, angle_wrist



    def move(self, target: RobotPosition):
        """Move to the specified target position: new version with one shot commands"""
        if not self.started:
            raise RuntimeError("Robot is not started")
        if not RobotPosition.limit(self.exp, target):
            raise ValueError(f"Unsafe robot target:\n{target}")
        normalpos = RobotPosition.to_normalized_vector(target, self.exp)
        logger.info(f"PositionController.move moving robot to target: {target},\n abs: {normalpos}")
        angle_z = 90 + target["heading"]
        angle_shoulder, angle_elbow, angle_wrist = self.ik_shoulder_elbow_wrist(target)
        angle_wrist_rotation = target["wrist_rotation"]        
        # safety check here
        angles = np.zeros(ANGLE_SERVO_COUNT)
        angles[SERVO_Z] = angle_z
        angles[SERVO_SHOULDER] = angle_shoulder
        angles[SERVO_ELBOW] = angle_elbow
        angles[SERVO_WRIST] = angle_wrist
        angles[SERVO_WRIST_ROTATION] = angle_wrist_rotation
        self.angle_controller.control_angles(angles, target["gripper"])
        self.pos = copy(target)
