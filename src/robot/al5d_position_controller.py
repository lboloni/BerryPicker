"""
al5d_position_controller.py

A high-level position controller for the al5d robot
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"
import numpy as np
from .al5d_helper import RobotHelper
from .al5d_pulse_controller import PulseController
from .al5d_angle_controller import AngleController
from math import sqrt, atan, acos, fabs, degrees
from copy import copy
import logging

logging.basicConfig(level=logging.WARNING)

class RobotPosition:
    """A data class describing the high level robot position. 
    The functions returning things here are heavily dependent on an exp of the position_controller type. 
    """

    FIELDS = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]

    def __init__(self, exp: Experiment, values:dict = None):
        # writing this list here to ensure that we have it in the right order
        # this should be used for iteration, not the values\
        assert exp["robot_name"]=="al5d" and exp["controller_type"]=="position_controller"
        if values is None:
            print(exp["POS_DEFAULT"])
            self.values = copy(exp["POS_DEFAULT"])
        else:
            self.values = copy(values)
    
    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    @staticmethod
    def limit(exp: Experiment, posc):
        """Verifies whether the given position is safe, defined between the mind and the max"""
        retval = True
        for fld in RobotPosition.FIELDS:
            retval = retval and posc.values[fld] <= exp["POS_MAX"][fld]
            retval = retval and posc.values[fld] >= exp["POS_MIN"][fld]
        return retval

    def to_normalized_vector(self, exp: Experiment):
        """Converts the positions from dictionary to a normalized vector"""
        retval = np.zeros(6, dtype = np.float32)
        for i, fld in enumerate(RobotPosition.FIELDS):
            retval[i] = RobotHelper.map_ranges(self.values[fld], exp["POS_MIN"][fld], exp["POS_MAX"][fld])
        return retval

    @staticmethod
    def from_normalized_vector(exp: Experiment, values):
        """Creates the rp from a normalized numpy vector"""
        rp = RobotPosition(exp)
        for i, fld in enumerate(RobotPosition.FIELDS):
            rp.values[fld] = RobotHelper.map_ranges(values[i], 0.0, 1.0, exp["POS_MIN"][fld], exp["POS_MAX"][fld])
        return rp

    @staticmethod
    def from_vector(exp: Experiment, values):
        """Creates a RobotPosition from a numpy vector"""
        rp = RobotPosition(exp)
        for i, fld in enumerate(RobotPosition.FIELDS):
            rp.values[fld] = values[i]
        return rp

    def empirical_distance(self, exp: Experiment, other):
        """A weighted distance function between two robot positions"""
        w = np.ones([6]) / 6.0
        norm1 = np.array(self.to_normalized_vector(exp))
        norm2 = np.array(other.to_normalized_vector(exp))
        val = np.inner(w, np.abs(norm1 - norm2))
        return val    

    def __str__(self):
        retval = "Position: \n"
        for fld in RobotPosition.FIELDS:
            v = self.values[fld]            
            retval += f" {fld}:{v:.2f}\n"
        return retval

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
        self.pulse_controller.start_robot()
        self.angle_controller = AngleController(self.exp_angle, self.pulse_controller)
        self.pos = RobotPosition(exp)
        self.move(self.pos)

    def get_position(self):
        return copy(self.pos)

    def stop_robot(self):
        print("***al5d_position_controller: Initiating the stopping of the robot")
        self.pulse_controller.stop_robot()
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
        normalpos = RobotPosition.to_normalized_vector(target, self.exp)
        print(f"PositionController.move moving robot to target: {target},\n abs: {normalpos}")
        angle_z = 90 + target["heading"]
        angle_shoulder, angle_elbow, angle_wrist = self.ik_shoulder_elbow_wrist(target)
        angle_wrist_rotation = target["wrist_rotation"]        
        # safety check here
        angles = np.zeros(5)
        angles[self.exp["SERVO_ELBOW"]] = angle_elbow
        angles[self.exp["SERVO_SHOULDER"]] = angle_shoulder
        angles[self.exp["SERVO_WRIST"]] = angle_wrist
        angles[self.exp["SERVO_WRIST_ROTATION"]] = angle_wrist_rotation
        angles[self.exp["SERVO_Z"]] = angle_z
        self.angle_controller.control_angles(angles, target["gripper"])
        self.pos = target

