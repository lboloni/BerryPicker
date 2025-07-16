import numpy as np
# from . import al5d_constants
from .al5d_helper import RobotHelper
from .al5d_pulse_controller import PulseController
from exp_run_config import Config, Experiment

class AngleController:
    """Implements a robot controller for the AL5D robot which performs control in the terms of angles (for the joints) and distance for the gripper.
    """

    def __init__(self, exp, pulse_controller: PulseController):
        self.exp = exp
        self.pulse_controller = pulse_controller
        self.positions = np.ones(self.pulse_controller.cnt_servos-1) * \
            RobotHelper.pulse_to_angle(self.pulse_controller.exp, exp,
                self.pulse_controller.pulse_position_default)
        # FIXME: how do we set this?
        self.gripper_distance = 30

    def __str__(self):
        """Print the status of the robot"""
        return f"RobotAngleController positions = {self.positions} gripper={self.gripper_distance}"

    def as_dict(self):
        """Return the angles as a dictionary, for saving"""
        retval = {}
        for i, v in enumerate(self.positions):
            retval[i] = v
        return retval

    def control_servo_angle(self, exp_angle: Experiment, exp_pulse: Experiment, servo, angle):
        """Controls the servo through angle, by converting the angle to pulse. It sets the position assuming success. Works only for the 5 angle servos."""
        speed = self.exp["CST_SPEED_DEFAULT"]
        pulse, _ = RobotHelper.servo_angle_to_pulse(exp_angle, exp_pulse, servo, angle)
        if servo < 0 or servo >= self.exp["no_servos"]:
            raise Exception(f"Invalid servo for control_servo_angle {servo}")
        self.pulse_controller.control_servo_pulse(servo, pulse, speed)
        self.positions[servo] = angle

    def calculate_gripper(self, distance):
        """Calculates the pulse necessary to set the gripper to a certain 
        opening distance"""
        pulse = 1000 + 15 * (100 - distance)
        return pulse

    def control_gripper(self, distance):
        """Sets the gripper to a certain opening distance [0..100]"""
        speed = self.exp["CST_SPEED_DEFAULT"]
        pulse = self.calculate_gripper(distance)
        self.pulse_controller.control_servo_pulse("SERVO_GRIP", pulse, speed)
        self.gripper_distance = distance

    def control_angles(self, positions, gripper_distance):
        """Controls all the angles and the gripper in one shot"""
        target_pulses = np.zeros(self.pulse_controller.cnt_servos)
        for i in range(self.pulse_controller.cnt_servos - 1):
            target_pulses[i], _ = RobotHelper.servo_angle_to_pulse(self.exp, self.pulse_controller.exp, i, positions[i])
        target_pulses[self.pulse_controller.cnt_servos-1] = self.calculate_gripper(gripper_distance)
        self.pulse_controller.control_pulses(target_pulses)
        self.positions = positions
        self.gripper_distance = gripper_distance
