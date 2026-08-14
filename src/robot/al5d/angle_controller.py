import numpy as np
from .helper import RobotHelper
from .pulse_controller import PulseController
from .constants import ANGLE_SERVO_COUNT, SERVO_COUNT, SERVO_GRIPPER
from exp_run_config import Config, Experiment

class AngleController:
    """Implements a robot controller for the AL5D robot which performs control in the terms of angles (for the joints) and distance for the gripper.
    """

    def __init__(self, exp, pulse_controller: PulseController):
        self.exp = exp
        self.pulse_controller = pulse_controller
        self.positions = np.ones(ANGLE_SERVO_COUNT) * \
            RobotHelper.pulse_to_angle(self.pulse_controller.exp, exp,
                self.pulse_controller.pulse_position_default)
        # FIXME: how do we set this?
        self.gripper_distance = 30

    def __str__(self):
        """Print the status of the robot"""
        return f"RobotAngleController positions = {self.positions} gripper={self.gripper_distance}"

    def as_dict(self):
        """Return the angles as a dictionary, for saving into a yaml file"""
        retval = {}
        for i, v in enumerate(self.positions):
            retval[i] = v.item()
        return retval

    def control_servo_angle(self, servo, angle):
        """Controls the servo through angle, by converting the angle to pulse. It sets the position assuming success. Works only for the 5 angle servos."""
        if not 0 <= servo < ANGLE_SERVO_COUNT:
            raise ValueError(f"Invalid angle-servo index: {servo}")
        pulse, _ = RobotHelper.servo_angle_to_pulse(
            self.exp, self.pulse_controller.exp, servo, angle)
        speed = self.pulse_controller.exp["CST_SPEED_DEFAULT"]
        self.pulse_controller.control_servo_pulse(servo, pulse, speed)
        self.positions[servo] = angle

    def calculate_gripper(self, distance):
        """Calculates the pulse necessary to set the gripper to a certain 
        opening distance"""
        pulse = 1000 + 15 * (100 - distance)
        return pulse

    def control_gripper(self, distance):
        """Sets the gripper to a certain opening distance [0..100]"""
        if not 0 <= distance <= 100:
            raise ValueError(f"Invalid gripper distance: {distance}")
        pulse = self.calculate_gripper(distance)
        servo = SERVO_GRIPPER
        speed = self.pulse_controller.exp["CST_SPEED_DEFAULT"]
        self.pulse_controller.control_servo_pulse(servo, pulse, speed)
        self.gripper_distance = distance

    def control_angles(self, positions, gripper_distance):
        """Controls all the angles and the gripper in one shot"""
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (ANGLE_SERVO_COUNT,) or not np.all(np.isfinite(positions)):
            raise ValueError(f"Expected {ANGLE_SERVO_COUNT} finite angle values")
        if not 0 <= gripper_distance <= 100:
            raise ValueError(f"Invalid gripper distance: {gripper_distance}")
        target_pulses = np.zeros(SERVO_COUNT, dtype=int)
        for i in range(ANGLE_SERVO_COUNT):
            target_pulses[i], _ = RobotHelper.servo_angle_to_pulse(self.exp, self.pulse_controller.exp, i, positions[i])
        target_pulses[SERVO_GRIPPER] = self.calculate_gripper(gripper_distance)
        self.pulse_controller.control_pulses(target_pulses)
        self.positions = positions
        self.gripper_distance = gripper_distance
