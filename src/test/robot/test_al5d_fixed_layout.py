import pathlib
import sys
import unittest

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from robot.al5d_angle_controller import AngleController
from robot.al5d_constants import SERVO_COUNT
from robot.al5d_position_controller import RobotPosition
from robot.al5d_pulse_controller import PulseController


class CapturingPulseController:
    exp = {
        "CST_PULSE_MIN": 500,
        "CST_PULSE_MAX": 2500,
        "PULSE_CORRECTION": [0] * SERVO_COUNT,
    }

    def control_pulses(self, pulses):
        self.pulses = pulses


class TestAL5DFixedLayout(unittest.TestCase):
    robot_exp = {
        "robot_name": "al5d",
        "controller_type": "position_controller",
        "POS_DEFAULT": {
            "height": 5.0, "distance": 5.0, "heading": 0.0,
            "wrist_angle": -45.0, "wrist_rotation": 75.0, "gripper": 100.0,
        },
        "POS_MIN": {
            "height": 1.0, "distance": 3.0, "heading": -90.0,
            "wrist_angle": -90.0, "wrist_rotation": 60.0, "gripper": 0.0,
        },
        "POS_MAX": {
            "height": 5.0, "distance": 10.0, "heading": 90.0,
            "wrist_angle": 0.0, "wrist_rotation": 90.0, "gripper": 100.0,
        },
    }

    def test_angle_controller_uses_six_fixed_channels(self):
        controller = AngleController.__new__(AngleController)
        controller.exp = {
            "ANGLE_LIMITS": [[0, 90, 180]] * SERVO_COUNT,
            "CST_ANGLE_MIN": 0,
            "CST_ANGLE_MAX": 180,
        }
        controller.pulse_controller = CapturingPulseController()

        controller.control_angles(np.array([0, 45, 90, 135, 180]), 100)

        np.testing.assert_array_equal(
            controller.pulse_controller.pulses,
            np.array([500, 1000, 1500, 2000, 2500, 1000]),
        )

    def test_position_vectors_are_strict(self):
        with self.assertRaises(ValueError):
            RobotPosition.from_vector(self.robot_exp, range(7))
        with self.assertRaises(ValueError):
            RobotPosition.from_normalized_vector(self.robot_exp, [1.1] * 6)

    def test_pulse_arrays_are_integer_and_bounded(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = {"CST_PULSE_MIN": 500, "CST_PULSE_MAX": 2500, "TIME_DEFAULT": 50}
        with self.assertRaises(ValueError):
            controller.control_pulses([1500.5] * SERVO_COUNT)
        with self.assertRaises(ValueError):
            controller.control_pulses([3000] * SERVO_COUNT)


if __name__ == "__main__":
    unittest.main()
