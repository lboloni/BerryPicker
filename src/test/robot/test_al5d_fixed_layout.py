import pathlib
import sys
import unittest

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from robot.al5d import PositionController, RobotPosition, SimulatedPositionController
from robot.al5d.angle_controller import AngleController
from robot.al5d.constants import SERVO_COUNT
from robot.al5d.move import move_position_by, move_position_towards, move_towards
from robot.al5d.pulse_controller import PulseController


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

    def test_public_package_api(self):
        self.assertEqual(
            {PositionController.__name__, RobotPosition.__name__, SimulatedPositionController.__name__},
            {"PositionController", "RobotPosition", "SimulatedPositionController"},
        )

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

    def test_position_motion_is_copied_and_limited(self):
        current = RobotPosition(self.robot_exp)
        target = move_position_by(self.robot_exp, current, {"height": -1.0})
        self.assertEqual(target["height"], 4.0)
        self.assertEqual(current["height"], 5.0)
        with self.assertRaises(ValueError):
            move_position_by(self.robot_exp, current, {"height": 1.0})

    def test_position_motion_towards_uses_per_field_steps(self):
        current = RobotPosition(self.robot_exp)
        target = RobotPosition(self.robot_exp, {
            "height": 3.0, "distance": 7.0, "heading": 10.0,
            "wrist_angle": -60.0, "wrist_rotation": 80.0, "gripper": 50.0,
        })
        steps = dict.fromkeys(RobotPosition.FIELDS, 1.0)
        moved = move_position_towards(self.robot_exp, current, target, steps)
        self.assertEqual(moved["height"], 4.0)
        self.assertEqual(moved["distance"], 6.0)
        self.assertEqual(moved["heading"], 1.0)
        self.assertEqual(move_towards(1.0, 2.0, 0.0), 1.0)
        with self.assertRaises(ValueError):
            move_towards(1.0, 2.0, -1.0)


if __name__ == "__main__":
    unittest.main()
