import pathlib
import sys
import unittest
from copy import copy
from unittest.mock import patch

import numpy as np

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import CapturingAngleController, CapturingPulseController, angle_exp, pulse_exp, robot_exp
from robot.al5d import PositionController, RobotPosition


class TestPositionController(unittest.TestCase):
    def make_controller(self):
        config = type("Config", (), {"get_experiment": lambda _, exp, run: pulse_exp() if exp == "pulse" else angle_exp()})()
        with patch("robot.al5d.position_controller.Config", return_value=config), \
             patch("robot.al5d.position_controller.PulseController", CapturingPulseController), \
             patch("robot.al5d.position_controller.AngleController", CapturingAngleController):
            return PositionController(robot_exp())

    def test_constructor_does_not_move_and_move_requires_start(self):
        controller = self.make_controller()
        self.assertEqual(controller.pulse_controller.started, 0)
        self.assertEqual(controller.angle_controller.calls, [])
        with self.assertRaises(RuntimeError):
            controller.move(controller.pos)

    def test_start_moves_default_and_failure_resets_started(self):
        controller = self.make_controller()
        controller.start_robot()
        self.assertEqual(controller.pulse_controller.started, 1)
        self.assertTrue(controller.started)
        self.assertEqual(len(controller.angle_controller.calls), 1)

        failing = self.make_controller()
        failing.angle_controller.fail = True
        with self.assertRaisesRegex(RuntimeError, "angle command failed"):
            failing.start_robot()
        self.assertFalse(failing.started)

    def test_move_transforms_target_and_keeps_an_independent_position(self):
        controller = self.make_controller()
        controller.started = True
        target = RobotPosition(robot_exp())
        controller.move(target)
        angles, gripper = controller.angle_controller.calls[0]
        expected = PositionController.ik_shoulder_elbow_wrist(target)
        np.testing.assert_array_equal(angles, [90.0, *expected, target["wrist_rotation"]])
        self.assertEqual(gripper, target["gripper"])
        target["height"] = 3.0
        self.assertEqual(controller.get_position()["height"], 4.0)

    def test_move_rejects_unsafe_and_unreachable_targets(self):
        controller = self.make_controller()
        controller.started = True
        unsafe = copy(controller.pos)
        unsafe["height"] = 6.0
        with self.assertRaises(ValueError):
            controller.move(unsafe)
        unreachable = copy(controller.pos)
        unreachable["distance"] = 0.0
        with self.assertRaises(Exception):
            PositionController.ik_shoulder_elbow_wrist(unreachable)


if __name__ == "__main__":
    unittest.main()
