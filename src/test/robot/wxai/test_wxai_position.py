import pathlib
import sys
import unittest
from copy import copy

import numpy as np

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.widowx import WidowXPose
from robot.wxai import WXAICommand, WXAIPose
from robot.wxai.move import move_pose_by_clamped, move_pose_towards
from robot.wxai.position import pose_from_driver, pose_to_driver
from wxai_test_support import robot_exp


class TestWXAIPose(unittest.TestCase):
    def test_pose_requires_a_wxai_exprun(self):
        exp = robot_exp()
        self.assertEqual(WXAIPose(exp).as_dict(), exp["POSE_DEFAULT"])
        with self.assertRaisesRegex(ValueError, "position-controller"):
            WidowXPose(exp)

    def test_copy_keeps_the_class(self):
        pose = WXAIPose(robot_exp())
        self.assertIsInstance(copy(pose), WXAIPose)
        self.assertIsInstance(copy(WXAICommand(pose)), WXAICommand)

    def test_move_helpers_return_wxai_poses(self):
        exp = robot_exp()
        pose = WXAIPose(exp)
        moved = move_pose_by_clamped(exp, pose, {"x": 10.0})
        self.assertIsInstance(moved, WXAIPose)
        self.assertEqual(moved["x"], exp["POSE_MAX"]["x"])
        towards = move_pose_towards(exp, pose, moved, {field: 0.01 for field in WXAIPose.FIELDS})
        self.assertIsInstance(towards, WXAIPose)
        self.assertAlmostEqual(towards["x"], pose["x"] + 0.01)

    def test_driver_conversion_round_trip(self):
        exp = robot_exp()
        pose = WXAIPose(exp, {"x": 0.3, "y": -0.1, "z": 0.25, "roll": 0.3, "pitch": -0.4, "yaw": 1.2})
        vector = pose_to_driver(pose)
        self.assertEqual(vector.shape, (6,))
        back = pose_from_driver(exp, vector)
        for field in WXAIPose.FIELDS:
            self.assertAlmostEqual(back[field], pose[field])

    def test_identity_rotation_is_a_zero_angle_axis(self):
        np.testing.assert_allclose(pose_to_driver(WXAIPose(robot_exp()))[3:], np.zeros(3))

    def test_command_joints(self):
        pose = WXAIPose(robot_exp())
        self.assertEqual(WXAICommand(pose, joints=range(7)).as_dict()["joints"], list(map(float, range(7))))
        with self.assertRaisesRegex(ValueError, "7 finite"):
            WXAICommand(pose, joints=[0.0] * 6)
        with self.assertRaisesRegex(TypeError, "WXAIPose"):
            WXAICommand(WidowXPose(dict(robot_exp(), robot_name="widowx")))


if __name__ == "__main__":
    unittest.main()
