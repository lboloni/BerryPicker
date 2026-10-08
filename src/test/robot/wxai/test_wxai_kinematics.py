import pathlib
import sys
import unittest

import numpy as np

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.wxai import WXAIPose
from robot.wxai.kinematics import WXAIKinematics
from wxai_test_support import robot_exp


class TestWXAIKinematics(unittest.TestCase):
    def setUp(self):
        self.exp = robot_exp()
        self.kinematics = WXAIKinematics(self.exp)

    def test_sleep_pose_is_folded_in_front_of_the_base(self):
        position, rotation = self.kinematics.forward(self.exp["SLEEP_JOINTS"])
        np.testing.assert_allclose(position, [0.2537, 0.0, 0.1635], atol=1e-3)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)

    def test_inverse_recovers_a_forward_pose(self):
        joints = np.array([0.3, 1.2, 1.0, 0.2, -0.3, 0.4])
        position, rotation = self.kinematics.forward(joints)
        solution, reached = self.kinematics.inverse(position, rotation, self.exp["HOME_JOINTS"])
        self.assertTrue(reached)
        solved_position, _ = self.kinematics.forward(solution)
        np.testing.assert_allclose(solved_position, position, atol=self.exp["ik_position_tolerance"])

    def test_default_pose_is_reachable_and_far_poses_are_not(self):
        _, reached = self.kinematics.inverse_pose(WXAIPose(self.exp), self.exp["SLEEP_JOINTS"])
        self.assertTrue(reached)
        far = WXAIPose(self.exp, dict(self.exp["POSE_DEFAULT"], x=0.8, z=0.8))
        _, reached = self.kinematics.inverse_pose(far, self.exp["SLEEP_JOINTS"])
        self.assertFalse(reached)

    def test_solutions_respect_the_joint_limits(self):
        far = WXAIPose(self.exp, dict(self.exp["POSE_DEFAULT"], x=0.8, z=0.8))
        solution, _ = self.kinematics.inverse_pose(far, self.exp["SLEEP_JOINTS"])
        self.assertTrue(np.all(solution >= self.kinematics.lower))
        self.assertTrue(np.all(solution <= self.kinematics.upper))


if __name__ == "__main__":
    unittest.main()
