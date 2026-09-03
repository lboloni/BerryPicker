import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.widowx import WidowXCommand, WidowXPose
from robot.widowx.move import move_pose_towards, move_towards
from widowx_test_support import robot_exp


class TestWidowXPosition(unittest.TestCase):
    def test_vector_and_normalized_vector_round_trip(self):
        pose = WidowXPose(robot_exp())
        restored = WidowXPose.from_vector(robot_exp(), pose.to_vector())
        normalized = pose.to_normalized_vector(robot_exp())
        restored_normalized = WidowXPose.from_normalized_vector(
            robot_exp(), normalized
        )
        np.testing.assert_allclose(restored.to_vector(), pose.to_vector(), atol=1e-7)
        np.testing.assert_allclose(
            restored_normalized.to_vector(), pose.to_vector(), atol=1e-7
        )

    def test_rejects_missing_nonfinite_and_out_of_range_values(self):
        values = WidowXPose(robot_exp()).as_dict()
        del values["yaw"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            WidowXPose(robot_exp(), values)
        for value in (math.nan, 2.0):
            values = WidowXPose(robot_exp()).as_dict()
            values["x"] = value
            with self.assertRaises(ValueError):
                WidowXPose(robot_exp(), values)

    def test_angular_movement_uses_the_shortest_path(self):
        self.assertAlmostEqual(move_towards(3.1, -3.1, 0.02, angular=True), 3.12)
        current = WidowXPose(robot_exp())
        target = WidowXPose(robot_exp())
        target["yaw"] = 0.2
        moved = move_pose_towards(
            robot_exp(), current, target,
            {field: 0.05 for field in WidowXPose.FIELDS},
        )
        self.assertAlmostEqual(moved["yaw"], 0.05)

    def test_command_copies_pose_and_validates_gripper(self):
        pose = WidowXPose(robot_exp())
        command = WidowXCommand(pose, "grasp", 0.7)
        pose["x"] = 0.4
        self.assertEqual(command.pose["x"], 0.3)
        with self.assertRaises(ValueError):
            WidowXCommand(pose, "close")
        with self.assertRaises(ValueError):
            WidowXCommand(pose, gripper_pressure=1.1)


if __name__ == "__main__":
    unittest.main()
