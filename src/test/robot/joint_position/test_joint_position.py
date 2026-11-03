import pathlib
import sys
import tempfile
import unittest
from copy import copy

import numpy as np
import yaml

sys.path.append(str(pathlib.Path(__file__).parents[3]))

from robot.franka import FrankaPosition
from robot.g1 import G1ArmPosition
from robot.al5d import RobotPosition
from robot.position_factory import action_key, create_position, position_fields, robot_output_size
from demonstration.demonstration import Demonstration
sys.path.append(str(pathlib.Path(__file__).parents[1] / "al5d"))
from al5d_test_support import robot_exp as al5d_exp


def joints_exp(enforce_limits=True):
    return {
        "robot_name": "franka",
        "action_key": "joint-position",
        "enforce_limits": enforce_limits,
        "JOINTS": ["b", "a"],
        "POS_DEFAULT": {"a": 0.0, "b": 1.0},
        "POS_MIN": {"a": -1.0, "b": 0.0},
        "POS_MAX": {"a": 1.0, "b": 4.0},
    }


class TestFrankaPosition(unittest.TestCase):
    def test_normalized_vector_follows_joint_order_and_round_trips(self):
        exp = joints_exp()
        position = FrankaPosition(exp, {"a": 0.5, "b": 2.0, "ignored": 9.0})
        vector = position.to_normalized_vector(exp)
        np.testing.assert_allclose(vector, [0.5, 0.75])
        back = FrankaPosition.from_normalized_vector(exp, vector)
        self.assertAlmostEqual(back["a"], 0.5)
        self.assertAlmostEqual(back["b"], 2.0)
        self.assertEqual(position.empirical_distance(exp, back), 0.0)

    def test_limits(self):
        position = FrankaPosition(joints_exp(), {"a": 2.0, "b": 2.0})
        with self.assertRaises(ValueError):
            position.to_normalized_vector(joints_exp())
        np.testing.assert_allclose(
            position.to_normalized_vector(joints_exp(enforce_limits=False)), [0.5, 1.5])

    def test_robot_name_is_checked(self):
        exp = joints_exp()
        exp["robot_name"] = "g1_arm"
        with self.assertRaises(AssertionError):
            FrankaPosition(exp)
        self.assertIsInstance(G1ArmPosition.from_vector(exp, [1.0, 0.0]), G1ArmPosition)

    def test_copies_are_independent(self):
        position = FrankaPosition(joints_exp())
        duplicate = copy(position)
        duplicate["a"] = 0.9
        self.assertEqual(position["a"], 0.0)


class TestPositionFactory(unittest.TestCase):
    def test_dispatch_on_robot_name(self):
        self.assertIsInstance(create_position(al5d_exp()), RobotPosition)
        self.assertIsInstance(create_position(joints_exp()), FrankaPosition)
        self.assertEqual(position_fields(joints_exp()), ["b", "a"])
        self.assertEqual(robot_output_size(al5d_exp()), 6)
        self.assertEqual(action_key(al5d_exp()), "rc-position-target")
        self.assertEqual(action_key(joints_exp()), "joint-position")

    def test_demonstration_get_action_returns_robot_position(self):
        class DemoExp:
            def __init__(self, path):
                self.path = path

            def data_dir(self):
                return self.path

        with tempfile.TemporaryDirectory() as directory:
            demo_dir = pathlib.Path(directory, "demo")
            demo_dir.mkdir()
            yaml.dump({"cameras": ["dev000"], "maxsteps": 1, "stored_as_images": True,
                       "stored_as_video": False}, open(demo_dir / "_metadata.yaml", "w"))
            yaml.dump([{"joint-position": {"a": 1.0, "b": 0.0}}], open(demo_dir / "_action.yaml", "w"))
            yaml.dump([{}], open(demo_dir / "_annotation.yaml", "w"))
            demo = Demonstration(DemoExp(pathlib.Path(directory)), "demo")
            exp = joints_exp()
            position = demo.get_action(0, action_key(exp), exp)
            self.assertIsInstance(position, FrankaPosition)
            np.testing.assert_allclose(position.to_normalized_vector(exp), [0.0, 1.0])
            self.assertEqual(demo.get_action(0, "joint-position"), {"a": 1.0, "b": 0.0})


if __name__ == "__main__":
    unittest.main()
