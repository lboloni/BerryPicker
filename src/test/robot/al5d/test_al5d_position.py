import pathlib
import sys
import unittest
from copy import copy

import numpy as np

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import robot_exp
from robot.al5d import RobotPosition


class TestRobotPosition(unittest.TestCase):
    def test_vector_round_trips_and_distance(self):
        exp = robot_exp()
        position = RobotPosition(exp)
        vector = position.to_normalized_vector(exp)
        np.testing.assert_allclose(vector, [0.75, 4 / 7, 0.5, 0.5, 0.5, 0.5])
        reconstructed = RobotPosition.from_normalized_vector(exp, vector)
        np.testing.assert_allclose(reconstructed.to_normalized_vector(exp), vector)
        self.assertEqual(position.empirical_distance(exp, reconstructed), 0.0)

    def test_invalid_vectors_and_positions_raise(self):
        exp = robot_exp()
        for values in ([0] * 5, [np.nan] * 6, [np.inf] * 6):
            with self.assertRaises(ValueError):
                RobotPosition.from_vector(exp, values)
        unsafe = RobotPosition(exp)
        unsafe["height"] = 6.0
        with self.assertRaises(ValueError):
            unsafe.to_normalized_vector(exp)

    def test_copies_are_independent(self):
        position = RobotPosition(robot_exp())
        duplicate = copy(position)
        duplicate["gripper"] = 0.0
        self.assertEqual(position["gripper"], 50.0)


if __name__ == "__main__":
    unittest.main()
