import pathlib
import sys
import unittest

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import robot_exp
from robot.al5d import RobotPosition, SimulatedPositionController


class TestSimulatedPositionController(unittest.TestCase):
    def test_move_and_get_position_use_independent_copies(self):
        controller = SimulatedPositionController(robot_exp())
        target = RobotPosition(robot_exp())
        target["height"] = 3.0
        controller.move(target)
        target["height"] = 2.0
        self.assertEqual(controller.get_position()["height"], 3.0)
        reported = controller.get_position()
        reported["height"] = 1.0
        self.assertEqual(controller.get_position()["height"], 3.0)


if __name__ == "__main__":
    unittest.main()
