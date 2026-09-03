import pathlib
import sys
import unittest

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.widowx import SimulatedPositionController, WidowXCommand, WidowXPose
from widowx_test_support import robot_exp


class TestSimulatedWidowXPositionController(unittest.TestCase):
    def test_enforces_lifecycle_and_copies_positions(self):
        controller = SimulatedPositionController(robot_exp())
        with self.assertRaises(RuntimeError):
            controller.get_position()
        controller.start_robot()
        target = WidowXPose(robot_exp())
        target["y"] = 0.2
        controller.move(WidowXCommand(target, "release"))
        target["y"] = 0.4
        self.assertEqual(controller.get_position()["y"], 0.2)
        self.assertEqual(controller.get_state()["gripper_action"], "release")
        controller.stop_robot()
        with self.assertRaises(RuntimeError):
            controller.stop_robot()

    def test_cartesian_trajectory_is_relative(self):
        controller = SimulatedPositionController(robot_exp())
        controller.start_robot()
        controller.move_cartesian(x=0.1, z=-0.05, yaw=0.2, moving_time=1.0)
        pose = controller.get_position()
        self.assertAlmostEqual(pose["x"], 0.4)
        self.assertAlmostEqual(pose["z"], 0.15)
        self.assertAlmostEqual(pose["yaw"], 0.2)


if __name__ == "__main__":
    unittest.main()
