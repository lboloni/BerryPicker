import pathlib
import sys
import unittest

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.wxai import SimulatedPositionController, WXAICommand, WXAIPose
from wxai_test_support import robot_exp


class TestWXAISimulatedController(unittest.TestCase):
    def test_pose_and_joint_commands(self):
        exp = robot_exp("position_controller_wxai_00")
        controller = SimulatedPositionController(exp)
        controller.start_robot()
        target = WXAIPose(exp, dict(exp["POSE_DEFAULT"], x=0.45))
        controller.move(WXAICommand(target, "grasp"))
        self.assertEqual(controller.get_position().as_dict(), target.as_dict())
        self.assertIsInstance(controller.get_position(), WXAIPose)
        controller.move(WXAICommand(target, joints=[0.1] * 7))
        self.assertEqual(controller.get_state()["joint_positions"], [0.1] * 7)
        controller.update(0.1)
        controller.stop_robot()


if __name__ == "__main__":
    unittest.main()
