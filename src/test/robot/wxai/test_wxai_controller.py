"""The real-arm PositionController on the kinematic fake driver."""

import pathlib
import sys
import unittest

import trossen_arm

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.wxai import PositionController, WXAICommand, WXAIPose
from robot.wxai.fake_driver import FakeTrossenArmDriver
from wxai_test_support import robot_exp


class TestWXAIPositionController(unittest.TestCase):
    def setUp(self):
        self.exp = robot_exp()
        self.driver = FakeTrossenArmDriver(self.exp)
        self.controller = PositionController(self.exp, driver=self.driver)

    def test_start_configures_the_driver_and_holds(self):
        self.controller.start_robot()
        self.assertEqual(self.driver.calls[0], ("configure", self.exp["ip_address"], False))
        self.assertEqual(self.driver.modes, {
            "arm": trossen_arm.Mode.position, "gripper": trossen_arm.Mode.external_effort})
        self.assertAlmostEqual(self.controller.get_target()["x"], 0.2537, places=3)

    def test_requires_start(self):
        with self.assertRaisesRegex(RuntimeError, "not started"):
            self.controller.get_position()

    def test_move_reaches_the_target(self):
        self.controller.start_robot()
        target = WXAIPose(self.exp)
        self.controller.move(WXAICommand(target), moving_time=0.5, blocking=False)
        self.assertEqual(self.driver.calls[-1], ("set_cartesian_positions", 0.5, False))
        reached = self.controller.get_position()
        for field in ("x", "y", "z"):
            self.assertAlmostEqual(reached[field], target[field], delta=self.exp["ik_position_tolerance"])
        self.assertEqual(self.controller.get_target().as_dict(), target.as_dict())

    def test_unreachable_target_is_rejected_without_motion(self):
        self.controller.start_robot()
        far = WXAIPose(self.exp, dict(self.exp["POSE_DEFAULT"], x=0.8, z=0.8))
        self.assertFalse(self.controller.can_reach(far))
        calls = len(self.driver.calls)
        with self.assertRaisesRegex(ValueError, "not reachable"):
            self.controller.move(far)
        self.assertEqual(len(self.driver.calls), calls)

    def test_can_reach_before_start_uses_the_sleep_joints(self):
        self.assertTrue(self.controller.can_reach(WXAIPose(self.exp)))

    def test_grasp_and_release_use_scaled_external_effort(self):
        self.controller.start_robot()
        pose = self.controller.get_position()
        self.controller.move(WXAICommand(pose, "grasp", 0.25))
        self.assertEqual(self.driver.calls[-1],
                         ("set_gripper_external_effort", -0.25 * self.exp["gripper_max_effort"]))
        self.controller.move(WXAICommand(pose, "release"))
        self.assertEqual(self.driver.calls[-1], ("set_gripper_external_effort",
                         self.exp["gripper_pressure"] * self.exp["gripper_max_effort"]))
        self.assertEqual(self.controller.get_state()["gripper_action"], "release")

    def test_joint_command_moves_all_joints_in_position_mode(self):
        self.controller.start_robot()
        joints = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.02]
        self.controller.move(WXAICommand(self.controller.get_target(), joints=joints))
        self.assertEqual(self.driver.modes["gripper"], trossen_arm.Mode.position)
        self.assertEqual(self.controller.get_state()["joint_positions"], joints)
        self.controller.move(WXAICommand(self.controller.get_target(), "grasp"))
        self.assertEqual(self.driver.modes["gripper"], trossen_arm.Mode.external_effort)

    def test_home_sleep_and_stop(self):
        self.controller.start_robot()
        self.controller.go_home()
        self.assertEqual(self.driver.get_arm_positions(), self.exp["HOME_JOINTS"])
        self.controller.stop_robot()
        self.assertEqual(self.driver.get_arm_positions(), self.exp["SLEEP_JOINTS"])
        self.assertEqual(self.driver.modes["arm"], trossen_arm.Mode.idle)
        self.assertEqual(self.driver.calls[-1], ("cleanup",))
        self.assertFalse(self.controller.started)

    def test_state_has_seven_joints_and_efforts(self):
        self.controller.start_robot()
        state = self.controller.get_state()
        self.assertEqual(len(state["joint_positions"]), 7)
        self.assertEqual(len(state["joint_efforts"]), 7)
        self.assertEqual(set(state["pose"]), set(WXAIPose.FIELDS))


if __name__ == "__main__":
    unittest.main()
