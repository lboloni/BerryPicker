import pathlib
import sys
import unittest

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.widowx import PositionController, WidowXCommand, WidowXPose, WidowXRuntime
from widowx_test_support import FakeAPI, FakeBot, robot_exp


class TestWidowXRuntime(unittest.TestCase):
    def test_shares_one_node_and_starts_and_stops_once(self):
        api = FakeAPI()
        runtime = WidowXRuntime(api)
        first, second = object(), object()
        first_bot = runtime.create_manipulator(first, robot_model="wx250s")
        second_bot = runtime.create_manipulator(
            second, robot_model="wx250s", robot_name="leader"
        )
        self.assertIs(first_bot.core.get_node(), second_bot.core.get_node())
        self.assertIs(api.created[1]["node"], first_bot.core.get_node())

        runtime.acquire(first)
        runtime.acquire(second)
        runtime.release(first)
        self.assertEqual(len(api.started), 1)
        self.assertEqual(api.stopped, [])
        runtime.release(second)
        self.assertEqual(len(api.stopped), 1)
        with self.assertRaisesRegex(RuntimeError, "already been shut down"):
            runtime.acquire(first)


class TestWidowXPositionController(unittest.TestCase):
    def make_controller(self):
        api = FakeAPI()
        runtime = WidowXRuntime(api)
        bot = FakeBot()
        return PositionController(robot_exp(), runtime=runtime, bot=bot), api, bot

    def test_lifecycle_and_observed_state_are_separate_from_target(self):
        controller, api, bot = self.make_controller()
        with self.assertRaises(RuntimeError):
            controller.get_position()
        controller.start_robot()
        self.assertEqual(bot.arm.capture_count, 1)
        self.assertEqual(len(api.started), 1)

        bot.arm.apply_pose_commands = False
        target = WidowXPose(robot_exp())
        target["x"] = 0.5
        controller.move(target, moving_time=0.2, blocking=False)
        self.assertEqual(controller.get_target()["x"], 0.5)
        self.assertEqual(controller.get_position()["x"], 0.3)
        self.assertEqual(bot.arm.pose_calls[-1]["moving_time"], 0.2)
        self.assertFalse(bot.arm.pose_calls[-1]["blocking"])

        controller.stop_robot()
        self.assertEqual(len(api.stopped), 1)

    def test_reachability_delegates_to_interbotix_without_execution(self):
        controller, _, bot = self.make_controller()
        target = WidowXPose(robot_exp())
        self.assertTrue(controller.can_reach(target))
        self.assertFalse(bot.arm.pose_calls[-1]["execute"])
        bot.arm.reachable = False
        self.assertFalse(controller.can_reach(target))

    def test_move_raises_on_ik_failure_and_delegates_gripper(self):
        controller, _, bot = self.make_controller()
        controller.start_robot()
        target = WidowXPose(robot_exp())
        controller.move(WidowXCommand(target, "grasp", 0.8))
        self.assertEqual(bot.gripper.pressures, [0.8])
        self.assertEqual(bot.gripper.actions, [("grasp", 1.0)])
        bot.arm.reachable = False
        with self.assertRaisesRegex(ValueError, "could not reach"):
            controller.move(target)

    def test_joint_and_cartesian_helpers_delegate(self):
        controller, _, bot = self.make_controller()
        controller.start_robot()
        controller.move_joint_positions([0.1] * 6, moving_time=0.4, blocking=False)
        self.assertEqual(bot.arm.joint_calls[-1][1], {
            "moving_time": 0.4, "blocking": False,
        })
        controller.move_cartesian(x=0.01, moving_time=0.5)
        self.assertEqual(bot.arm.cartesian_calls[-1], {"x": 0.01, "moving_time": 0.5})


if __name__ == "__main__":
    unittest.main()
