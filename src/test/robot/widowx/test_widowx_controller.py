import pathlib
import sys
import unittest

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.widowx import PositionController, WidowXCommand, WidowXPose, WidowXRuntime
from widowx_test_support import SLEEP_JOINTS, FakeAPI, FakeBot, robot_exp


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

    def test_reachability_ignores_the_speed_and_seeds_from_ready_before_start(self):
        controller, _, bot = self.make_controller()
        target = WidowXPose(robot_exp())
        controller.can_reach(target)
        self.assertEqual(bot.arm.pose_calls[-1]["check_moving_time"], float("inf"))
        self.assertEqual(bot.arm.moving_time, 0.1)
        self.assertEqual(bot.arm.pose_calls[-1]["custom_guess"], robot_exp()["READY_JOINTS"])

    def test_move_raises_on_ik_failure_and_delegates_gripper(self):
        controller, _, bot = self.make_controller()
        controller.start_robot()
        target = WidowXPose(robot_exp())
        controller.move(WidowXCommand(target, "grasp", 0.8))
        self.assertEqual(bot.gripper.pressures, [0.8])
        self.assertEqual(bot.gripper.actions, [("grasp", 1.0)])

        target_before_rejection = controller.get_target().as_dict()
        rejected_target = WidowXPose(robot_exp())
        rejected_target["x"] = 0.5
        bot.arm.reachable = False
        with self.assertRaisesRegex(ValueError, "could not reach"):
            controller.move(WidowXCommand(rejected_target, "release", 0.4))
        self.assertEqual(controller.get_target().as_dict(), target_before_rejection)
        self.assertEqual(bot.gripper.pressures, [0.8])
        self.assertEqual(bot.gripper.actions, [("grasp", 1.0)])

    def test_ik_uses_fresh_measured_joints_for_checks_and_moves(self):
        controller, _, bot = self.make_controller()
        controller.start_robot()
        target = WidowXPose(robot_exp())
        bot.arm.joints = [0.03, -1.83, 1.58, -0.01, 0.80, 0.006]
        controller.can_reach(target)
        self.assertEqual(bot.arm.pose_calls[-1]["custom_guess"], bot.arm.joints)
        self.assertFalse(bot.arm.pose_calls[-1]["execute"])

        bot.arm.joints = [0.04, -1.80, 1.55, -0.01, 0.79, 0.007]
        controller.move(target)
        self.assertEqual(bot.arm.pose_calls[-1]["custom_guess"], bot.arm.joints)

    def make_ready_controller(self):
        exp = robot_exp()
        exp["startup_pose"] = "ready"
        exp["shutdown_pose"] = "sleep"
        api = FakeAPI()
        bot = FakeBot()
        bot.arm.joints = list(SLEEP_JOINTS)
        return PositionController(exp, runtime=WidowXRuntime(api), bot=bot), bot

    def test_start_from_sleep_goes_to_ready_and_stop_goes_through_ready_to_sleep(self):
        controller, bot = self.make_ready_controller()
        ready = robot_exp()["READY_JOINTS"]
        controller.start_robot()
        self.assertEqual(bot.core.torque_calls, [("group", "all", True)])
        self.assertEqual(bot.core.torque_moving_times, [3.0])
        self.assertEqual(bot.arm.capture_count, 1)
        self.assertEqual(bot.arm.joints, ready)
        self.assertEqual(bot.arm.joint_calls[-1][1]["check_moving_time"], 3.0)
        self.assertEqual((bot.arm.moving_time, bot.arm.accel_time), (0.1, 0.05))

        # the start restored Goal_PWM on every listed motor before the torque on
        self.assertEqual(bot.core.register_calls, [
            ("single", "shoulder", "Goal_PWM", 885),
            ("single", "shoulder_shadow", "Goal_PWM", 885),
        ])

        bot.arm.joints = [0.1, -0.3, 1.0, 0.0, -0.6, 0.0]
        bot.core.register_calls.clear()
        controller.stop_robot()
        self.assertEqual(
            [joints for joints, _ in bot.arm.joint_calls[-2:]], [ready, SLEEP_JOINTS]
        )
        self.assertEqual(bot.core.torque_calls[-1], ("group", "all", False))
        self.assertFalse(controller.started)
        # the soft stop: a decreasing Goal_PWM down to 0 before the torque off, then back
        shoulder = [value for _, name, _, value in bot.core.register_calls if name == "shoulder"]
        self.assertEqual(len(shoulder), 22)
        self.assertEqual(shoulder[:-1], sorted(shoulder[:-1], reverse=True))
        self.assertEqual((shoulder[0], shoulder[-2], shoulder[-1]), (90, 0, 885))

    def test_sleep_goes_to_where_the_arm_rested_at_the_start(self):
        controller, bot = self.make_ready_controller()
        controller.exp["torque_off_ramp_time"] = 0
        bot.arm.joints = [-0.03, -1.91, 1.63, -0.02, 0.79, 0.0]
        controller.start_robot()
        bot.core.register_calls.clear()
        controller.stop_robot()
        # shoulder and elbow clamped into their joint limits
        self.assertEqual(bot.arm.joints, [-0.03, -1.885, 1.6057, -0.02, 0.79, 0.0])
        self.assertEqual(bot.core.register_calls, [])
        self.assertEqual(bot.core.torque_calls[-1], ("group", "all", False))

    def test_ready_start_accepts_the_arm_resting_below_the_sleep_joints(self):
        controller, bot = self.make_ready_controller()
        bot.arm.joints = [-0.03, -1.91, 1.63, -0.02, 0.79, 0.0]
        controller.start_robot()
        self.assertEqual(bot.arm.joints, robot_exp()["READY_JOINTS"])

    def test_ready_start_requires_the_sleep_pose(self):
        controller, bot = self.make_ready_controller()
        bot.arm.joints = robot_exp()["READY_JOINTS"]
        with self.assertRaisesRegex(RuntimeError, "sleep pose"):
            controller.start_robot()
        self.assertEqual(bot.core.torque_calls, [])
        self.assertFalse(controller.started)

    def test_transition_that_does_not_settle_raises(self):
        controller, bot = self.make_ready_controller()
        bot.arm.apply_joint_commands = False
        with self.assertRaisesRegex(RuntimeError, "did not settle"):
            controller.start_robot()

    def test_joint_and_cartesian_helpers_delegate(self):
        controller, _, bot = self.make_controller()
        controller.start_robot()
        controller.move_joint_positions([0.1] * 6, moving_time=0.4, blocking=False)
        self.assertEqual(bot.arm.joint_calls[-1][1], {
            "moving_time": 0.4, "blocking": False, "check_moving_time": 0.1,
        })
        controller.move_cartesian(x=0.01, moving_time=0.5)
        self.assertEqual(bot.arm.cartesian_calls[-1], {"x": 0.01, "moving_time": 0.5})


if __name__ == "__main__":
    unittest.main()
