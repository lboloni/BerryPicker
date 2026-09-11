import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[1] / "robot" / "widowx"),
])

from remote_control.widowx_xbox_end_effector_controller import (
    WidowXXboxEndEffectorController,
)
from robot.widowx import SimulatedPositionController, WidowXPose
from widowx_test_support import robot_exp


def xbox_end_effector_exp():
    return {
        "button_exit": "square",
        "button_home": "home",
        "button_release": "l1",
        "button_grasp": "r1",
        "max_input_dt": 0.25,
        "gripper_pressure": 0.6,
        "velocity": {field: 1.0 for field in WidowXPose.FIELDS},
    }


class FakeJoystick:
    def __init__(
        self, *, lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, dy=0.0,
        presses=(), connected=True, held=(),
    ):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.lt = lt
        self.rt = rt
        self.dy = dy
        self.presses = list(presses)
        self.connected = connected
        self.held = set(held)

    def __getitem__(self, name):
        return 0.0 if name in self.held else None

    def check_presses(self):
        presses, self.presses = self.presses, []
        return SimpleNamespace(names=presses)


class TestWidowXXboxEndEffectorController(unittest.TestCase):
    def make_controller(self, exp=None):
        robot = SimulatedPositionController(robot_exp())
        controller = WidowXXboxEndEffectorController(
            exp or xbox_end_effector_exp(), robot
        )
        robot.start_robot()
        controller.synchronize(robot.get_position())
        return controller, robot

    def test_requires_synchronization_before_polling(self):
        robot = SimulatedPositionController(robot_exp())
        controller = WidowXXboxEndEffectorController(
            xbox_end_effector_exp(), robot
        )
        with self.assertRaisesRegex(RuntimeError, "not synchronized"):
            controller.poll_controller(FakeJoystick(), 0.1)

    def test_maps_translation_and_all_rotations_without_a_mode(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(
            lx=0.5, ly=1.0, rx=0.25, ry=-0.25,
            lt=0.75, rt=0.25, dy=-1.0,
        ), 0.1)

        self.assertAlmostEqual(command.pose["x"], 0.4)
        self.assertAlmostEqual(command.pose["y"], 0.05)
        self.assertAlmostEqual(command.pose["z"], 0.175)
        self.assertAlmostEqual(command.pose["yaw"], 0.025)
        self.assertAlmostEqual(command.pose["roll"], 0.05)
        self.assertAlmostEqual(command.pose["pitch"], 0.1)
        self.assertNotIn("orientation_mode", controller.get_state())

    def test_dpad_controls_both_pitch_directions(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(dy=-1.0), 0.1)
        self.assertAlmostEqual(command.pose["pitch"], 0.1)
        command = controller.poll_controller(FakeJoystick(dy=1.0), 0.1)
        self.assertAlmostEqual(command.pose["pitch"], 0.0)

    def test_trigger_difference_cannot_exceed_configured_velocity(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(
            FakeJoystick(lt=1.0, rt=-1.0), 0.1
        )
        self.assertAlmostEqual(command.pose["roll"], 0.1)

    def test_caps_large_timestep_and_saturates_pose_limit(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(ly=1.0), 2.0)
        self.assertAlmostEqual(command.pose["x"], 0.4)

        near_limit = command.pose.__copy__()
        near_limit["x"] = 0.99
        controller.pos_target = near_limit
        command = controller.poll_controller(FakeJoystick(ly=1.0), 0.1)
        self.assertAlmostEqual(command.pose["x"], 1.0)

    def test_unreachable_candidate_is_rejected(self):
        controller, robot = self.make_controller()
        robot.can_reach = lambda pose: False
        command = controller.poll_controller(FakeJoystick(ly=1.0), 0.1)
        self.assertAlmostEqual(command.pose["x"], 0.3)
        self.assertTrue(controller.last_target_rejected)
        self.assertEqual(controller.rejected_target_count, 1)

    def test_home_and_gripper_buttons_are_one_shot_commands(self):
        controller, _ = self.make_controller()
        controller.poll_controller(FakeJoystick(ly=1.0), 0.1)
        command = controller.poll_controller(FakeJoystick(
            presses=["home", "r1"]
        ), 0.1)
        self.assertAlmostEqual(command.pose["x"], 0.3)
        self.assertEqual(command.gripper_action, "grasp")
        self.assertEqual(command.gripper_pressure, 0.6)

        command = controller.poll_controller(FakeJoystick(), 0.1)
        self.assertEqual(command.gripper_action, "hold")
        self.assertIsNone(command.gripper_pressure)

    def test_rejects_conflicting_gripper_buttons_and_handles_exit(self):
        controller, _ = self.make_controller()
        with self.assertRaisesRegex(ValueError, "simultaneously"):
            controller.poll_controller(
                FakeJoystick(presses=["l1", "r1"], held=["l1", "r1"]), 0.1
            )
        self.assertIsNone(controller.poll_controller(
            FakeJoystick(presses=["square"]), 0.1
        ))
        self.assertTrue(controller.exit_control)

    def test_sequential_gripper_presses_use_held_state_or_hold(self):
        controller, _ = self.make_controller()
        actions = [(["r1"], "grasp"), (["l1"], "release"), ([], "hold")]
        for held, expected in actions:
            command = controller.poll_controller(
                FakeJoystick(presses=["l1", "r1"], held=held), 2.0
            )
            self.assertEqual(command.gripper_action, expected)

    def test_displacement_respects_shorter_robot_movement_time(self):
        controller, robot = self.make_controller()
        robot.exp["moving_time"] = 0.05
        command = controller.poll_controller(FakeJoystick(ly=1.0), 2.0)
        self.assertAlmostEqual(command.pose["x"], 0.35)

    def test_rejects_duplicate_buttons_and_incomplete_velocity(self):
        exp = xbox_end_effector_exp()
        exp["button_grasp"] = exp["button_release"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            WidowXXboxEndEffectorController(
                exp, SimulatedPositionController(robot_exp())
            )
        exp = xbox_end_effector_exp()
        del exp["velocity"]["yaw"]
        with self.assertRaisesRegex(ValueError, "every pose field"):
            WidowXXboxEndEffectorController(
                exp, SimulatedPositionController(robot_exp())
            )

    def test_rejects_invalid_dpad_axis(self):
        controller, _ = self.make_controller()
        for dy in (float("nan"), 1.1, -1.1):
            with self.assertRaisesRegex(ValueError, "axis dy"):
                controller.poll_controller(FakeJoystick(dy=dy), 0.1)


if __name__ == "__main__":
    unittest.main()
