import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[1] / "robot" / "widowx"),
])

from remote_control.widowx_gamepad_controller import WidowXGamepadController
from robot.widowx import SimulatedPositionController, WidowXPose
from widowx_test_support import robot_exp


def gamepad_exp():
    return {
        "button_exit": "square",
        "button_home": "home",
        "button_orientation_mode": "triangle",
        "button_release": "l1",
        "button_grasp": "r1",
        "initial_orientation_mode": "roll",
        "max_input_dt": 0.25,
        "gripper_pressure": 0.6,
        "velocity": {field: 1.0 for field in WidowXPose.FIELDS},
    }


class FakeJoystick:
    def __init__(
        self, *, lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0,
        presses=(), connected=True,
    ):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.lt = lt
        self.rt = rt
        self.presses = list(presses)
        self.connected = connected

    def check_presses(self):
        presses, self.presses = self.presses, []
        return SimpleNamespace(names=presses)


class TestWidowXGamepadController(unittest.TestCase):
    def make_controller(self, exp=None):
        robot = SimulatedPositionController(robot_exp())
        controller = WidowXGamepadController(exp or gamepad_exp(), robot)
        robot.start_robot()
        controller.synchronize(robot.get_position())
        return controller, robot

    def test_requires_synchronization_before_polling(self):
        robot = SimulatedPositionController(robot_exp())
        controller = WidowXGamepadController(gamepad_exp(), robot)
        with self.assertRaisesRegex(RuntimeError, "not synchronized"):
            controller.poll_controller(FakeJoystick(), 0.1)

    def test_maps_translation_yaw_and_roll_with_timestep_scaling(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(
            lx=0.5, ly=1.0, rx=0.25, ry=-0.25, lt=0.75, rt=0.25,
        ), 0.1)

        self.assertAlmostEqual(command.pose["x"], 0.4)
        self.assertAlmostEqual(command.pose["y"], 0.05)
        self.assertAlmostEqual(command.pose["z"], 0.175)
        self.assertAlmostEqual(command.pose["yaw"], 0.025)
        self.assertAlmostEqual(command.pose["roll"], 0.05)
        self.assertAlmostEqual(command.pose["pitch"], 0.0)

    def test_mode_button_redirects_trigger_axis_to_pitch(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(
            lt=1.0, presses=["triangle"],
        ), 0.1)
        self.assertEqual(controller.orientation_mode, "pitch")
        self.assertAlmostEqual(command.pose["roll"], 0.0)
        self.assertAlmostEqual(command.pose["pitch"], 0.1)

    def test_trigger_difference_cannot_exceed_configured_velocity(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(lt=1.0, rt=-1.0), 0.1)
        self.assertAlmostEqual(command.pose["roll"], 0.1)

    def test_caps_large_timestep_and_saturates_pose_limit(self):
        controller, _ = self.make_controller()
        command = controller.poll_controller(FakeJoystick(ly=1.0), 2.0)
        self.assertAlmostEqual(command.pose["x"], 0.55)

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
                FakeJoystick(presses=["l1", "r1"]), 0.1
            )
        self.assertIsNone(controller.poll_controller(
            FakeJoystick(presses=["square"]), 0.1
        ))
        self.assertTrue(controller.exit_control)

    def test_rejects_duplicate_buttons_and_incomplete_velocity(self):
        exp = gamepad_exp()
        exp["button_grasp"] = exp["button_release"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            WidowXGamepadController(
                exp, SimulatedPositionController(robot_exp())
            )
        exp = gamepad_exp()
        del exp["velocity"]["yaw"]
        with self.assertRaisesRegex(ValueError, "every pose field"):
            WidowXGamepadController(
                exp, SimulatedPositionController(robot_exp())
            )


if __name__ == "__main__":
    unittest.main()
