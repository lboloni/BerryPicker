import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[1] / "robot" / "widowx"),
])

from exp_run_config import Config, Experiment
from remote_control.widowx_automove_controller import WidowXAutoMoveController
from robot.widowx import SimulatedPositionController, WidowXCommand, WidowXPose
from widowx_test_support import robot_exp


def automove_exp():
    return Experiment({
        "automove_type": "random_widowx_pose",
        "random_seed": 17,
        "waypoint_count": 4,
        "max_timesteps": 100,
        "max_sampling_attempts": 100,
        "interactive_confirm": False,
        "controller_interval": 0.1,
        "robot_interval": 0.1,
        "waypoint_reached_distance": 0.001,
        "pose": {
            "x": {"random": [0.2, 0.5]},
            "y": {"random": [-0.2, 0.2]},
            "z": {"random": [0.1, 0.4]},
            "roll": {"fixed": 0.0},
            "pitch": {"choices": [-0.2, 0.2]},
            "yaw": {"fixed": 0.0},
        },
        "gripper_actions": ["hold", "grasp", "release"],
        "gripper_pressure": 0.6,
        "motion": {
            "type": "widowx_pose_velocity",
            "velocity": {field: 0.5 for field in WidowXPose.FIELDS},
        },
    })


class TestWidowXAutoMoveController(unittest.TestCase):
    def make_controller(self, exp=None):
        robot = SimulatedPositionController(robot_exp())
        with patch.object(Config, "_instance", SimpleNamespace(runtime={})):
            controller = WidowXAutoMoveController(exp or automove_exp(), robot)
        return controller, robot

    def test_seed_is_reproducible_and_waypoints_are_commands(self):
        first, _ = self.make_controller()
        second, _ = self.make_controller()
        first.generate_waypoints()
        second.generate_waypoints()
        self.assertEqual(
            [command.as_dict() for command in first.waypoints],
            [command.as_dict() for command in second.waypoints],
        )
        self.assertTrue(all(isinstance(item, WidowXCommand) for item in first.waypoints))

    def test_runtime_seed_override(self):
        robot = SimulatedPositionController(robot_exp())
        runtime = {"automove_random_seed": 29}
        with patch.object(Config, "_instance", SimpleNamespace(runtime=runtime)):
            controller = WidowXAutoMoveController(automove_exp(), robot)
        self.assertEqual(controller.random_seed, 29)

    def test_motion_is_velocity_limited_and_gripper_waits_for_arrival(self):
        controller, robot = self.make_controller()
        robot.start_robot()
        target = WidowXPose(robot_exp())
        target["x"] = 0.5
        controller.waypoints = [WidowXCommand(target, "grasp", 0.7)]
        command = controller.next_command(0.1)
        self.assertEqual(command.gripper_action, "hold")
        self.assertAlmostEqual(command.pose["x"], 0.35)
        robot.move(command)
        robot.move(WidowXCommand(target))
        command = controller.next_command(0.1)
        self.assertEqual(command.gripper_action, "grasp")
        robot.move(command)
        self.assertIsNone(controller.next_command(0.1))

    def test_raises_when_reachable_waypoint_cannot_be_sampled(self):
        controller, robot = self.make_controller()
        robot.can_reach = lambda pose: False
        controller.max_sampling_attempts = 2
        with self.assertRaisesRegex(RuntimeError, "Could not sample"):
            controller.generate_waypoints()

    def test_rejects_incomplete_pose_configuration(self):
        exp = automove_exp()
        del exp.values["pose"]["yaw"]
        with self.assertRaisesRegex(ValueError, "every WidowX pose field"):
            self.make_controller(exp)


if __name__ == "__main__":
    unittest.main()
