import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[2] / "demonstration"),
    str(pathlib.Path(__file__).parents[1] / "robot" / "widowx"),
])

from demonstration_participant import (
    DemonstrationContext,
    WidowXParticipant,
    create_participants,
)
from exp_run_config import Config, Experiment
from robot.widowx import WidowXCommand, WidowXPose
from widowx_test_support import robot_exp


class FakeJoystick:
    connected = True
    lx = 0.0
    ly = 1.0
    rx = 0.0
    ry = 0.0
    lt = 0.0
    rt = 0.0

    @staticmethod
    def check_presses():
        return SimpleNamespace(names=[])


def gamepad_exp():
    return {
        "button_exit": "square",
        "button_home": "home",
        "button_orientation_mode": "triangle",
        "button_release": "l1",
        "button_grasp": "r1",
        "initial_orientation_mode": "roll",
        "max_input_dt": 0.25,
        "gripper_pressure": 0.5,
        "velocity": {field: 1.0 for field in WidowXPose.FIELDS},
    }


class TestWidowXParticipant(unittest.TestCase):
    def test_simulated_participant_records_native_action_and_observed_state(self):
        participant = WidowXParticipant(
            "widowx", {"command": "target"}, robot_exp(), simulated=True
        )
        context = DemonstrationContext()
        participant.start(context)
        target = WidowXPose(robot_exp())
        target["x"] = 0.45
        context.commands["target"] = WidowXCommand(target, "grasp", 0.6)

        participant.update(context, 0.1)
        sample = participant.sample(context)

        self.assertEqual(sample.action["widowx-command"]["pose"]["x"], 0.45)
        self.assertEqual(sample.action["widowx-command"]["gripper_action"], "grasp")
        self.assertEqual(sample.telemetry["pose"]["x"], 0.45)
        self.assertEqual(sample.telemetry["gripper_action"], "grasp")
        participant.stop(context)
        self.assertFalse(participant.controller.started)

    def test_factory_connects_native_automove_to_simulated_widowx(self):
        auto_exp = Experiment({
            "automove_type": "random_widowx_pose",
            "random_seed": 17,
            "waypoint_count": 1,
            "max_timesteps": 10,
            "max_sampling_attempts": 10,
            "interactive_confirm": False,
            "controller_interval": 0.1,
            "robot_interval": 0.1,
            "waypoint_reached_distance": 0.001,
            "pose": {
                field: {"fixed": value}
                for field, value in robot_exp()["POSE_DEFAULT"].items()
            },
            "motion": {
                "type": "widowx_pose_velocity",
                "velocity": {field: 0.5 for field in WidowXPose.FIELDS},
            },
        })
        collection = {
            "tick_interval": 0.1,
            "participants": [
                {
                    "name": "auto", "binding": "auto", "exp": "automove",
                    "run": "native", "emits": "target", "target_robot": "robot",
                },
                {"name": "robot", "binding": "robot", "command": "target"},
            ],
        }
        machine = {"bindings": {
            "auto": {"factory": "widowx_automove_leader", "available": True},
            "robot": {
                "factory": "widowx_simulated", "available": True,
                "exp": "robot_widowx", "run": "simulated",
            },
        }}

        def load(spec, binding):
            return auto_exp if spec["name"] == "auto" else robot_exp()

        with patch(
            "demonstration_participant._load_participant_experiment", side_effect=load
        ), patch.object(Config, "_instance", SimpleNamespace(runtime={})):
            auto, robot = create_participants(collection, machine)

        context = DemonstrationContext()
        auto.start(context)
        robot.start(context)
        auto.update(context, 0.1)
        robot.update(context, 0.1)
        self.assertIn("widowx-command", robot.sample(context).action)
        robot.stop(context)

    def test_factory_connects_xbox_to_simulated_widowx_without_importing_approxeng(self):
        collection = {
            "tick_interval": 0.1,
            "participants": [
                {
                    "name": "xbox", "binding": "xbox", "emits": "target",
                    "target_robot": "robot",
                },
                {"name": "robot", "binding": "robot", "command": "target"},
            ],
        }
        machine = {"bindings": {
            "xbox": {
                "factory": "widowx_xbox_leader", "available": True,
                "exp": "controllers", "run": "gamepad_widowx_00",
            },
            "robot": {
                "factory": "widowx_simulated", "available": True,
                "exp": "robot_widowx", "run": "simulated",
            },
        }}

        def load(spec, binding):
            return gamepad_exp() if spec["name"] == "xbox" else robot_exp()

        with patch(
            "demonstration_participant._load_participant_experiment", side_effect=load
        ):
            xbox, robot = create_participants(collection, machine)

        context = DemonstrationContext()
        xbox.joystick = FakeJoystick()
        robot.start(context)
        xbox.update(context, 0.1)
        robot.update(context, 0.1)

        self.assertTrue(xbox.controller.synchronized)
        self.assertAlmostEqual(robot.sample(context).action[
            "widowx-command"
        ]["pose"]["x"], 0.4)
        robot.stop(context)


if __name__ == "__main__":
    unittest.main()
