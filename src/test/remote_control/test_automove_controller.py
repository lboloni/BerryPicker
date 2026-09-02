import copy
import math
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from exp_run_config import Config, Experiment
from remote_control.automove_controller import AutoMoveController
from robot.al5d import SimulatedPositionController


def robot_exp():
    return Experiment({
        "robot_name": "al5d",
        "controller_type": "position_controller",
        "POS_DEFAULT": {
            "height": 5.0, "distance": 5.0, "heading": 0.0,
            "wrist_angle": -45.0, "wrist_rotation": 75.0, "gripper": 100.0,
        },
        "POS_MIN": {
            "height": 1.0, "distance": 3.0, "heading": -90.0,
            "wrist_angle": -90.0, "wrist_rotation": 45.0, "gripper": 0.0,
        },
        "POS_MAX": {
            "height": 5.0, "distance": 10.0, "heading": 90.0,
            "wrist_angle": 0.0, "wrist_rotation": 105.0, "gripper": 100.0,
        },
    })


def base_config(automove_type):
    return {
        "automove_type": automove_type,
        "random_seed": 17,
        "waypoint_count": 4,
        "max_timesteps": 100,
        "max_sampling_attempts": 100,
        "interactive_confirm": False,
        "controller_interval": 0.1,
        "robot_interval": 0.1,
        "waypoint_reached_distance": 0.001,
    }


def robot_position_config():
    config = base_config("random_robot_position")
    config["robot_position"] = {
        "height": {"random": [2.0, 5.0]},
        "distance": {"random": [3.0, 8.0]},
        "heading": {"random": [-45.0, 45.0]},
        "wrist_angle": {"fixed": -45.0},
        "wrist_rotation": {"fixed": 75.0},
        "gripper": {"fixed": 100.0},
    }
    config["motion"] = {
        "type": "robot_position_velocity",
        "velocity": {
            "height": 1.0, "distance": 1.0, "heading": 15.0,
            "wrist_angle": 15.0, "wrist_rotation": 5.0, "gripper": 50.0,
        },
    }
    return config


def fixed_end_effector_fields():
    return {"wrist_angle": -45.0, "wrist_rotation": 75.0, "gripper": 100.0}


class TestAutoMoveController(unittest.TestCase):
    def make_controller(self, config):
        return AutoMoveController(
            Experiment(copy.deepcopy(config)),
            SimulatedPositionController(robot_exp()),
        )

    def test_robot_position_seed_is_reproducible_and_fixed_fields_do_not_move(self):
        first = self.make_controller(robot_position_config())
        second = self.make_controller(robot_position_config())
        self.assertEqual(first.random_seed, 17)
        first.generate_waypoints()
        second.generate_waypoints()
        first_values = [waypoint.values for waypoint in first.waypoints]
        second_values = [waypoint.values for waypoint in second.waypoints]
        self.assertEqual(first_values, second_values)
        for values in first_values:
            self.assertEqual(values["wrist_angle"], -45.0)
            self.assertEqual(values["wrist_rotation"], 75.0)
            self.assertEqual(values["gripper"], 100.0)

    def test_runtime_seed_overrides_experiment_seed(self):
        with patch.dict(Config().runtime, {"automove_random_seed": 29}):
            controller = self.make_controller(robot_position_config())

        self.assertEqual(controller.random_seed, 29)

    def test_rejects_invalid_runtime_seed(self):
        with patch.dict(Config().runtime, {"automove_random_seed": "29"}):
            with self.assertRaisesRegex(ValueError, "random_seed must be an integer"):
                self.make_controller(robot_position_config())

    def test_robot_position_choices_sample_only_configured_values(self):
        config = robot_position_config()
        config["waypoint_count"] = 20
        config["robot_position"]["gripper"] = {"choices": [0.0, 100.0]}
        controller = self.make_controller(config)

        controller.generate_waypoints()

        self.assertEqual(
            {waypoint["gripper"] for waypoint in controller.waypoints},
            {0.0, 100.0},
        )

    def test_rejects_empty_robot_position_choices(self):
        config = robot_position_config()
        config["robot_position"]["gripper"] = {"choices": []}
        with self.assertRaisesRegex(ValueError, "choices must be a nonempty list"):
            self.make_controller(config)

    def test_rejects_invalid_robot_position_choices(self):
        for choices, message in (([0.0, math.nan], "finite numbers"),
                                 ([0.0, 101.0], "outside AL5D limits")):
            with self.subTest(choices=choices):
                config = robot_position_config()
                config["robot_position"]["gripper"] = {"choices": choices}
                with self.assertRaisesRegex(ValueError, message):
                    self.make_controller(config)

    def test_box_waypoints_are_inside_the_box_and_use_linear_speed(self):
        config = base_config("random_ee_box")
        config["end_effector_box"] = {
            "x": [3.0, 7.0], "y": [-2.0, 2.0], "z": [4.0, 5.0],
        }
        config["end_effector_fixed_robot_position"] = fixed_end_effector_fields()
        config["motion"] = {"type": "end_effector_linear_velocity", "linear_velocity": 1.0}
        controller = self.make_controller(config)
        controller.generate_waypoints()
        for waypoint in controller.waypoints:
            self.assertTrue(3.0 <= waypoint["x"] <= 7.0)
            self.assertTrue(-2.0 <= waypoint["y"] <= 2.0)
            self.assertTrue(4.0 <= waypoint["z"] <= 5.0)
        target = controller.next_pos()
        current = controller.to_end_effector_xyz(controller.pos_current)
        next_point = controller.to_end_effector_xyz(target)
        distance = math.sqrt(sum((next_point[axis] - current[axis]) ** 2 for axis in ("x", "y", "z")))
        self.assertAlmostEqual(distance, 0.1)

    def test_plane_waypoints_keep_the_specified_height(self):
        config = base_config("random_ee_plane")
        config["end_effector_plane"] = {
            "fixed": {"z": 5.0},
            "ranges": {"x": [3.0, 7.0], "y": [-2.0, 2.0]},
        }
        config["end_effector_fixed_robot_position"] = fixed_end_effector_fields()
        config["motion"] = {"type": "end_effector_linear_velocity", "linear_velocity": 0.75}
        controller = self.make_controller(config)
        controller.generate_waypoints()
        self.assertTrue(all(waypoint["z"] == 5.0 for waypoint in controller.waypoints))
        self.assertEqual(controller.next_pos()["height"], 5.0)

    def test_rejects_removed_legacy_automove_type(self):
        config = robot_position_config()
        config["automove_type"] = "random_waypoint_6D"
        with self.assertRaisesRegex(ValueError, "Unsupported automove_type"):
            self.make_controller(config)


if __name__ == "__main__":
    unittest.main()
