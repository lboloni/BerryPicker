"""WidowX AI participants built by create_participants from the checked-in expruns."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[2] / "demonstration"),
    str(pathlib.Path(__file__).parents[1] / "robot" / "wxai"),
])

from demonstration_participant import (
    DemonstrationContext,
    MujocoCameraParticipant,
    WXAILeaderParticipant,
    WXAIParticipant,
    create_participants,
)
from exp_run_config import Config, Experiment
from robot.wxai import WXAICommand, WXAIPose
from wxai_test_support import load_exp, robot_exp


class FakeJoystick:
    connected = True
    lx = 0.0
    ly = 1.0
    rx = 0.0
    ry = 0.0
    lt = 0.0
    rt = 0.0
    dy = 0.0

    def __getitem__(self, name):
        return None

    @staticmethod
    def check_presses():
        return SimpleNamespace(names=[])


BINDINGS = {
    "widowx_automove": {"factory": "widowx_automove_leader", "available": True},
    "widowx_xbox_end_effector": {
        "factory": "widowx_xbox_end_effector_leader", "available": True,
        "exp": "controllers", "run": "widowx_xbox_end_effector_controller_00"},
    "fake_wxai": {"factory": "wxai_hardware", "available": True,
                  "exp": "robot_wxai", "run": "position_controller_fake_wxai_00"},
    "mujoco_wxai": {"factory": "wxai_hardware", "available": True,
                    "exp": "robot_wxai", "run": "position_controller_mujoco_wxai_00"},
    "mujoco_cameras": {"factory": "mujoco_cameras", "available": True,
                       "exp": "controllers", "run": "mujoco_cameras"},
    "fake_wxai_leader": {"factory": "wxai_leader", "available": True,
                         "exp": "controllers", "run": "wxai_leader_fake_00"},
}


def load(spec, binding):
    """The checked-in exprun of a participant, without the machine Config."""
    return Experiment(load_exp(spec.get("exp", binding.get("exp")), spec.get("run", binding.get("run"))))


def create(collection):
    """Participants of a collection recipe of data/expruns/demonstration_collector."""
    with patch("demonstration_participant._load_participant_experiment", side_effect=load), \
            patch.object(Config, "_instance", SimpleNamespace(runtime={})):
        return create_participants(load_exp("demonstration_collector", collection),
                                   {"bindings": BINDINGS})


def run(participants, ticks):
    context = DemonstrationContext()
    for participant in participants:
        participant.start(context)
    samples = []
    for _ in range(ticks):
        for participant in participants:
            participant.update(context, 0.1)
        samples.append({participant.name: participant.sample(context) for participant in participants})
    for participant in reversed(participants):
        participant.stop(context)
    return samples


class TestWXAIParticipant(unittest.TestCase):
    def test_simulated_participant_records_wxai_action_and_observed_state(self):
        exp = robot_exp("position_controller_wxai_00")
        participant = WXAIParticipant("wxai", {"command": "target"}, exp, simulated=True)
        context = DemonstrationContext()
        participant.start(context)
        target = WXAIPose(exp, dict(exp["POSE_DEFAULT"], x=0.45))
        context.commands["target"] = WXAICommand(target, "grasp", 0.6)
        participant.update(context, 0.1)
        sample = participant.sample(context)
        self.assertEqual(sample.action["wxai-command"]["pose"]["x"], 0.45)
        self.assertEqual(sample.telemetry["gripper_action"], "grasp")
        participant.stop(context)

    def test_automove_drives_the_mujoco_wxai_with_rendered_cameras(self):
        participants = create("automove_mujoco_wxai_mujoco_cameras")
        self.assertIsInstance(participants[2], MujocoCameraParticipant)
        participants[2].exp.values["saved_image_size"] = [64, 48]
        samples = run(participants, 5)
        self.assertIn("wxai-command", samples[-1]["wxai"].action)
        self.assertEqual(set(samples[-1]["cameras"].images),
                         {"mujoco_front", "mujoco_side", "mujoco_wrist"})
        self.assertEqual(samples[-1]["cameras"].images["mujoco_front"].shape, (48, 64, 3))

    def test_xbox_drives_the_wxai_controller_on_the_fake_driver(self):
        collection = load_exp("demonstration_collector", "xbox_simulated_wxai_cameras")
        collection["participants"] = [
            dict(collection["participants"][0]),
            dict(collection["participants"][1], binding="fake_wxai"),
        ]
        with patch("demonstration_participant._load_participant_experiment", side_effect=load):
            xbox, robot = create_participants(collection, {"bindings": BINDINGS})
        # forward motion is not reachable from the folded sleep pose, start at home
        robot.exp.values["startup_pose"] = "home"
        context = DemonstrationContext()
        xbox.joystick = FakeJoystick()
        robot.start(context)
        start = robot.controller.get_position()["x"]
        xbox.update(context, 0.1)
        robot.update(context, 0.1)
        self.assertGreater(robot.sample(context).action["wxai-command"]["pose"]["x"], start)
        robot.stop(context)

    def test_leader_joints_command_the_mujoco_follower(self):
        participants = create("fake_wxai_leader_mujoco_wxai_mujoco_cameras")
        self.assertIsInstance(participants[0], WXAILeaderParticipant)
        participants[2].exp.values["saved_image_size"] = [64, 48]
        participants[0].controller.driver.joints[:] = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        samples = run(participants, 20)
        self.assertEqual(samples[-1]["wxai"].action["wxai-command"]["joints"],
                         [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        follower = samples[-1]["wxai"].telemetry["joint_positions"]
        for leader_joint, follower_joint in zip([0.0, 0.5, 0.5], follower):
            self.assertAlmostEqual(leader_joint, follower_joint, delta=0.02)

    def test_leader_requires_a_wxai_target(self):
        collection = load_exp("demonstration_collector", "fake_wxai_leader_mujoco_wxai_mujoco_cameras")
        collection["participants"] = [collection["participants"][0]]
        collection["participants"][0] = dict(collection["participants"][0], target_robot="missing")
        with patch("demonstration_participant._load_participant_experiment", side_effect=load):
            with self.assertRaisesRegex(ValueError, "unknown target robot"):
                create_participants(collection, {"bindings": BINDINGS})


if __name__ == "__main__":
    unittest.main()
