import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[2] / "demonstration"),
])

from demonstration_participant import (
    DemonstrationContext,
    RosImageCameraParticipant,
    create_participants,
)


class FakeController:
    def __init__(self, exp):
        self.exp = exp
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    @staticmethod
    def update():
        return False

    @staticmethod
    def get_images():
        return {"gazebo_camera_0": np.zeros((2, 2, 3), dtype=np.uint8)}

    @staticmethod
    def get_metadata():
        return {"views": {"gazebo_camera_0": {"sequence": 1}}}

    def stop(self):
        self.stopped = True


class TestRosImageCameraParticipant(unittest.TestCase):
    def test_participant_exposes_images_and_camera_metadata(self):
        context = DemonstrationContext()
        participant = RosImageCameraParticipant("camera", {}, {"views": {}})
        with patch(
            "camera.ros_image_camera_controller.RosImageCameraController",
            FakeController,
        ):
            participant.start(context)
        participant.update(context, 0.1)
        sample = participant.sample(context)

        self.assertTrue(participant.controller.started)
        self.assertIn("gazebo_camera_0", sample.images)
        self.assertEqual(
            sample.telemetry["views"]["gazebo_camera_0"]["sequence"], 1
        )
        participant.stop(context)
        self.assertTrue(participant.controller.stopped)

    def test_factory_registers_ros_image_camera_binding(self):
        collection = {"participants": [
            {"name": "camera", "binding": "gazebo_cameras"},
        ]}
        machine = {"bindings": {"gazebo_cameras": {
            "factory": "ros_image_cameras",
            "available": True,
            "exp": "controllers",
            "run": "gazebo_cameras",
        }}}
        with patch(
            "demonstration_participant._load_participant_experiment",
            return_value={"views": {}},
        ):
            participants = create_participants(collection, machine)

        self.assertEqual(len(participants), 1)
        self.assertIsInstance(participants[0], RosImageCameraParticipant)


if __name__ == "__main__":
    unittest.main()
