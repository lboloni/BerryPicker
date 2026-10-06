"""Tests for the teacher-forced controller verification helper."""

import pathlib
import sys
import unittest

import numpy as np
import torch


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_controller.controller_verification import teacher_forcing


class Position:
    def __init__(self, value):
        self.value = value

    def to_normalized_vector(self, exp):
        return [self.value / 10] * 2


class FakeDemonstration:
    """Frame i is an image filled with i; the action at i normalizes to i/10."""

    def __init__(self, exp, demo):
        self.demo = demo
        self.metadata = {"cameras": ["dev0"], "maxsteps": 4}
        self.actions = [None] * 4

    def get_image(self, frame, camera, transform):
        return torch.full((1, 3, 8, 8), float(frame)), None

    def get_action(self, i, type, exp):
        return Position(i)


class RecordingController:
    """Outputs the mean of the last received image."""

    def __init__(self):
        self.calls = []

    def reset_context(self):
        self.calls.append("reset")

    def receive_input(self, label, value, time=None):
        self.calls.append((label, time))
        self.image = value

    def propagate(self):
        self.output = torch.full((1, 2), self.image.mean().item())

    def read_output(self, label):
        return self.output


class TestTeacherForcing(unittest.TestCase):
    def test_predictions_follow_frames_and_targets_lead_by_one(self):
        controller = RecordingController()
        predicted, target = teacher_forcing(
            controller, ["pack", "demo", "dev0"], {"image_size": [8, 8]},
            {}, 2, demonstration_factory=FakeDemonstration,
            experiment_loader=lambda experiment, run: {})

        np.testing.assert_allclose(predicted, [[0, 0], [1, 1], [2, 2]])
        np.testing.assert_allclose(target, [[0.1] * 2, [0.2] * 2, [0.3] * 2])
        self.assertEqual(controller.calls, [
            "reset", ("image_input", 0), ("image_input", 1),
            ("image_input", 2)])


class WarmingUpController(RecordingController):
    """Has no output for the first two frames, like a sliding window."""

    def propagate(self):
        super().propagate()
        if len(self.calls) <= 3:
            self.output = None


class TestTeacherForcingWarmUp(unittest.TestCase):
    def test_steps_without_output_are_nan(self):
        predicted, target = teacher_forcing(
            WarmingUpController(), ["pack", "demo", "dev0"],
            {"image_size": [8, 8]}, {}, 2,
            demonstration_factory=FakeDemonstration,
            experiment_loader=lambda experiment, run: {})
        self.assertTrue(np.isnan(predicted[:2]).all())
        np.testing.assert_allclose(predicted[2], [2, 2])
        self.assertEqual(len(target), 3)


if __name__ == "__main__":
    unittest.main()
