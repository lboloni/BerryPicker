import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sensorprocessing import helper_training_data


class TestHelperTrainingData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.exp = {
            "data_dir": str(self.data_dir),
            "proprioception_input_file": "inputs.pt",
            "proprioception_target_file": "targets.pt",
            "training_data": [["demo-run", "demo-name", "camera-1"]],
            "image_size": [8, 8],
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_complete_cache_without_accessing_demonstrations(self):
        inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        targets = torch.arange(5, dtype=torch.float32).reshape(5, 1)
        torch.save(inputs, self.data_dir / "inputs.pt")
        torch.save(targets, self.data_dir / "targets.pt")

        with patch.object(
            helper_training_data.Config,
            "get_experiment",
            side_effect=AssertionError(
                "complete cache should avoid demonstration loading"
            ),
        ):
            result = helper_training_data.load_images_as_proprioception_training(
                self.exp, {}, generator=torch.Generator().manual_seed(7)
            )

        self.assertTrue(torch.equal(result["inputs"], inputs))
        self.assertTrue(torch.equal(result["targets"], targets))
        self.assertEqual(len(result["inputs_training"]), 3)
        self.assertEqual(len(result["inputs_validation"]), 2)
        self.assertEqual(
            len(result["inputs_training"]) + len(result["inputs_validation"]),
            5,
        )

    def test_rebuilds_both_tensors_when_cache_is_incomplete(self):
        torch.save(torch.full((2, 1), -1.0), self.data_dir / "inputs.pt")
        robot_exp = {"robot": "configuration"}

        class FakePosition:
            def __init__(self, index):
                self.index = index

            def to_normalized_vector(self, received_robot_exp):
                assert received_robot_exp is robot_exp
                return np.array(
                    [self.index, self.index + 0.5], dtype=np.float32
                )

        class FakeDemonstration:
            metadata = {"maxsteps": 4}

            def __init__(self, exp_demo, demo_name):
                assert exp_demo == {
                    "experiment": "demonstration",
                    "run": "demo-run",
                }
                assert demo_name == "demo-name"

            def get_image(self, index, transform, camera):
                assert transform == "transform"
                assert camera == "camera-1"
                return torch.tensor([[float(index)]]), None

            def get_action(self, index, action_name, received_robot_exp):
                assert action_name == "rc-position-target"
                assert received_robot_exp is robot_exp
                return FakePosition(index)

        def fake_get_experiment(_config, experiment, run):
            return {"experiment": experiment, "run": run}

        with (
            patch.object(
                helper_training_data.Config,
                "get_experiment",
                new=fake_get_experiment,
            ),
            patch.object(
                helper_training_data,
                "get_transform_to_sp",
                return_value="transform",
            ),
            patch.object(
                helper_training_data,
                "Demonstration",
                FakeDemonstration,
            ),
        ):
            result = helper_training_data.load_images_as_proprioception_training(
                self.exp,
                robot_exp,
                generator=torch.Generator().manual_seed(2),
            )

        self.assertEqual(result["inputs"].shape, (4, 1))
        self.assertEqual(result["targets"].shape, (4, 2))
        self.assertEqual(len(result["inputs_training"]), 2)
        self.assertEqual(len(result["inputs_validation"]), 2)
        for input_split, target_split in (
            (result["inputs_training"], result["targets_training"]),
            (result["inputs_validation"], result["targets_validation"]),
        ):
            self.assertTrue(
                torch.equal(input_split.flatten(), target_split[:, 0])
            )
        self.assertTrue(
            torch.equal(
                torch.load(self.data_dir / "inputs.pt", weights_only=True),
                result["inputs"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.load(self.data_dir / "targets.pt", weights_only=True),
                result["targets"],
            )
        )

    def test_split_is_reproducible_and_preserves_every_sample(self):
        inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        targets = torch.arange(10, dtype=torch.float32).reshape(10, 1)
        torch.save(inputs, self.data_dir / "inputs.pt")
        torch.save(targets, self.data_dir / "targets.pt")

        first = helper_training_data.load_images_as_proprioception_training(
            self.exp, {}, generator=torch.Generator().manual_seed(19)
        )
        second = helper_training_data.load_images_as_proprioception_training(
            self.exp, {}, generator=torch.Generator().manual_seed(19)
        )

        self.assertTrue(
            torch.equal(first["inputs_training"], second["inputs_training"])
        )
        combined_targets = torch.cat(
            [first["targets_training"], first["targets_validation"]]
        )
        self.assertEqual(
            sorted(combined_targets.flatten().tolist()), list(range(10))
        )

    def test_rejects_mismatched_cached_sample_counts(self):
        torch.save(torch.zeros((3, 1)), self.data_dir / "inputs.pt")
        torch.save(torch.zeros((2, 1)), self.data_dir / "targets.pt")

        with self.assertRaisesRegex(ValueError, "different sample counts"):
            helper_training_data.load_images_as_proprioception_training(
                self.exp, {}
            )

    def test_rejects_invalid_training_fraction(self):
        for fraction in [0.0, 1.0, -0.1, 1.1]:
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "training_fraction"):
                    helper_training_data.load_images_as_proprioception_training(
                        self.exp, {}, training_fraction=fraction
                    )


if __name__ == "__main__":
    unittest.main()
