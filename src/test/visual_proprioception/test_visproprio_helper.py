import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from visual_proprioception import visproprio_helper


class TestVisproprioHelper(unittest.TestCase):
    def test_sensor_processor_factory_uses_experiment_only(self):
        exp = {"sp_experiment": "sensorprocessing", "sp_run": "run"}
        spexp = {"class": "TestSensorProcessing"}

        with (
            patch.object(
                visproprio_helper.Config,
                "get_experiment",
                return_value=spexp,
            ) as get_experiment,
            patch.object(
                visproprio_helper.sp_factory,
                "create_sp",
                return_value=sentinel.processor,
            ) as create_sp,
        ):
            result = visproprio_helper.get_visual_proprioception_sp(exp)

        self.assertIs(result, sentinel.processor)
        get_experiment.assert_called_once_with("sensorprocessing", "run")
        create_sp.assert_called_once_with(spexp)

    def test_multiview_loader_does_not_pass_device_to_demonstration(self):
        exp = {"training_data": [["demo-run", "demo-name", ["cam-1", "cam-2"]]]}
        spexp = {"num_views": 2}
        exp_robot = object()

        class FakePosition:
            def to_normalized_vector(self, received_exp_robot):
                assert received_exp_robot is exp_robot
                return np.array([0.25, 0.75], dtype=np.float32)

        class FakeDemonstration:
            metadata = {"maxsteps": 1}

            def __init__(self, exp_demo, demo_name):
                self.exp_demo = exp_demo
                self.demo_name = demo_name

            def get_image(self, index, camera, transform):
                self.assertions = (index, camera, transform)
                return torch.tensor([[float(index)]]), None

            def get_action(self, index, action_name, received_exp_robot):
                assert index == 0
                assert action_name == "rc-position-target"
                assert received_exp_robot is exp_robot
                return FakePosition()

        class FakeSensorProcessor:
            def process(self, view_images):
                assert len(view_images) == 2
                return np.array([1.0, 2.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "inputs.pt"
            target_path = Path(directory) / "targets.pt"

            with (
                patch.object(
                    visproprio_helper.Config,
                    "get_experiment",
                    return_value={"demo": "experiment"},
                ),
                patch.object(
                    visproprio_helper,
                    "Demonstration",
                    FakeDemonstration,
                ),
                patch.object(
                    visproprio_helper.sp_helper,
                    "get_transform_to_sp",
                    return_value="transform",
                ),
            ):
                result = (
                    visproprio_helper.load_multiview_demonstrations_as_proprioception_training(
                        FakeSensorProcessor(),
                        exp,
                        spexp,
                        exp_robot,
                        "training_data",
                        input_path,
                        target_path,
                    )
                )

        self.assertEqual(result["inputs"].shape, (1, 2))
        self.assertEqual(result["targets"].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
