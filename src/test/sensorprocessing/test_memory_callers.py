"""Temporal sequence boundaries and cache policy in existing callers."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from exp_run_config import Config
from visual_proprioception import visproprio_helper as vp
from behavior_cloning import bc_trainingdata as bc
from robot_controller.rcco_sp_cnn import RCCO_SP_CNN
from robot_controller.rcco_sp_vae import RCCO_SP_VAE
from robot_controller.abstract_rcco import AbstractRCComponent


class TemporalProcessor:
    temporal = True

    def __init__(self):
        self.resets = 0
        self.value = 0
        self.outputs = []

    def reset_context(self):
        self.resets += 1
        self.value = 0

    def process(self, image, *, dt):
        self.value += dt
        self.outputs.append(self.value)
        return np.array([self.value], dtype=np.float32)

    def process_demonstration(self, demo, timestep, cameras, transform, *, dt):
        return self.process(None, dt=dt)


class Demo:
    metadata = {"maxsteps": 3}

    def __init__(self, exp, name):
        self.name = name

    def get_image(self, *args, **kwargs):
        return torch.ones(1, 1), None

    def get_action(self, *args):
        return self

    def to_normalized_vector(self, exp):
        return np.array([1.], dtype=np.float32)


class TestMemoryCallers(unittest.TestCase):
    def test_vp_loaders_reset_and_ignore_legacy_caches(self):
        for loader in (vp.load_demonstrations_as_proprioception_training,
                       vp.load_multiview_demonstrations_as_proprioception_training):
            with self.subTest(loader=loader.__name__), tempfile.TemporaryDirectory() as directory:
                input_path, target_path = Path(directory) / "inputs.pt", Path(directory) / "targets.pt"
                torch.save(torch.tensor([-99.]), input_path)
                torch.save(torch.tensor([-99.]), target_path)
                sp = TemporalProcessor()
                exp = {"training_data": [["run", "a", ["left", "right"]],
                                         ["run", "b", ["left", "right"]]]}
                with (patch.object(vp.Config, "get_experiment", return_value={}),
                      patch.object(vp, "Demonstration", Demo),
                      patch.object(vp.sp_helper, "get_transform_to_sp", return_value=None)):
                    result = loader(sp, exp, {"num_views": 2}, {}, "training_data",
                                    input_path, target_path, timestep_interval=lambda demo, i: i + 1.)
                self.assertEqual(sp.resets, 2)
                torch.testing.assert_close(result["inputs"].flatten(), torch.tensor([1., 3., 6., 1., 3., 6.]))
                torch.testing.assert_close(torch.load(input_path, weights_only=True), torch.tensor([-99.]))

    def test_temporal_multiview_failure_is_not_skipped(self):
        sp = TemporalProcessor()
        sp.process_demonstration = Mock(side_effect=RuntimeError("frame failed"))
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(vp.Config, "get_experiment", return_value={}),
                  patch.object(vp, "Demonstration", Demo),
                  patch.object(vp.sp_helper, "get_transform_to_sp", return_value=None)):
                with self.assertRaisesRegex(RuntimeError, "frame failed"):
                    vp.load_multiview_demonstrations_as_proprioception_training(
                        sp, {"training_data": [["run", "a", ["left", "right"]]]},
                        {"num_views": 2}, {}, "training_data",
                        Path(directory) / "input.pt", Path(directory) / "target.pt",
                        timestep_interval=lambda demo, i: 1.,
                    )

    def test_bc_loader_resets_and_preserves_framewise_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(Config().runtime, {"device": "cpu"}):
            class Exp(dict):
                def data_dir(self):
                    return Path(directory)

                def start_timer(self, name):
                    pass

                def end_timer(self, name):
                    pass

            exp = Exp(training_data=[["run", "a", "left"], ["run", "b", "left"]], sequence_length=0)
            input_path = Path(directory) / "training_input.pth"
            torch.save(torch.tensor([-99.]), input_path)
            sp = TemporalProcessor()
            with (patch.object(bc, "is_temporal_sp", return_value=True),
                  patch.object(bc, "create_sp", return_value=sp),
                  patch.object(bc.Config, "get_experiment", return_value={}),
                  patch.object(bc, "Demonstration", Demo),
                  patch.object(bc, "get_transform_to_sp", return_value=None)):
                bc.create_trainingdata_bc(exp, {}, {}, timestep_interval=lambda demo, i: i + 1.)
            self.assertEqual(sp.resets, 2)
            self.assertEqual(sp.outputs, [1., 3., 1., 3.])
            torch.testing.assert_close(torch.load(input_path, weights_only=True), torch.tensor([-99.]))

    def test_controller_reset_reaches_sp(self):
        for cls in (RCCO_SP_CNN, RCCO_SP_VAE):
            component = cls.__new__(cls)
            AbstractRCComponent.__init__(component, {})
            component.sp = Mock()
            component.dirty = True
            component.reset_context()
            component.sp.reset_context.assert_called_once_with()
            self.assertFalse(component.dirty)


if __name__ == "__main__":
    unittest.main()
