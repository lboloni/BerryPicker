"""Tests for the transfer experiments (src/visual_proprioception/transfer).

They use synthetic data and untrained models only; no demonstration pack is
read. The flow test generates the queue of the smoke flow in a temporary
workspace, with made-up demonstration groups.
"""

import math
import pathlib
import sys
import tempfile
import unittest

import torch
import torch.nn as nn


SOURCE_ROOT = pathlib.Path(__file__).parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from visual_proprioception.transfer.transfer_data import (
    apply_mask, color_jitter, foreground_mask, kshot_rows, masked_batch, median_background,
    physical_errors, synthetic_backgrounds)
from visual_proprioception.transfer.transfer_models import (
    SpatialSoftmax, TransferEncoder, adapt_batchnorm, fomaml_task, ridge_fit, ridge_predict)


class TestSpatialSoftmax(unittest.TestCase):
    def test_peak_gives_its_position(self):
        softmax = SpatialSoftmax(8, 8)
        softmax.log_temperature.data.fill_(math.log(0.01))
        heatmaps = torch.zeros(1, 2, 8, 8)
        heatmaps[0, 0, 0, 7] = 10.0   # top right: x = 1, y = -1
        heatmaps[0, 1, 7, 0] = 10.0   # bottom left: x = -1, y = 1
        z = softmax(heatmaps)
        self.assertEqual(tuple(z.shape), (1, 4))
        torch.testing.assert_close(z[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-3, rtol=0)


class TestMasks(unittest.TestCase):
    def test_median_background_ignores_a_moving_square(self):
        images = torch.full((5, 3, 16, 16), 100, dtype=torch.uint8)
        for i in range(5):
            images[i, :, 2 * i:2 * i + 2, 2 * i:2 * i + 2] = 255
        background = median_background(images, step=1)
        self.assertTrue((background == 100).all())

    def test_foreground_mask_is_the_dilated_square(self):
        background = torch.full((3, 16, 16), 100, dtype=torch.uint8)
        image = background.clone().unsqueeze(0)
        image[0, :, 6:8, 6:8] = 200
        mask = foreground_mask(image, background, threshold=0.1, dilate=1)
        self.assertEqual(tuple(mask.shape), (1, 1, 16, 16))
        self.assertEqual(mask.sum().item(), 16)       # the 2x2 square dilated to 4x4
        self.assertEqual(mask[0, 0, 5:9, 5:9].sum().item(), 16)

    def test_apply_mask(self):
        images = torch.rand(2, 3, 4, 4)
        torch.testing.assert_close(apply_mask(images, torch.ones(2, 1, 4, 4), 0.5), images)
        torch.testing.assert_close(apply_mask(images, torch.zeros(2, 1, 4, 4), 0.5), torch.full_like(images, 0.5))

    def test_masked_batch_with_backgrounds_and_jitter(self):
        generator = torch.Generator().manual_seed(0)
        images = torch.randint(0, 256, (3, 3, 8, 8), dtype=torch.uint8)
        masks = torch.zeros(3, 1, 8, 8, dtype=torch.uint8)
        backgrounds = synthetic_backgrounds(4, [8, 8], generator)
        x = masked_batch(images, masks, backgrounds=backgrounds, jitter=0.0, generator=generator)
        for row in x:   # the robot is masked out: every image is one of the backgrounds
            self.assertTrue(any(torch.allclose(row, b) for b in backgrounds))
        y = color_jitter(x, 0.2, generator)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(((y >= 0) & (y <= 1)).all())


class TestKShotAndUnits(unittest.TestCase):
    def test_kshot_rows(self):
        torch.testing.assert_close(kshot_rows(10, None, 0), torch.arange(10))
        torch.testing.assert_close(kshot_rows(10, 20, 0), torch.arange(10))
        a, b = kshot_rows(100, 10, 3), kshot_rows(100, 10, 3)
        torch.testing.assert_close(a, b)
        self.assertEqual(len(set(a.tolist())), 10)

    def test_physical_errors(self):
        exp_robot = {"robot_name": "g1_arm", "JOINTS": ["j1", "j2"],
                     "POS_MIN": {"j1": -1.0, "j2": 0.0}, "POS_MAX": {"j1": 1.0, "j2": math.pi}}
        errors = physical_errors(exp_robot, [0.5, 0.1])
        self.assertAlmostEqual(errors[0][0], math.degrees(1.0))
        self.assertAlmostEqual(errors[1][0], 18.0)
        self.assertEqual(errors[0][1], "deg")


class TestRidge(unittest.TestCase):
    def test_recovers_a_linear_map(self):
        torch.manual_seed(0)
        a, b = torch.randn(5, 3), torch.randn(3)
        for n in [4, 50]:   # the Woodbury form, and the normal equations
            z = torch.randn(n, 5)
            y = z @ a + b
            w = ridge_fit(z, y, math.log(1e-8))
            torch.testing.assert_close(ridge_predict(z, w), y, atol=1e-3, rtol=0)
        w = ridge_fit(torch.randn(50, 5).mul(1), torch.randn(50, 5) @ a + b, math.log(1e-8))
        self.assertEqual(tuple(w.shape), (6, 3))

    def test_gradients_reach_the_projection_and_lambda(self):
        projection = nn.Linear(8, 4)
        log_lambda = nn.Parameter(torch.tensor(0.0))
        f_s, y_s, f_q, y_q = torch.randn(5, 8), torch.randn(5, 2), torch.randn(9, 8), torch.randn(9, 2)
        w = ridge_fit(projection(f_s), y_s, log_lambda)
        loss = ((ridge_predict(projection(f_q), w) - y_q) ** 2).mean()
        loss.backward()
        self.assertIsNotNone(projection.weight.grad)
        self.assertIsNotNone(log_lambda.grad)

    def test_fomaml_task(self):
        projection = nn.Sequential(nn.Linear(8, 6), nn.LayerNorm(6), nn.ReLU(), nn.Linear(6, 4))
        task = (torch.randn(5, 8), torch.randn(5, 2), torch.randn(9, 8), torch.randn(9, 2))
        loss, grads = fomaml_task(projection, *task, inner_steps=3, inner_lr=0.01)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(grads), {name for name, _ in projection.named_parameters()})
        for name, p in projection.named_parameters():
            self.assertEqual(grads[name].shape, p.shape)


class TestAdaBN(unittest.TestCase):
    def test_running_mean_moves_to_the_target(self):
        torch.manual_seed(0)
        network = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 2))
        network.train()
        for _ in range(20):
            network(torch.randn(32, 4))
        before = network[1].running_mean.clone()
        batches = [torch.randn(64, 4) + 5.0 for _ in range(10)]
        count = adapt_batchnorm(network, list(network.modules()), network, batches)
        self.assertEqual(count, 1)
        expected = network[0](torch.cat(batches)).mean(0)
        torch.testing.assert_close(network[1].running_mean, expected.detach(), atol=1e-4, rtol=1e-4)
        self.assertFalse(torch.allclose(before, network[1].running_mean))
        self.assertFalse(network.training)
        self.assertFalse(network[2].training)


class TestTransferEncoder(unittest.TestCase):
    def make(self, backbone, latent):
        return TransferEncoder({
            "backbone": backbone, "latent_size": latent, "image_size": [256, 256],
            "pretrained_backbone": False, "head_hidden_size": 16,
            "robots": [{"name": "g1", "output_size": 4}, {"name": "panda", "output_size": 7}]})

    def test_shapes(self):
        device = Config().runtime["device"]
        images = torch.rand(2, 3, 256, 256, device=device)
        for backbone in ["vgg19", "vgg19_keypoints"]:
            model = self.make(backbone, 32)
            model.eval()
            with torch.no_grad():
                self.assertEqual(tuple(model.encode(images).shape), (2, 32))
                self.assertEqual(tuple(model(images, "g1").shape), (2, 4))
                self.assertEqual(tuple(model(images, "panda").shape), (2, 7))

    def test_only_the_projection_and_heads_are_trained(self):
        model = self.make("vgg19", 16)
        trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"projection", "heads"})
        model.train()
        self.assertFalse(model.backbone.training)


class TestTransferFlow(unittest.TestCase):
    """Builds the queue of the smoke flow in a temporary workspace"""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.old_exprun_path = Config().get_exprun_path()
        self.old_results_path = Config().get_results_path()

    def tearDown(self):
        Config().set_exprun_path(self.old_exprun_path)
        Config().values["experiment_data"] = self.old_results_path
        self.temporary_directory.cleanup()

    def test_smoke_queue(self):
        from flow import setup_flow
        from visual_proprioception.transfer.transfer_flow import DESIGNS, FAMILIES, TransferFlow
        setup_flow("transfer_test", FAMILIES, flows_path=self.temporary_directory.name)
        params = {"flow_name": "transfer_test", "datasets_run": "szenes", "packs": ["panda", "g1", "hand"],
                  "targets": ["g1"], "backbones": ["vgg19", "vit_b_16"], "designs": DESIGNS,
                  "latent_size": 32, "seeds": [0], "k_shots": [10, None], "frame_step": 25,
                  "epochs_sp": 1, "epochs_vae": 1, "meta_steps": 10, "regressor_steps": 50,
                  "regressors": ["mlp", "ridge"], "eval_group": "vp_testing", "creation_style": "exist-ok"}
        builder = TransferFlow(params)
        groups = ["sp_training", "sp_validation", "vp_training", "vp_testing"]
        builder.selections = {name: {group: [f"{group}_00000", f"{group}_00001"] for group in groups}
                              for name in params["packs"]}
        entries, compare_run = builder.build()

        notebooks = {entry["notebook"] for entry in entries}
        for notebook in notebooks:
            self.assertTrue((SOURCE_ROOT / notebook).exists(), notebook)
        names = [entry["name"] for entry in entries]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(entries[-1]["experiment"], "visual_proprioception_transfer_compare")
        # every regressor stage comes after the training of its SP
        position = {entry["run"]: i for i, entry in enumerate(entries)}
        for i, entry in enumerate(entries):
            if entry["experiment"] != "visual_proprioception_transfer":
                continue
            exp = Config().get_experiment(entry["experiment"], entry["run"], create_data_dir=False)
            sp_run = exp["sp_run"]
            if exp["sp_experiment"] == "sensorprocessing_transfer_adabn":
                sp_run = Config().get_experiment(exp["sp_experiment"], sp_run, create_data_dir=False)["base_sp_run"]
            if sp_run in position:
                self.assertLess(position[sp_run], i, entry["run"])
        methods = {Config().get_experiment("visual_proprioception_transfer", run, create_data_dir=False)["method"]
                   for run in builder.vp_runs}
        for method in ["native-vgg19", "single-vgg19-from-panda", "multi-vgg19", "randproj-vgg19",
                       "randproj-vit_b_16", "kp-multi", "mask-from-hand", "maskswap-from-panda",
                       "adabn-vit_b_16-from-panda", "adabn-vae-from-hand", "vae-target", "meta-ridge",
                       "meta-fomaml"]:
            self.assertIn(method, methods)


if __name__ == "__main__":
    unittest.main()
