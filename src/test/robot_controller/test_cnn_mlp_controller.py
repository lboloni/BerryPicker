"""Tests for the deterministic encoder--MLP controller architecture."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from robot_controller.graph_robot_controller import GraphRobotController
from robot_controller.rcco_mlp import MLPController, RCCO_MLP
from robot_controller.training_recipe import (
    StagedEncoderMLPTrainingRecipe, create_training_recipe,
)
from sensorprocessing.conv_vae_neo import ConvVAENeo


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 3)

    def encode(self, images):
        return self.project(images.mean(dim=(-2, -1)))

    def forward(self, images):
        return self.encode(images)


class TestMLPComponent(unittest.TestCase):
    def test_hidden_activations_and_bounded_action(self):
        model = MLPController(3, [5, 4], 2, "sigmoid")
        self.assertEqual(
            [type(module) for module in model.model],
            [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Sigmoid],
        )
        action = model(torch.randn(4, 3))
        self.assertEqual(action.shape, (4, 2))
        self.assertTrue(torch.all((0 <= action) & (action <= 1)))

    def test_component_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = {
                "input_size": 3, "hidden_sizes": [4], "output_size": 2,
                "output_activation": "sigmoid", "data_dir": directory,
                "model_file": "mlp.pth",
            }
            torch.save(
                MLPController(3, [4], 2, "sigmoid").state_dict(),
                pathlib.Path(directory) / "mlp.pth",
            )
            component = RCCO_MLP(exp)
            component.set_input("z", torch.randn(1, 3))
            self.assertTrue(component.propagate())
            self.assertEqual(component.outputs["a"].shape, (1, 2))


class TestCNNMLPTrainingRecipe(unittest.TestCase):
    def setUp(self):
        Config().runtime["device"] = "cpu"
        self.temporary = tempfile.TemporaryDirectory()
        directory = self.temporary.name
        self.experiments = {
            ("robot_controller", "input"): {"rcco-type": "Input"},
            ("robot_controller", "cnn"): {
                "rcco-type": "SP_CNN", "sp_experiment": "sp", "sp_run": "cnn",
            },
            ("sp", "cnn"): {
                "class": "ProprioTunedCNNSensorProcessing",
                "model": "ResNetProprioTunedRegression",
                "image_size": [4, 4], "latent_size": 3,
                "output_size": 2, "reductor_step_1": 4,
                "proprio_step_1": 4, "proprio_step_2": 3,
                "model_file": "cnn.pth", "data_dir": directory,
            },
            ("robot_controller", "mlp"): {
                "rcco-type": "MLP", "input_size": 3, "hidden_sizes": [4],
                "output_size": 2, "output_activation": "sigmoid",
                "model_file": "mlp.pth", "data_dir": directory,
            },
            ("robot_controller", "output"): {"rcco-type": "Output", "size": 2},
            ("robot", "tiny"): {},
        }
        self.controller = {
            "name": "CNN MLP test",
            "components": {
                "image": {"run": "input"}, "cnn_encoder": {"run": "cnn"},
                "mlp": {"run": "mlp"}, "output": {"run": "output"},
            },
            "connections": [
                {"from_component": "image", "from_output": "input",
                 "to_component": "cnn_encoder", "to_input": "image"},
                {"from_component": "cnn_encoder", "from_output": "z",
                 "to_component": "mlp", "to_input": "z"},
                {"from_component": "mlp", "from_output": "a",
                 "to_component": "output", "to_input": "output"},
            ],
        }
        self.experiments[("robot_controller", "controller")] = self.controller
        torch.save(TinyCNN().state_dict(), pathlib.Path(directory) / "cnn.pth")
        dataset = TensorDataset(
            torch.rand(4, 1, 3, 4, 4), torch.rand(4, 2)
        )
        self.loaders = (
            DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def load_experiment(self, experiment, run):
        return self.experiments[(experiment, run)]

    def dataloaders(self, *_args):
        return self.loaders

    def test_recipe_exports_bundle_usable_by_graph(self):
        exp = {
            "class": "StagedEncoderMLPTrainingRecipe",
            "data_dir": self.temporary.name, "model_file": "bundle.pth",
            "controller": {"exp": "robot_controller", "run": "controller"},
            "robot": {"exp": "robot", "run": "tiny"},
            "initial_states": {
                "cnn_encoder": {"mode": "configured"},
                "mlp": {"mode": "random"},
            },
            "stages": [{
                "name": "mlp_warmup", "trainable_components": ["mlp"],
                "epochs": 1, "optimizer": "Adam",
                "learning_rates": {"mlp": 0.001},
                "monitor": "validation_mse",
            }],
            "training_data": [["demo", "train", "camera"]],
            "validation_data": [["demo", "validation", "camera"]],
            "batch_size": 2, "random_seed": 7, "keep_checkpoints": 2,
        }
        with patch(
            "robot_controller.rcco_sp_cnn._create_cnn",
            side_effect=lambda _exp: TinyCNN(),
        ):
            recipe = create_training_recipe(
                exp, experiment_loader=self.load_experiment,
                dataloader_factory=self.dataloaders,
            )
            self.assertIsInstance(recipe, StagedEncoderMLPTrainingRecipe)
            recipe.train()
            bundle = pathlib.Path(self.temporary.name) / "bundle.pth"
            controller = GraphRobotController(
                self.controller, experiment_loader=self.load_experiment,
                bundle_path=bundle,
            )
        controller.receive_input("image", torch.rand(3, 4, 4))
        action = controller.propagate()["output"]
        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(torch.all((0 <= action) & (action <= 1)))
        source_cnn = torch.load(
            pathlib.Path(self.temporary.name) / "cnn.pth", weights_only=True
        )
        exported = torch.load(
            pathlib.Path(self.temporary.name) / "bundle.pth", weights_only=True
        )["component_state_dicts"]["cnn_encoder"]
        for name in source_cnn:
            self.assertTrue(torch.equal(source_cnn[name], exported[name]))
        records = [
            json.loads(line) for line in (
                pathlib.Path(self.temporary.name) / "metrics.jsonl"
            ).read_text().splitlines()
        ]
        self.assertIn("train_mse", records[0])
        self.assertIn("validation_mse", records[0])


class TestVAEMLPTrainingRecipe(unittest.TestCase):
    """An SP_VAE encoder (as used by Conv-VAE-Neo and VAE-GAN) with an MLP."""

    def setUp(self):
        Config().runtime["device"] = "cpu"
        self.temporary = tempfile.TemporaryDirectory()
        directory = self.temporary.name
        self.experiments = {
            ("robot_controller", "input"): {"rcco-type": "Input"},
            ("robot_controller", "vae"): {
                "rcco-type": "SP_VAE", "sp_experiment": "sp", "sp_run": "neo",
            },
            ("sp", "neo"): {
                "architecture_version": 1, "image_size": [8, 8],
                "input_channels": 3, "latent_size": 2, "base_channels": 2,
                "max_channels": 4, "bottleneck_max_size": 4,
                "group_norm_groups": 1, "model_file": "vae.pth",
                "data_dir": directory,
            },
            ("robot_controller", "mlp"): {
                "rcco-type": "MLP", "input_size": 2, "hidden_sizes": [4],
                "output_size": 2, "output_activation": "sigmoid",
                "model_file": "mlp.pth", "data_dir": directory,
            },
            ("robot_controller", "output"): {"rcco-type": "Output", "size": 2},
            ("robot", "tiny"): {},
        }
        self.controller = {
            "name": "VAE MLP test",
            "components": {
                "image": {"run": "input"}, "encoder": {"run": "vae"},
                "mlp": {"run": "mlp"}, "output": {"run": "output"},
            },
            "connections": [
                {"from_component": "image", "from_output": "input",
                 "to_component": "encoder", "to_input": "image"},
                {"from_component": "encoder", "from_output": "z",
                 "to_component": "mlp", "to_input": "z"},
                {"from_component": "mlp", "from_output": "a",
                 "to_component": "output", "to_input": "output"},
            ],
        }
        self.experiments[("robot_controller", "controller")] = self.controller
        # a full VAE checkpoint, including the decoder
        self.vae = ConvVAENeo(self.experiments[("sp", "neo")])
        torch.save(self.vae.state_dict(), pathlib.Path(directory) / "vae.pth")
        dataset = TensorDataset(torch.rand(4, 1, 3, 8, 8), torch.rand(4, 2))
        self.loaders = (
            DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def load_experiment(self, experiment, run):
        return self.experiments[(experiment, run)]

    def test_recipe_exports_vae_encoder_bundle_usable_by_graph(self):
        exp = {
            "class": "StagedEncoderMLPTrainingRecipe",
            "data_dir": self.temporary.name, "model_file": "bundle.pth",
            "controller": {"exp": "robot_controller", "run": "controller"},
            "robot": {"exp": "robot", "run": "tiny"},
            "initial_states": {
                "encoder": {"mode": "configured"},
                "mlp": {"mode": "random"},
            },
            "stages": [{
                "name": "mlp_warmup", "trainable_components": ["mlp"],
                "epochs": 1, "optimizer": "Adam",
                "learning_rates": {"mlp": 0.001},
                "monitor": "validation_mse",
            }],
            "training_data": [["demo", "train", "camera"]],
            "validation_data": [["demo", "validation", "camera"]],
            "batch_size": 2, "random_seed": 7, "keep_checkpoints": 2,
        }
        recipe = create_training_recipe(
            exp, experiment_loader=self.load_experiment,
            dataloader_factory=lambda *_args: self.loaders,
        )
        self.assertIsInstance(recipe, StagedEncoderMLPTrainingRecipe)
        recipe.train()

        # the bundle holds exactly the encoder part of the VAE checkpoint
        exported = torch.load(
            pathlib.Path(self.temporary.name) / "bundle.pth", weights_only=True
        )["component_state_dicts"]["encoder"]
        source = self.vae.state_dict()
        self.assertTrue(exported)
        for name, value in exported.items():
            self.assertTrue(name.startswith(("encoder.", "fc_mu.")), name)
            self.assertTrue(torch.equal(source[name], value), name)

        # the deployed graph computes what the training model computes
        controller = GraphRobotController(
            self.controller, experiment_loader=self.load_experiment,
            bundle_path=pathlib.Path(self.temporary.name) / "bundle.pth",
        )
        image = torch.rand(3, 8, 8)
        controller.receive_input("image", image)
        action = controller.propagate()["output"]
        recipe.model.eval()
        with torch.no_grad():
            expected = recipe.model(image.reshape(1, 1, 3, 8, 8))
        self.assertTrue(torch.allclose(action, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
