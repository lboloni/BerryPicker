"""Focused tests for staged robot-controller training and recovery."""

import json
import pathlib
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from robot_controller.graph_robot_controller import GraphRobotController
from robot_controller.training_recipe import StagedRobotControllerTrainingRecipe
from robot_controller.visualize_trec import TrainingRecipeVisualizer


class TestStagedRobotControllerTrainingRecipe(unittest.TestCase):
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
            ("robot_controller", "lstm"): {
                "rcco-type": "LSTM", "architecture": "residual",
                "input_size": 2, "hidden_size": 3, "num_layers": 1,
                "sequence_length": 2, "context_mode": "sliding_window",
                "model_file": "lstm.pth", "data_dir": directory,
            },
            ("robot_controller", "mdn"): {
                "rcco-type": "MDN", "input_dim": 3, "hidden_size": 4,
                "output_dim": 2, "num_gaussians": 2,
                "action_selection": "expected_value", "model_file": "mdn.pth",
                "data_dir": directory,
            },
            ("robot_controller", "output"): {"rcco-type": "Output", "size": 2},
            ("robot", "tiny"): {},
        }
        self.controller = {
            "name": "tiny",
            "components": {
                "image": {"run": "input"},
                "vae_encoder": {"run": "vae"},
                "lstm": {"run": "lstm"},
                "mdn": {"run": "mdn"},
                "output": {"run": "output"},
            },
            "connections": [
                {"from_component": "image", "from_output": "input",
                 "to_component": "vae_encoder", "to_input": "image"},
                {"from_component": "vae_encoder", "from_output": "z",
                 "to_component": "lstm", "to_input": "z"},
                {"from_component": "lstm", "from_output": "h",
                 "to_component": "mdn", "to_input": "h"},
                {"from_component": "mdn", "from_output": "a",
                 "to_component": "output", "to_input": "output"},
            ],
        }
        self.experiments[("robot_controller", "roco")] = self.controller
        self.dataset = TensorDataset(
            torch.rand(4, 2, 3, 8, 8), torch.rand(4, 2)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def load_experiment(self, experiment, run):
        return self.experiments[(experiment, run)]

    def dataloaders(self, *_args):
        return (
            DataLoader(self.dataset, batch_size=2),
            DataLoader(self.dataset, batch_size=2),
        )

    def recipe_exp(self, epochs=1):
        return {
            "class": "StagedRobotControllerTrainingRecipe",
            "data_dir": self.temporary.name, "model_file": "bundle.pth",
            "controller": {"exp": "robot_controller", "run": "roco"},
            "robot": {"exp": "robot", "run": "tiny"},
            "initial_states": {
                "vae_encoder": {"mode": "random"},
                "lstm": {"mode": "random"}, "mdn": {"mode": "random"},
            },
            "stages": [{
                "name": "policy", "trainable_components": ["lstm", "mdn"],
                "epochs": epochs, "optimizer": "Adam",
                "learning_rates": {"lstm": 0.001, "mdn": 0.001},
                "monitor": "validation_nll",
            }],
            "training_data": [["demo", "train", "camera"]],
            "validation_data": [["demo", "validation", "camera"]],
            "batch_size": 2, "random_seed": 13, "keep_checkpoints": 2,
        }

    def make_recipe(self, exp):
        return StagedRobotControllerTrainingRecipe(
            exp, experiment_loader=self.load_experiment,
            dataloader_factory=self.dataloaders,
        )

    def test_exported_bundle_loads_in_runtime_graph(self):
        exp = self.recipe_exp()
        recipe = self.make_recipe(exp)
        recipe.train()
        self.assertEqual(recipe.status["state"], "completed")
        bundle = pathlib.Path(self.temporary.name) / "bundle.pth"
        self.assertTrue(bundle.is_file())
        manifest = json.loads(
            (pathlib.Path(self.temporary.name) / "component_manifest.json").read_text()
        )
        self.assertEqual(
            set(manifest["components"]), {"vae_encoder", "lstm", "mdn"}
        )

        controller = GraphRobotController(
            self.controller, experiment_loader=self.load_experiment,
            bundle_path=bundle,
        )
        image = torch.rand(3, 8, 8)
        controller.receive_input("image", image)
        self.assertIsNone(controller.propagate()["output"])
        controller.receive_input("image", image)
        self.assertEqual(controller.propagate()["output"].shape, (1, 2))

    def test_interrupt_then_resume_at_epoch_boundary(self):
        exp = self.recipe_exp(epochs=2)
        recipe = self.make_recipe(exp)

        def interrupt_after_first_epoch(status):
            if status["state"] == "running" and status["epoch"] == 1:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            recipe.train(progress_callback=interrupt_after_first_epoch)
        self.assertEqual(recipe.status["state"], "interrupted")

        resumed = self.make_recipe(exp)
        resumed.train()
        self.assertEqual(resumed.status["state"], "completed")
        records = [
            json.loads(line) for line in (
                pathlib.Path(self.temporary.name) / "metrics.jsonl"
            ).read_text().splitlines()
        ]
        self.assertEqual([record["epoch"] for record in records], [1, 2])

    def test_visualization_marks_current_trainability(self):
        exp = self.recipe_exp()
        status = {
            "state": "running", "current_stage_index": 0,
            "stages": [{"name": "policy", "state": "running", "epoch": 0,
                        "epochs": 1, "best_metric": None}],
        }
        source = TrainingRecipeVisualizer(
            exp, experiment_loader=self.load_experiment, status=status
        ).build().source
        self.assertIn("trainable in current stage", source)
        self.assertIn("Training stages", source)


if __name__ == "__main__":
    unittest.main()
