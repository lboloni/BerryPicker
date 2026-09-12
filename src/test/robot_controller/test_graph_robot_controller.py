"""Tests for the configurable VAE-LSTM-MDN robot-controller graph."""

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from behavior_cloning.mdn import MDN
from exp_run_config import Config
from robot_controller.abstract_rcco import AbstractRCComponent
from robot_controller.graph_robot_controller import (
    GraphRobotController,
    load_controller_spec,
)
from robot_controller.rcco_factory import RCCO_Input, RCCO_Output
from robot_controller import rcco_factory
from robot_controller.rcco_lstm import RCCO_LSTM, ResidualLSTM
from robot_controller.rcco_mdn import RCCO_MDN
from robot_controller.rcco_sp_vae import RCCO_SP_VAE
from robot_controller.roco_factory import create_controller
from robot_controller.visualize_rcco import RCCOVisualizer


class TestResidualLSTMComponent(unittest.TestCase):
    def test_sliding_window_warmup_output_and_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            exp = {
                "architecture": "residual",
                "context_mode": "sliding_window",
                "input_size": 4,
                "hidden_size": 3,
                "num_layers": 3,
                "sequence_length": 2,
                "data_dir": directory,
                "model_file": "lstm.pth",
            }
            torch.save(
                ResidualLSTM(4, 3, 3).state_dict(),
                pathlib.Path(directory) / "lstm.pth",
            )
            component = RCCO_LSTM(exp)

            component.set_input("z", torch.ones(1, 4))
            self.assertFalse(component.propagate())
            self.assertIsNone(component.outputs["h"])

            component.set_input("z", torch.full((1, 4), 2.0))
            self.assertTrue(component.propagate())
            self.assertEqual(component.outputs["h"].shape, (1, 3))

            component.reset_context()
            self.assertEqual(len(component.context), 0)
            self.assertIsNone(component.outputs["h"])

    def test_residual_lstm_rejects_wrong_shape(self):
        model = ResidualLSTM(4, 3, 2)
        with self.assertRaisesRegex(ValueError, "Expected LSTM input shape"):
            model(torch.ones(2, 4))


class TestMDNComponent(unittest.TestCase):
    def _component(self, directory, action_selection):
        exp = {
            "input_dim": 3,
            "hidden_size": 4,
            "output_dim": 2,
            "num_gaussians": 5,
            "action_selection": action_selection,
            "data_dir": directory,
            "model_file": "mdn.pth",
        }
        torch.save(MDN(exp).state_dict(), pathlib.Path(directory) / "mdn.pth")
        return RCCO_MDN(exp)

    def test_distribution_and_expected_action(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            component = self._component(directory, "expected_value")
            component.set_input("h", torch.ones(1, 3))
            self.assertTrue(component.propagate())
            self.assertEqual(component.outputs["mu"].shape, (1, 2, 5))
            self.assertEqual(component.outputs["sigma"].shape, (1, 2, 5))
            self.assertEqual(component.outputs["pi"].shape, (1, 2, 5))
            self.assertEqual(component.outputs["a"].shape, (1, 2))
            expected = torch.sum(
                component.outputs["pi"] * component.outputs["mu"], dim=-1
            )
            self.assertTrue(torch.equal(component.outputs["a"], expected))

    def test_sample_and_maximum_component_modes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            for selection in ("sample", "maximum_component"):
                with self.subTest(selection=selection):
                    component = self._component(directory, selection)
                    component.set_input("h", torch.ones(1, 3))
                    component.propagate()
                    self.assertEqual(component.outputs["a"].shape, (1, 2))
                    self.assertTrue(torch.isfinite(component.outputs["a"]).all())


class TestVAEComponent(unittest.TestCase):
    def test_uses_tensor_encoder_and_preserves_batch(self):
        class Encoder(nn.Module):
            def encode(self, images):
                return images.mean(dim=(-2, -1))[:, :2]

        class SensorProcessor:
            def __init__(self):
                self.enc = Encoder()

        sp_exp = {"latent_size": 2}
        with patch.dict(Config().runtime, {"device": "cpu"}), patch.object(
            Config, "get_experiment", return_value=sp_exp
        ), patch(
            "robot_controller.rcco_sp_vae.create_sp",
            return_value=SensorProcessor(),
        ):
            component = RCCO_SP_VAE({
                "sp_experiment": "sensorprocessing_conv_vae_neo",
                "sp_run": "test",
            })
            component.set_input("image", torch.ones(3, 8, 8))
            self.assertTrue(component.propagate())
            self.assertEqual(component.outputs["z"].shape, (1, 2))


class PassVAE(AbstractRCComponent):
    def __init__(self, exp):
        super().__init__(exp)
        self.inputs["image"] = None
        self.outputs["z"] = None

    def propagate(self):
        self.outputs["z"] = self.inputs["image"]
        return True


class PassLSTM(AbstractRCComponent):
    def __init__(self, exp):
        super().__init__(exp)
        self.inputs["z"] = None
        self.outputs["h"] = None

    def propagate(self):
        self.outputs["h"] = self.inputs["z"][:3]
        return True


class PassMDN(AbstractRCComponent):
    def __init__(self, exp):
        super().__init__(exp)
        self.inputs["h"] = None
        self.outputs.update({"mu": None, "sigma": None, "pi": None, "a": None})

    def propagate(self):
        self.outputs["mu"] = torch.zeros(1)
        self.outputs["sigma"] = torch.ones(1)
        self.outputs["pi"] = torch.ones(1)
        self.outputs["a"] = self.inputs["h"][:2]
        return True


class TestGraphRobotController(unittest.TestCase):
    def setUp(self):
        self.experiments = {
            ("robot_controller", "input"): {"rcco-type": "Input"},
            ("robot_controller", "vae"): {
                "rcco-type": "SP_VAE",
                "sp_experiment": "sp",
                "sp_run": "vae",
            },
            ("sp", "vae"): {"latent_size": 4},
            ("robot_controller", "lstm"): {
                "rcco-type": "LSTM",
                "input_size": 4,
                "hidden_size": 3,
            },
            ("robot_controller", "mdn"): {
                "rcco-type": "MDN",
                "input_dim": 3,
                "hidden_size": 4,
                "output_dim": 2,
                "num_gaussians": 2,
            },
            ("robot_controller", "output"): {"rcco-type": "Output"},
        }
        self.exp = {
            "name": "test graph",
            "components": {
                "image": {"run": "input"},
                "encoder": {"run": "vae"},
                "temporal": {"run": "lstm"},
                "density": {"run": "mdn"},
                "action": {"run": "output"},
            },
            "connections": [
                {
                    "from_component": "image",
                    "from_output": "input",
                    "to_component": "encoder",
                    "to_input": "image",
                },
                {
                    "from_component": "encoder",
                    "from_output": "z",
                    "to_component": "temporal",
                    "to_input": "z",
                },
                {
                    "from_component": "temporal",
                    "from_output": "h",
                    "to_component": "density",
                    "to_input": "h",
                },
                {
                    "from_component": "density",
                    "from_output": "a",
                    "to_component": "action",
                    "to_input": "output",
                },
            ],
        }

    def load_experiment(self, experiment, run):
        return self.experiments[(experiment, run)]

    @staticmethod
    def create_component(exp):
        factories = {
            "Input": RCCO_Input,
            "SP_VAE": PassVAE,
            "LSTM": PassLSTM,
            "MDN": PassMDN,
            "Output": RCCO_Output,
        }
        return factories[exp["rcco-type"]](exp)

    def test_spec_and_end_to_end_routing(self):
        spec = load_controller_spec(self.exp, self.load_experiment)
        self.assertEqual(
            spec["topological_order"],
            ["image", "encoder", "temporal", "density", "action"],
        )
        controller = GraphRobotController(
            self.exp,
            experiment_loader=self.load_experiment,
            component_factory=self.create_component,
        )
        controller.receive_input("image", torch.arange(4, dtype=torch.float32))
        outputs = controller.propagate()
        self.assertTrue(torch.equal(outputs["action"], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(
            controller.read_output("action"), outputs["action"]
        ))

        factory_exp = {**self.exp, "class": "GraphRobotController"}
        controller = create_controller(
            factory_exp,
            experiment_loader=self.load_experiment,
            component_factory=self.create_component,
        )
        self.assertIsInstance(controller, GraphRobotController)

    def test_visualizer_builds_port_aware_graph(self):
        spec = load_controller_spec(self.exp, self.load_experiment)
        dot = RCCOVisualizer(spec).build()
        self.assertIn('PORT="inputs"', dot.source)
        self.assertIn('PORT="outputs"', dot.source)
        self.assertIn("image:outputs", dot.source)
        self.assertIn("encoder:inputs", dot.source)
        self.assertTrue(dot.pipe(format="svg").startswith(b"<?xml"))

    def test_spec_rejects_size_mismatch_and_cycle(self):
        self.experiments[("robot_controller", "mdn")]["input_dim"] = 4
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_controller_spec(self.exp, self.load_experiment)

        self.experiments[("robot_controller", "combine")] = {
            "rcco-type": "Z-combinator",
            "input_size": 2,
            "output_size": 2,
        }
        cyclic = {
            "components": {
                "left": {"run": "combine"},
                "right": {"run": "combine"},
            },
            "connections": [
                {
                    "from_component": "left",
                    "from_output": "z",
                    "to_component": "right",
                    "to_input": "z1",
                },
                {
                    "from_component": "right",
                    "from_output": "z",
                    "to_component": "left",
                    "to_input": "z1",
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "acyclic"):
            load_controller_spec(cyclic, self.load_experiment)

    def test_real_lstm_and_mdn_wait_for_complete_window(self):
        class Encoder(nn.Module):
            def encode(self, images):
                means = images.mean(dim=(-2, -1))
                return torch.cat((means, means[:, :1]), dim=1)

        class SensorProcessor:
            def __init__(self):
                self.enc = Encoder()

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            lstm_exp = self.experiments[("robot_controller", "lstm")]
            lstm_exp.update({
                "architecture": "residual",
                "num_layers": 3,
                "sequence_length": 3,
                "context_mode": "sliding_window",
                "data_dir": directory,
                "model_file": "lstm.pth",
            })
            mdn_exp = self.experiments[("robot_controller", "mdn")]
            mdn_exp.update({
                "action_selection": "expected_value",
                "data_dir": directory,
                "model_file": "mdn.pth",
            })
            torch.save(
                ResidualLSTM(4, 3, 3).state_dict(),
                pathlib.Path(directory) / "lstm.pth",
            )
            torch.save(
                MDN(mdn_exp).state_dict(),
                pathlib.Path(directory) / "mdn.pth",
            )
            with patch.object(
                Config, "get_experiment", return_value={"latent_size": 4}
            ), patch(
                "robot_controller.rcco_sp_vae.create_sp",
                return_value=SensorProcessor(),
            ):
                controller = GraphRobotController(
                    self.exp,
                    experiment_loader=self.load_experiment,
                    component_factory=rcco_factory.create_component,
                )

            for _ in range(2):
                controller.receive_input("image", torch.ones(1, 3, 8, 8))
                self.assertIsNone(controller.propagate()["action"])
            controller.receive_input("image", torch.ones(1, 3, 8, 8))
            action = controller.propagate()["action"]
            self.assertEqual(action.shape, (1, 2))
            self.assertTrue(torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
