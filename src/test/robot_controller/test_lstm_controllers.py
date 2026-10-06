"""Tests for the LSTM controllers: recurrent cores, context modes, the chain
training model, stateful chunk training, and the latent cache."""

import pathlib
import sys
import tempfile
import unittest

import torch


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from robot_controller.graph_robot_controller import GraphRobotController
from robot_controller.rcco_lstm import RCCO_LSTM, RECURRENT_CORES
from robot_controller.training_data import make_controller_dataloaders
from robot_controller.training_recipe import create_training_recipe
from sensorprocessing.conv_vae_neo import ConvVAENeo


class Position:
    def __init__(self, value):
        self.value = value

    def to_normalized_vector(self, exp):
        return [self.value] * 2


class FakeDemonstration:
    """A demonstration whose frame i is a deterministic image, and whose
    action at i normalizes to i / 10; the length depends on the name."""

    def __init__(self, exp, demo):
        self.demo = demo
        self.length = 5 + 2 * (len(demo) % 3)
        self.metadata = {"cameras": ["dev0"], "maxsteps": self.length}
        self.actions = [None] * self.length
        generator = torch.Generator().manual_seed(len(demo))
        self.images = torch.rand(self.length, 1, 3, 8, 8, generator=generator)

    def get_image(self, frame, camera, transform):
        return self.images[frame], None

    def get_action(self, i, type, exp):
        return Position(i / 10)


def load_demonstration(experiment, run):
    return {}


class TestRecurrentCores(unittest.TestCase):
    def test_stepwise_state_equals_whole_sequence(self):
        torch.manual_seed(1)
        sequence = torch.randn(2, 6, 4)
        for name, core_class in RECURRENT_CORES.items():
            core = core_class(4, 3, 2).eval()
            whole, _ = core(sequence)
            state = None
            steps = []
            for t in range(6):
                output, state = core(sequence[:, t:t + 1], state)
                steps.append(output)
            self.assertTrue(
                torch.allclose(whole, torch.cat(steps, dim=1), atol=1e-6), name)


class TestContextModes(unittest.TestCase):
    def setUp(self):
        Config().runtime["device"] = "cpu"
        torch.manual_seed(2)
        self.latents = [torch.randn(1, 4) for _ in range(5)]

    def component(self, **values):
        exp = {"input_size": 4, "hidden_size": 3, "num_layers": 2, **values}
        return RCCO_LSTM(exp, load_state=False)

    def run_steps(self, component):
        outputs = []
        for latent in self.latents:
            component.set_input("z", latent)
            component.propagate()
            outputs.append(component.outputs["h"])
        return outputs

    def test_stateful_matches_sequence_and_resets(self):
        component = self.component(architecture="plain", context_mode="stateful")
        outputs = self.run_steps(component)
        whole, _ = component.model(torch.cat(self.latents).unsqueeze(0))
        self.assertTrue(torch.allclose(torch.cat(outputs), whole[0], atol=1e-6))
        component.reset_context()
        self.assertIsNone(component.state)
        self.assertTrue(torch.allclose(self.run_steps(component)[0], outputs[0]))

    def test_sliding_window_reruns_the_window(self):
        component = self.component(
            architecture="residual", context_mode="sliding_window",
            sequence_length=3)
        outputs = self.run_steps(component)
        self.assertIsNone(outputs[1])
        window, _ = component.model(torch.cat(self.latents[2:5]).unsqueeze(0))
        self.assertTrue(torch.allclose(outputs[4], window[:, -1], atol=1e-6))


class TestChainTraining(unittest.TestCase):
    """VAE encoder -> LSTM -> head controllers on fake demonstrations."""

    def setUp(self):
        Config().runtime["device"] = "cpu"
        self.temporary = tempfile.TemporaryDirectory()
        directory = self.temporary.name
        self.experiments = {
            ("robot_controller", "input"): {"rcco-type": "Input"},
            ("robot_controller", "vae"): {
                "rcco-type": "SP_VAE", "sp_experiment": "sp", "sp_run": "neo"},
            ("sp", "neo"): {
                "architecture_version": 1, "image_size": [8, 8],
                "input_channels": 3, "latent_size": 2, "base_channels": 2,
                "max_channels": 4, "bottleneck_max_size": 4,
                "group_norm_groups": 1, "model_file": "vae.pth",
                "data_dir": directory},
            ("robot_controller", "stateful"): {
                "rcco-type": "LSTM", "architecture": "plain", "input_size": 2,
                "hidden_size": 3, "num_layers": 2, "context_mode": "stateful",
                "model_file": "lstm.pth", "data_dir": directory},
            ("robot_controller", "window"): {
                "rcco-type": "LSTM", "architecture": "residual",
                "input_size": 2, "hidden_size": 3, "num_layers": 2,
                "context_mode": "sliding_window", "sequence_length": 2,
                "model_file": "lstm.pth", "data_dir": directory},
            ("robot_controller", "mlp"): {
                "rcco-type": "MLP", "input_size": 3, "hidden_sizes": [4],
                "output_size": 2, "output_activation": "sigmoid",
                "model_file": "mlp.pth", "data_dir": directory},
            ("robot_controller", "mdn"): {
                "rcco-type": "MDN", "input_dim": 3, "hidden_size": 4,
                "output_dim": 2, "num_gaussians": 2,
                "action_selection": "expected_value", "model_file": "mdn.pth",
                "data_dir": directory},
            ("robot_controller", "output"): {"rcco-type": "Output", "size": 2},
            ("robot", "tiny"): {},
        }
        torch.save(ConvVAENeo(self.experiments[("sp", "neo")]).state_dict(),
                   pathlib.Path(directory) / "vae.pth")

    def tearDown(self):
        self.temporary.cleanup()

    def load_experiment(self, experiment, run):
        return self.experiments[(experiment, run)]

    def controller(self, core_run, head_run):
        head_input = "h" if head_run == "mdn" else "z"
        controller = {
            "name": "chain", "components": {
                "image": {"run": "input"}, "encoder": {"run": "vae"},
                "lstm": {"run": core_run}, "head": {"run": head_run},
                "output": {"run": "output"}},
            "connections": [
                {"from_component": "image", "from_output": "input",
                 "to_component": "encoder", "to_input": "image"},
                {"from_component": "encoder", "from_output": "z",
                 "to_component": "lstm", "to_input": "z"},
                {"from_component": "lstm", "from_output": "h",
                 "to_component": "head", "to_input": head_input},
                {"from_component": "head", "from_output": "a",
                 "to_component": "output", "to_input": "output"}]}
        self.experiments[("robot_controller", "controller")] = controller
        return controller

    def recipe(self, trainable, extra=None):
        exp = {
            "class": "StagedControllerTrainingRecipe",
            "data_dir": self.temporary.name, "model_file": "bundle.pth",
            "controller": {"exp": "robot_controller", "run": "controller"},
            "robot": {"exp": "robot", "run": "tiny"},
            "initial_states": {"encoder": {"mode": "configured"},
                               "lstm": {"mode": "random"},
                               "head": {"mode": "random"}},
            "stages": [{
                "name": "stage", "trainable_components": trainable,
                "epochs": 1, "optimizer": "Adam",
                "learning_rates": {label: 0.01 for label in trainable}}],
            "training_data": [["pack", "train_a", "dev0"],
                              ["pack", "train_bb", "dev0"],
                              ["pack", "train_c", "dev0"]],
            "validation_data": [["pack", "valid", "dev0"]],
            "batch_size": 2, "chunk_length": 3, "random_seed": 7,
            "keep_checkpoints": 2, **(extra or {})}

        def dataloaders(*args, **kwargs):
            return make_controller_dataloaders(
                *args, **kwargs, demonstration_factory=FakeDemonstration,
                experiment_loader=load_demonstration)

        return create_training_recipe(
            exp, experiment_loader=self.load_experiment,
            dataloader_factory=dataloaders)

    def deployed_outputs(self, demo):
        """The controller bundle run frame by frame over a demonstration."""
        controller = GraphRobotController(
            self.experiments[("robot_controller", "controller")],
            experiment_loader=self.load_experiment,
            bundle_path=pathlib.Path(self.temporary.name) / "bundle.pth")
        controller.reset_context()
        outputs = []
        for frame in demo.images:
            controller.receive_input("image", frame[0])
            outputs.append(controller.propagate().get("output"))
        return outputs

    def test_stateful_lstm_mlp_round_trip(self):
        self.controller("stateful", "mlp")
        recipe = self.recipe(["encoder", "lstm", "head"])
        self.assertEqual(recipe.model.context_mode, "stateful")
        recipe.train()
        demo = FakeDemonstration({}, "valid")
        recipe.model.eval()
        with torch.no_grad():
            expected, _ = recipe.model(demo.images.transpose(0, 1))
        deployed = torch.cat(self.deployed_outputs(demo))
        self.assertTrue(torch.allclose(deployed, expected[0], atol=1e-5))

    def test_sliding_window_lstm_mdn_round_trip(self):
        self.controller("window", "mdn")
        recipe = self.recipe(["lstm", "head"])
        self.assertTrue(recipe.model.stochastic)
        recipe.train()
        demo = FakeDemonstration({}, "valid")
        deployed = self.deployed_outputs(demo)
        self.assertIsNone(deployed[0])
        recipe.model.eval()
        with torch.no_grad():
            (mu, sigma, pi), _ = recipe.model(demo.images[3:5].transpose(0, 1))
        expected = torch.sum(pi[:, -1] * mu[:, -1], dim=-1)
        self.assertTrue(torch.allclose(deployed[4], expected, atol=1e-5))

    def test_chunks_carry_the_state_of_whole_demonstrations(self):
        self.controller("stateful", "mlp")
        recipe = self.recipe(["lstm", "head"])
        recipe.initialize()
        recipe.model.set_trainable(["lstm", "head"])
        recipe.model.eval()
        short, _ = recipe._make_dataloaders()
        long, _ = make_controller_dataloaders(
            {**recipe.exp, "chunk_length": 100}, recipe.model.sensor_exp,
            recipe.robot_exp, 1, 2, stateful=True,
            demonstration_factory=FakeDemonstration,
            experiment_loader=load_demonstration)
        short.shuffle = long.shuffle = False

        def per_demonstration(loader):
            """The model outputs, carrying the state between chunks,
            regrouped as one sequence per demonstration."""
            demonstrations, state = [], None
            with torch.no_grad():
                for inputs, _, mask, reset in loader:
                    if reset:
                        state, rows = None, [[] for _ in range(len(inputs))]
                        demonstrations += rows
                    output, state = recipe.model(inputs, state)
                    for row, values, valid in zip(rows, output, mask):
                        row.append(values[valid])
            return [torch.cat(row) for row in demonstrations]

        chunked, whole = per_demonstration(short), per_demonstration(long)
        self.assertEqual(len(chunked), 3)
        for a, b in zip(chunked, whole):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_latent_cache_gives_the_losses_of_the_frames(self):
        self.controller("stateful", "mlp")
        recipe = self.recipe(["lstm", "head"])
        recipe.initialize()
        recipe.model.set_trainable(["lstm", "head"])
        _, validation = recipe._make_dataloaders()
        frames = recipe._validate_epoch(validation)
        recipe._set_latent_cache([validation])
        self.assertIsNotNone(validation.dataset.latents)
        cached = recipe._validate_epoch(validation)
        for key, value in frames.items():
            self.assertAlmostEqual(value, cached[key], places=6)


if __name__ == "__main__":
    unittest.main()
