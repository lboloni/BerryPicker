"""
test_steps.py

Tests for the epoch-step factory and the early-stopping option of the
shared training harness.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import training_harness as helper_training
from training_harness import runner, steps


class TwoViewSum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, batch_views):
        return self.weights[0] * batch_views[0] + self.weights[1] * batch_views[1]


class TestEpochSteps(unittest.TestCase):
    def setUp(self):
        self.config_patch = patch.object(
            runner.Config, "runtime", {"device": torch.device("cpu")}, create=True
        )
        self.config_patch.start()
        patch.object(steps.Config, "runtime", {"device": torch.device("cpu")}, create=True).start()

    def tearDown(self):
        patch.stopall()

    def test_steps_handle_multiview_batches_and_clip_gradients(self):
        view_a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        view_b = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        targets = view_a + 2 * view_b

        def collate(batch):
            return [torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])], torch.stack([b[2] for b in batch])

        dataset = TensorDataset(view_a, view_b, targets)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
        model = TwoViewSum()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        train_step, eval_step = steps.make_epoch_steps(
            torch.nn.MSELoss(), loader, loader, grad_clip_norm=1.0
        )
        loss_before = eval_step(model)
        train_loss = train_step(model, optimizer)
        self.assertGreater(train_loss, 0.0)
        self.assertLess(eval_step(model), loss_before)
        # with a clip norm of 1 and lr 0.001 the weights cannot move by more than 0.002 per epoch
        self.assertTrue(torch.all((model.weights - 1.0).abs() <= 0.002 + 1e-6))

    def test_single_view_batches_and_evaluate_loss(self):
        inputs = torch.tensor([[1.0], [2.0]])
        targets = 3 * inputs
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        self.assertAlmostEqual(steps.evaluate_loss(model, torch.nn.MSELoss(), loader), 0.0)


class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.exp = {
            "data_dir": self.temporary_directory.name,
            "proprioception_mlp_model_file": "model.pth",
            "epochs": 50,
            "early_stopping_patience": 3,
        }
        patch.object(runner.Config, "runtime", {"device": torch.device("cpu")}, create=True).start()

    def tearDown(self):
        patch.stopall()
        self.temporary_directory.cleanup()

    def test_training_stops_after_patience_epochs_without_improvement(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        epochs_run = []
        validation_losses = iter([1.0, 0.9, 0.95, 0.95, 0.95, 0.5, 0.5])

        def train_step(model, optimizer):
            epochs_run.append(1)
            return 1.0

        def eval_step(model):
            return next(validation_losses)

        with patch("builtins.print"):
            helper_training.load_or_train(self.exp, model, optimizer, train_step, eval_step)

        # best at epoch 2 (0.9), then three epochs without improvement -> stop after epoch 5
        self.assertEqual(len(epochs_run), 5)
        self.assertTrue((Path(self.temporary_directory.name) / "model.pth").exists())
        checkpoint = torch.load(
            Path(self.temporary_directory.name) / "checkpoints" / "epoch_000005.pth",
            weights_only=True,
        )
        self.assertEqual(checkpoint["epochs_without_improvement"], 3)
        self.assertAlmostEqual(checkpoint["best_val_loss"], 0.9)

    def test_early_stopping_is_off_by_default(self):
        exp = dict(self.exp)
        del exp["early_stopping_patience"]
        exp["epochs"] = 6
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        epochs_run = []
        with patch("builtins.print"):
            helper_training.load_or_train(
                exp, model, optimizer,
                lambda m, o: epochs_run.append(1) or 1.0,
                lambda m: 1.0,
            )
        self.assertEqual(len(epochs_run), 6)


if __name__ == "__main__":
    unittest.main()
