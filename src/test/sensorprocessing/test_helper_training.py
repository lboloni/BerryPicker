import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sensorprocessing import helper_training


class TestHelperTraining(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.exp = {
            "data_dir": str(self.data_dir),
            "proprioception_mlp_model_file": "model.pth",
            "epochs": 3,
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def make_model_and_optimizer(value=0.0):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return model, optimizer

    def test_model_available_uses_configured_path(self):
        self.assertFalse(helper_training.model_available(self.exp))
        model_path = self.data_dir / self.exp["proprioception_mlp_model_file"]
        torch.save({"weight": torch.ones((1, 1))}, model_path)
        self.assertTrue(helper_training.model_available(self.exp))

    def test_find_latest_checkpoint_uses_numeric_epoch_order(self):
        checkpoint_dir = self.data_dir / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch_000002.pth").touch()
        (checkpoint_dir / "epoch_000010.pth").touch()
        (checkpoint_dir / "epoch_invalid.pth").touch()

        checkpoint, epoch = helper_training.find_latest_checkpoint(self.data_dir)

        self.assertEqual(checkpoint, checkpoint_dir / "epoch_000010.pth")
        self.assertEqual(epoch, 10)

    def test_load_or_train_loads_existing_final_model(self):
        saved_model, _ = self.make_model_and_optimizer(4.0)
        torch.save(saved_model.state_dict(), self.data_dir / "model.pth")
        model, optimizer = self.make_model_and_optimizer(0.0)

        def should_not_train(*args):
            raise AssertionError("training callback should not run")

        with patch.dict(helper_training.Config().runtime, {"device": "cpu"}):
            result = helper_training.load_or_train(
                self.exp,
                model,
                optimizer,
                should_not_train,
                should_not_train,
            )

        self.assertAlmostEqual(result.weight.item(), 4.0)

    def test_load_or_train_resumes_latest_checkpoint(self):
        checkpoint_dir = self.data_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_model, checkpoint_optimizer = self.make_model_and_optimizer(3.0)
        torch.save(
            {
                "epoch": 2,
                "model_state_dict": checkpoint_model.state_dict(),
                "optimizer_state_dict": checkpoint_optimizer.state_dict(),
                "best_val_loss": 0.5,
                "train_loss": 1.0,
                "val_loss": 0.5,
            },
            checkpoint_dir / "epoch_000002.pth",
        )
        model, optimizer = self.make_model_and_optimizer(0.0)
        observed_weights = []

        def train_step(current_model, _optimizer):
            observed_weights.append(current_model.weight.item())
            return 0.4

        with patch.dict(helper_training.Config().runtime, {"device": "cpu"}):
            result = helper_training.load_or_train(
                self.exp,
                model,
                optimizer,
                train_step,
                lambda _model: 0.3,
                epochs=3,
            )

        self.assertEqual(len(observed_weights), 1)
        self.assertAlmostEqual(observed_weights[0], 3.0)
        self.assertAlmostEqual(result.weight.item(), 3.0)
        self.assertTrue((self.data_dir / "model.pth").is_file())

    def test_load_or_train_restores_optimizer_and_scheduler_state(self):
        checkpoint_dir = self.data_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_model, checkpoint_optimizer = self.make_model_and_optimizer(2.0)
        checkpoint_optimizer.param_groups[0]["lr"] = 0.025
        checkpoint_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            checkpoint_optimizer
        )
        checkpoint_scheduler.step(0.2)
        torch.save(
            {
                "epoch": 2,
                "model_state_dict": checkpoint_model.state_dict(),
                "optimizer_state_dict": checkpoint_optimizer.state_dict(),
                "scheduler_state_dict": checkpoint_scheduler.state_dict(),
                "best_val_loss": 0.2,
                "train_loss": 0.4,
                "val_loss": 0.2,
            },
            checkpoint_dir / "epoch_000002.pth",
        )
        model, optimizer = self.make_model_and_optimizer(0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        with (
            patch.dict(helper_training.Config().runtime, {"device": "cpu"}),
            patch.object(
                helper_training,
                "train_model",
                return_value=model,
            ) as train_model,
        ):
            helper_training.load_or_train(
                self.exp,
                model,
                optimizer,
                lambda _model, _optimizer: 0.0,
                lambda _model: 0.0,
                epochs=3,
                scheduler=scheduler,
            )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)
        self.assertAlmostEqual(scheduler.best, 0.2)
        self.assertEqual(train_model.call_args.kwargs["start_epoch"], 2)
        self.assertAlmostEqual(
            train_model.call_args.kwargs["best_val_loss"], 0.2
        )

    def test_train_model_keeps_only_most_recent_checkpoints(self):
        model, optimizer = self.make_model_and_optimizer()
        validation_losses = iter([0.5, 0.4, 0.3])

        helper_training.train_model(
            self.exp,
            model,
            optimizer,
            lambda _model, _optimizer: 1.0,
            lambda _model: next(validation_losses),
            keep_checkpoints=2,
        )

        checkpoint_names = sorted(
            path.name
            for path in (self.data_dir / "checkpoints").glob("epoch_*.pth")
        )
        self.assertEqual(
            checkpoint_names, ["epoch_000002.pth", "epoch_000003.pth"]
        )
        self.assertTrue(
            (self.data_dir / "checkpoints" / "best_model.pth").is_file()
        )
        self.assertTrue((self.data_dir / "model.pth").is_file())


if __name__ == "__main__":
    unittest.main()
