"""Synthetic CPU tests; no demonstration downloads or training artifacts."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))
from sensorprocessing.vae_gan import VAEGAN
from sensorprocessing.vae_gan_training import VAEGANTrainer, train, initialize_vae
from sensorprocessing.conv_vae_neo import ConvVAENeo
from sensorprocessing.sp_factory import create_sp
from robot_controller.rcco_sp_vae import ConvVAENeoEncoder
from exp_run_config import Config, Experiment


def settings(directory="", **overrides):
    exp = dict(architecture_version=1, image_size=[16, 24], input_channels=3,
               latent_size=3, base_channels=4, max_channels=8, bottleneck_max_size=4,
               group_norm_groups=2, discriminator_base_channels=4,
               discriminator_max_channels=8, discriminator_feature_layer=-2,
               data_dir=directory, model_file="vae_gan_vae.pth", epochs=2,
               training_data=[["run", "train", "camera"]],
               validation_data=[["run", "validation", "camera"]], random_seed=12,
               keep_checkpoints=1, **{"class": "VAEGANSensorProcessing"})
    exp.update(overrides)
    return exp


def loaders():
    data = torch.rand(4, 3, 16, 24, generator=torch.Generator().manual_seed(7))
    return (DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(8)),
            DataLoader(data[:2], batch_size=2))


class TestVAEGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_shapes_and_large_non_square(self):
        for size in ([16, 24], [256, 320], [33, 49]):
            model = VAEGAN(settings(image_size=size)).eval()
            images = torch.rand(2, 3, *size)
            with torch.no_grad():
                recon, mu, logvar = model(images)
                self.assertEqual(recon.shape, images.shape)
                self.assertEqual(mu.shape, (2, 3))
                self.assertEqual(model.discriminator(images)[0].shape, (2, 1))
                torch.testing.assert_close(mu, model.encode(images))

    def test_isolated_updates(self):
        model = VAEGAN(settings())
        trainer = VAEGANTrainer(model, settings())
        for part in trainer.groups:
            before = {key: value.clone() for key, value in model.state_dict().items()}
            trainer.update(torch.rand(2, 3, 16, 24), part)
            for name, group in trainer.groups.items():
                params = list(group.parameters())
                if name == part:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params))
                else:
                    self.assertTrue(all(p.grad is None for p in params))
                self.assertTrue(all(p.requires_grad for p in params))
            changed = [key for key, value in model.state_dict().items() if not torch.equal(value, before[key])]
            prefixes = {"encoder": ("vae.encoder.", "vae.fc_mu.", "vae.fc_logvar."),
                        "generator": ("vae.decoder.", "vae.fc_decode.", "vae.output_layer."),
                        "discriminator": ("discriminator.",)}
            self.assertTrue(changed)
            self.assertTrue(all(key.startswith(prefixes[part]) for key in changed))

    def test_resume_matches_uninterrupted_and_export(self):
        with tempfile.TemporaryDirectory() as continuous, tempfile.TemporaryDirectory() as resumed:
            exp = settings(continuous)
            full, full_history = train(exp, loaders=loaders(), device="cpu")
            resumed_exp = settings(resumed, time_started="first invocation")
            def interrupt(model, history):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                train(Experiment(resumed_exp), loaders=loaders(), device="cpu", callback=interrupt)
            del resumed_exp["time_started"]  # Config omits this when reopening an existing run.
            # Derived files can be recovered from the authoritative last checkpoint.
            Path(resumed, "metrics.jsonl").write_text("incomplete", encoding="utf-8")
            continued, history = train(resumed_exp, loaders=loaders(), device="cpu")
            self.assertEqual(full_history, history)
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, continued.state_dict()[key], rtol=0, atol=0)
            checkpoint = torch.load(Path(resumed, "checkpoints/last.pt"), weights_only=True)
            self.assertEqual(set(checkpoint["optimizers"]), {"encoder", "generator", "discriminator"})
            self.assertTrue(all(opt["state"] for opt in checkpoint["optimizers"].values()))
            self.assertEqual(len(list(Path(resumed, "checkpoints").glob("epoch_*.pt"))), 1)
            exported = torch.load(Path(resumed, "vae_gan_vae.pth"), weights_only=True)
            vae = ConvVAENeo(resumed_exp).eval()
            vae.load_state_dict(exported)
            encoder = ConvVAENeoEncoder(resumed_exp).eval()
            encoder.load_vae_state_dict(exported)
            inputs = next(iter(loaders()[1]))
            torch.testing.assert_close(vae.encode(inputs), encoder(inputs))
            with patch.dict(Config().runtime, {"device": "cpu"}):
                processor = create_sp(resumed_exp)
                torch.testing.assert_close(processor.enc.encode(inputs), vae.encode(inputs))
            self.assertNotIn("discriminator", dict(processor.enc.named_modules()))
            with self.assertRaises(ValueError):
                train(settings(resumed, kl_weight=3), loaders=loaders(), device="cpu")

    def test_initialization_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = settings(directory)
            exp = settings(initialization="conv_vae_neo", initialization_experiment="neo", initialization_run="test")
            model = VAEGAN(exp)
            with patch.object(Config, "get_experiment", return_value=source):
                with self.assertRaises(FileNotFoundError):
                    initialize_vae(model, exp)
                original = ConvVAENeo(source)
                torch.save(original.state_dict(), Path(directory, source["model_file"]))
                self.assertIn("sha256", initialize_vae(model, exp))
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, model.vae.state_dict()[key])
                with self.assertRaises(ValueError):
                    initialize_vae(model, {**exp, "group_norm_groups": 1})

    def test_warmup_and_demonstration_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = settings(directory, discriminator_warmup_epochs=1)
            _, history = train(exp, loaders=loaders(), device="cpu")
            self.assertEqual([row["phase"] for row in history], ["discriminator_warmup", "joint"])
            self.assertNotIn("encoder", history[0])
            exp["validation_data"] = [["run", "train", "different_camera"]]
            with self.assertRaises(ValueError):
                train(exp, loaders=loaders(), device="cpu")

    def test_mid_epoch_interrupt_replays_last_committed_epoch(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as reference:
            expected, expected_history = train(settings(reference), loaders=loaders(), device="cpu")
            original_batch = VAEGANTrainer.batch
            calls = 0

            def interrupted_batch(trainer, images, warmup=False):
                nonlocal calls
                result = original_batch(trainer, images, warmup=warmup)
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return result

            with patch.object(VAEGANTrainer, "batch", interrupted_batch):
                with self.assertRaises(KeyboardInterrupt):
                    train(settings(directory), loaders=loaders(), device="cpu")
            actual, history = train(settings(directory), loaders=loaders(), device="cpu")
            self.assertEqual(history, expected_history)
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)

    def test_visualization_helpers(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sensorprocessing.vae_gan_visualization import plot_history, plot_reconstructions
        model = VAEGAN(settings()).eval()
        figure = plot_reconstructions(model, torch.rand(2, 3, 16, 24), torch.rand(2, 3))
        self.assertEqual(len(figure.axes), 6)
        history = [{"epoch": 1, "encoder": 1., "validation_mse": .2, "validation_feature": .3}]
        self.assertEqual(len(plot_history(history).axes), 3)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
