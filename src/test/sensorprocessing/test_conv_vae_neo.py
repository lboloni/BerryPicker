import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from sensorprocessing.conv_vae_neo import (
    ConvVAENeo,
    ConvVAENeoLoss,
    DemonstrationImageDataset,
    make_vae_epoch_steps,
)
from sensorprocessing import sp_factory
from sensorprocessing.sp_conv_vae_neo import ConvVaeNeoSensorProcessing
from training_harness.checkpoints import model_file


def model_exp(**overrides):
    exp = {
        "architecture_version": 1,
        "image_size": [32, 48],
        "input_channels": 3,
        "latent_size": 5,
        "base_channels": 4,
        "max_channels": 16,
        "bottleneck_max_size": 8,
        "group_norm_groups": 4,
        "reconstruction_loss": "mse",
        "kl_weight": 0.001,
    }
    exp.update(overrides)
    return exp


class TestConvVAENeo(unittest.TestCase):
    def test_forward_encode_decode_shapes_for_rectangular_input(self):
        model = ConvVAENeo(model_exp())
        images = torch.rand(2, 3, 32, 48)

        reconstruction, mu, logvar = model(images)

        self.assertEqual(reconstruction.shape, images.shape)
        self.assertEqual(mu.shape, (2, 5))
        self.assertEqual(logvar.shape, (2, 5))
        self.assertEqual(model.encode(images).shape, (2, 5))
        self.assertEqual(model.decode(mu).shape, images.shape)

    def test_scales_to_256_pixel_images_without_a_fixed_flatten_size(self):
        model = ConvVAENeo(model_exp(
            image_size=[256, 256], latent_size=3, base_channels=2,
            max_channels=4,
        ))
        model.eval()

        with torch.no_grad():
            reconstruction, mu, _ = model(torch.rand(1, 3, 256, 256))

        self.assertEqual(reconstruction.shape, (1, 3, 256, 256))
        self.assertEqual(mu.shape, (1, 3))
        self.assertLessEqual(max(model.encoded_size), 8)

    def test_evaluation_reconstruction_is_deterministic(self):
        model = ConvVAENeo(model_exp()).eval()
        images = torch.rand(1, 3, 32, 48)
        with torch.no_grad():
            first = model(images)[0]
            second = model(images)[0]
        self.assertTrue(torch.equal(first, second))

    def test_loss_is_finite_and_backpropagates(self):
        model = ConvVAENeo(model_exp())
        images = torch.rand(2, 3, 32, 48)
        loss_function = ConvVAENeoLoss(model_exp())

        components = loss_function.components(model(images), images)
        components["loss"].backward()

        self.assertTrue(torch.isfinite(components["loss"]))
        self.assertTrue(torch.isfinite(components["reconstruction"]))
        self.assertTrue(torch.isfinite(components["kl"]))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_invalid_configuration_and_input_raise(self):
        with self.assertRaisesRegex(ValueError, "image_size"):
            ConvVAENeo(model_exp(image_size=[32]))
        with self.assertRaisesRegex(ValueError, "architecture_version"):
            ConvVAENeo(model_exp(architecture_version=2))
        model = ConvVAENeo(model_exp())
        with self.assertRaisesRegex(ValueError, "expected input shape"):
            model(torch.rand(1, 3, 64, 64))

    def test_epoch_steps_train_and_validate(self):
        exp = model_exp(image_size=[16, 16], latent_size=2)
        model = ConvVAENeo(exp)
        images = torch.rand(4, 3, 16, 16)
        loader = DataLoader(TensorDataset(images), batch_size=2)
        # TensorDataset returns a one-item tuple; the VAE loader returns tensors.
        tensor_loader = [batch[0] for batch in loader]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_step, validation_step = make_vae_epoch_steps(
            ConvVAENeoLoss(exp), tensor_loader, tensor_loader, grad_clip_norm=1.0
        )
        with patch.dict(Config().runtime, {"device": "cpu"}):
            train_loss = train_step(model, optimizer)
            validation_loss = validation_step(model)
        self.assertTrue(np.isfinite(train_loss))
        self.assertTrue(np.isfinite(validation_loss))


class TestConvVAENeoIntegration(unittest.TestCase):
    def test_runtime_processor_loads_model_file_and_returns_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = model_exp(
                data_dir=directory,
                model_file="conv_vae_neo.pth",
                **{"class": "ConvVaeNeoSensorProcessing"},
            )
            expected_model = ConvVAENeo(exp)
            torch.save(
                expected_model.state_dict(), pathlib.Path(directory) / exp["model_file"]
            )

            with patch.dict(Config().runtime, {"device": "cpu"}):
                processor = ConvVaeNeoSensorProcessing(exp)
                latent = processor.process(torch.rand(1, 3, 32, 48))

            self.assertEqual(latent.shape, (5,))
            self.assertTrue(np.all(np.isfinite(latent)))

    def test_factory_registers_neo(self):
        with patch.object(
            sp_factory.sp_conv_vae_neo, "ConvVaeNeoSensorProcessing"
        ) as processor_class:
            exp = {"class": "ConvVaeNeoSensorProcessing"}
            sp_factory.create_sp(exp)
        processor_class.assert_called_once_with(exp)

    def test_neutral_and_legacy_model_file_keys(self):
        self.assertEqual(
            model_file({"data_dir": "/tmp/example", "model_file": "neo.pth"}).name,
            "neo.pth",
        )
        self.assertEqual(
            model_file({
                "data_dir": "/tmp/example",
                "proprioception_mlp_model_file": "legacy.pth",
            }).name,
            "legacy.pth",
        )

    def test_lazy_image_dataset_uses_frame_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            for frame in range(3):
                Image.new("RGB", (20, 20), color=(frame, 2, 3)).save(
                    directory / f"{frame:05d}_camera.jpg"
                )

            class FakeDemonstration:
                def __init__(self, _exp, name):
                    self.demo = name
                    self.metadata = {
                        "cameras": ["camera"],
                        "maxsteps": 3,
                        "stored_as_images": True,
                        "stored_as_video": False,
                    }

                @staticmethod
                def get_image_path(frame, camera):
                    return directory / f"{frame:05d}_{camera}.jpg"

            exp = model_exp(
                image_size=[16, 16],
                frame_stride=2,
                training_data=[["run", "demo", "camera"]],
            )
            dataset = DemonstrationImageDataset(
                exp,
                "training_data",
                demonstration_factory=FakeDemonstration,
                experiment_loader=lambda *_args: {},
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].shape, (3, 16, 16))


if __name__ == "__main__":
    unittest.main()
