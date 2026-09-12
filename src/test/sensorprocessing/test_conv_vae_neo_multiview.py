"""Tests for the separate Neo MultiView Concat and Fusion model families."""

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from sensorprocessing.conv_vae_neo import ConvVAENeo, ConvVAENeoLoss
from sensorprocessing.conv_vae_neo_multiview_concat import (
    ConvVAENeoMultiViewConcatModel,
    make_vae_epoch_steps,
)
from sensorprocessing.conv_vae_neo_multiview_fusion import (
    ConvVAENeoMeanEncoder,
    ConvVAENeoMultiViewFusionModel,
    initialize_from_source_vae,
)
from sensorprocessing.multiview_data import (
    DemonstrationMultiViewDataset,
    collate_multiview_images,
    collate_multiview_proprioception,
    validate_multiview_partitions,
)
from sensorprocessing import sp_factory
from sensorprocessing.sp_conv_vae_neo_multiview_concat import (
    ConvVaeNeoMultiViewConcatSensorProcessing,
)
from sensorprocessing.sp_conv_vae_neo_multiview_fusion import (
    ConvVaeNeoMultiViewFusionSensorProcessing,
)


def base_exp(**overrides):
    exp = {
        "architecture_version": 1,
        "image_size": [16, 24],
        "input_channels": 3,
        "latent_size": 5,
        "base_channels": 2,
        "max_channels": 8,
        "bottleneck_max_size": 4,
        "group_norm_groups": 2,
        "num_views": 2,
        "cameras": ["dev2", "dev3"],
        "stack_mode": "width",
        "fusion_type": "concat_proj",
        "fusion_dropout": 0.0,
        "shared_backbone": True,
        "batched_backbone": True,
        "freeze_vae_encoder": False,
        "output_size": 6,
        "proprio_step_1": 8,
        "proprio_step_2": 7,
        "reconstruction_loss": "mse",
        "kl_weight": 0.001,
    }
    exp.update(overrides)
    return exp


def views(batch=4):
    return [torch.rand(batch, 3, 16, 24) for _ in range(2)]


class TestMultiViewConcatModel(unittest.TestCase):
    def test_preserves_full_composite_width_and_reconstructs(self):
        model = ConvVAENeoMultiViewConcatModel(base_exp())
        camera_views = views(batch=2)

        composite = model.compose_views(camera_views)
        reconstruction, mu, logvar = model(camera_views)

        self.assertEqual(composite.shape, (2, 3, 16, 48))
        self.assertEqual(reconstruction.shape, composite.shape)
        self.assertEqual(mu.shape, (2, 5))
        self.assertEqual(logvar.shape, (2, 5))
        self.assertEqual(model.encode_views(camera_views).shape, (2, 5))
        self.assertEqual(
            [tuple(view.shape) for view in model.split_composite(reconstruction)],
            [(2, 3, 16, 24), (2, 3, 16, 24)],
        )

    def test_rejects_channel_stacking_and_bad_views(self):
        with self.assertRaisesRegex(ValueError, "stack_mode='width'"):
            ConvVAENeoMultiViewConcatModel(base_exp(stack_mode="channel"))
        model = ConvVAENeoMultiViewConcatModel(base_exp())
        with self.assertRaisesRegex(ValueError, "Expected 2 views"):
            model([views()[0]])
        with self.assertRaisesRegex(ValueError, "must have shape"):
            model([torch.rand(2, 3, 8, 24), views(batch=2)[1]])

    def test_concat_epoch_steps_backpropagate(self):
        exp = base_exp()
        model = ConvVAENeoMultiViewConcatModel(exp)
        batches = [views(batch=2), views(batch=2)]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_step, validation_step = make_vae_epoch_steps(
            ConvVAENeoLoss(exp), batches, batches, grad_clip_norm=1.0
        )
        with patch.dict(Config().runtime, {"device": "cpu"}):
            train_loss = training_step(model, optimizer)
            validation_loss = validation_step(model)
        self.assertTrue(np.isfinite(train_loss))
        self.assertTrue(np.isfinite(validation_loss))


class TestMultiViewFusionModel(unittest.TestCase):
    def test_mean_encoder_loads_exact_neo_mean_weights(self):
        exp = base_exp()
        vae = ConvVAENeo(exp).eval()
        encoder = ConvVAENeoMeanEncoder(exp).eval()
        encoder.load_vae_state_dict(vae.state_dict())
        images = torch.rand(2, 3, 16, 24)
        with torch.no_grad():
            self.assertTrue(torch.equal(encoder(images), vae.encode(images)))

    def test_all_existing_fusion_variants(self):
        for fusion_type in (
            "concat_proj",
            "indiv_proj",
            "attention",
            "weighted_sum",
            "gated",
        ):
            with self.subTest(fusion_type=fusion_type):
                model = ConvVAENeoMultiViewFusionModel(
                    base_exp(fusion_type=fusion_type)
                )
                camera_views = views()
                self.assertEqual(model.encode_views(camera_views).shape, (4, 5))
                output = model(camera_views)
                self.assertEqual(output.shape, (4, 6))
                output.sum().backward()

    def test_shared_frozen_and_separate_backbones(self):
        shared = ConvVAENeoMultiViewFusionModel(
            base_exp(freeze_vae_encoder=True)
        )
        self.assertFalse(any(
            parameter.requires_grad for parameter in shared.backbones.parameters()
        ))
        separate = ConvVAENeoMultiViewFusionModel(
            base_exp(shared_backbone=False, batched_backbone=False)
        )
        self.assertEqual(len(separate.backbones.backbones), 2)

    def test_source_vae_initializes_encoder_without_retaining_decoder(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            source_exp = base_exp(data_dir=directory, model_file="source.pth")
            source_vae = ConvVAENeo(source_exp)
            torch.save(
                source_vae.state_dict(), pathlib.Path(directory) / "source.pth"
            )
            fusion_exp = base_exp(
                source_vae_experiment="sensorprocessing_conv_vae_neo",
                source_vae_run="source",
            )
            model = ConvVAENeoMultiViewFusionModel(fusion_exp)
            with patch.object(Config, "get_experiment", return_value=source_exp):
                initialize_from_source_vae(fusion_exp, model)
            loaded = model.backbones.backbone.state_dict()
            self.assertTrue(torch.equal(
                loaded["fc_mu.weight"], source_vae.state_dict()["fc_mu.weight"]
            ))
            self.assertFalse(any(key.startswith("decoder") for key in loaded))


class TestMultiViewData(unittest.TestCase):
    def test_lazy_synchronized_views_and_optional_target(self):
        class Position:
            @staticmethod
            def to_normalized_vector(_robot_exp):
                return np.arange(6, dtype=np.float32)

        class Demonstration:
            def __init__(self, _exp, name):
                self.demo = name
                self.metadata = {
                    "cameras": ["dev2", "dev3"],
                    "maxsteps": 3,
                }

            @staticmethod
            def get_image(timestep, camera, transform):
                del transform
                value = timestep + (0 if camera == "dev2" else 10)
                return torch.full((1, 3, 16, 24), float(value)), None

            @staticmethod
            def get_action(timestep, action, robot_exp):
                del timestep, action, robot_exp
                return Position()

        exp = base_exp(
            frame_stride=2,
            training_data=[["run", "train-demo", ["dev2", "dev3"]]],
            validation_data=[["run", "validation-demo", ["dev2", "dev3"]]],
        )
        dataset = DemonstrationMultiViewDataset(
            exp,
            "training_data",
            robot_exp={"robot": True},
            demonstration_factory=Demonstration,
            experiment_loader=lambda experiment, run: {},
        )
        self.assertEqual(len(dataset), 2)
        sample_views, target = dataset[1]
        self.assertEqual([float(view[0, 0, 0]) for view in sample_views], [2.0, 12.0])
        self.assertEqual(target.tolist(), list(range(6)))
        batched_views = collate_multiview_images([sample_views, sample_views])
        self.assertEqual(batched_views[0].shape, (2, 3, 16, 24))
        paired_views, paired_targets = collate_multiview_proprioception(
            [(sample_views, target), (sample_views, target)]
        )
        self.assertEqual(paired_views[1].shape, (2, 3, 16, 24))
        self.assertEqual(paired_targets.shape, (2, 6))

    def test_rejects_partition_overlap_and_camera_reordering(self):
        with self.assertRaisesRegex(ValueError, "same demonstrations"):
            validate_multiview_partitions(base_exp(
                training_data=[["run", "demo", ["dev2", "dev3"]]],
                validation_data=[["run", "demo", ["dev2", "dev3"]]],
            ))
        with self.assertRaisesRegex(ValueError, "Camera order"):
            validate_multiview_partitions(base_exp(
                training_data=[["run", "train", ["dev3", "dev2"]]],
                validation_data=[["run", "validation", ["dev2", "dev3"]]],
            ))


class TestMultiViewRuntimeIntegration(unittest.TestCase):
    def _runtime_exp(self, directory, model_file, class_name, **overrides):
        return base_exp(
            data_dir=directory,
            model_file=model_file,
            **{"class": class_name},
            **overrides,
        )

    def test_concat_wrapper_loads_and_encodes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            exp = self._runtime_exp(
                directory,
                "concat.pth",
                "ConvVaeNeoMultiViewConcatSensorProcessing",
            )
            trained = ConvVAENeoMultiViewConcatModel(exp)
            torch.save(trained.state_dict(), pathlib.Path(directory) / "concat.pth")
            processor = ConvVaeNeoMultiViewConcatSensorProcessing(exp)
            self.assertEqual(processor.process(views(batch=1)).shape, (5,))
            with self.assertRaisesRegex(ValueError, "differs from trained order"):
                processor._warn_on_camera_order(["dev3", "dev2"])

    def test_fusion_wrapper_loads_and_factory_registers_both(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            Config().runtime, {"device": "cpu"}
        ):
            exp = self._runtime_exp(
                directory,
                "fusion.pth",
                "ConvVaeNeoMultiViewFusionSensorProcessing",
            )
            trained = ConvVAENeoMultiViewFusionModel(exp)
            torch.save(trained.state_dict(), pathlib.Path(directory) / "fusion.pth")
            processor = ConvVaeNeoMultiViewFusionSensorProcessing(exp)
            self.assertEqual(processor.process(views(batch=1)).shape, (5,))

        self.assertTrue(sp_factory.is_multiview_sp({
            "class": "ConvVaeNeoMultiViewConcatSensorProcessing"
        }))
        self.assertTrue(sp_factory.is_multiview_sp({
            "class": "ConvVaeNeoMultiViewFusionSensorProcessing"
        }))

    def test_factory_constructs_both_new_processors(self):
        concat_name = "ConvVaeNeoMultiViewConcatSensorProcessing"
        fusion_name = "ConvVaeNeoMultiViewFusionSensorProcessing"
        with patch.object(
            sp_factory.sp_conv_vae_neo_multiview_concat, concat_name
        ) as concat_class, patch.object(
            sp_factory.sp_conv_vae_neo_multiview_fusion, fusion_name
        ) as fusion_class:
            sp_factory.create_sp({"class": concat_name})
            sp_factory.create_sp({"class": fusion_name})
        concat_class.assert_called_once_with({"class": concat_name})
        fusion_class.assert_called_once_with({"class": fusion_name})


if __name__ == "__main__":
    unittest.main()
