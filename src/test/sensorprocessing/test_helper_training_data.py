import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Experiment
from sensorprocessing import (
    helper_training_data,
    sensor_processing,
    sp_conv_vae_concat_multiview,
    sp_factory,
    sp_propriotuned_cnn,
    vit_helper,
)


class TestMultiViewSensorProcessing(unittest.TestCase):
    def test_processes_an_ordered_demonstration_timestep(self):
        class RecordingProcessor(sensor_processing.MultiViewSensorProcessing):
            def __init__(self):
                super().__init__({"latent_size": 2, "num_views": 2, "image_size": [8, 8]})
                self.num_views = 2
                self.received_views = None

            def process(self, views):
                self.received_views = views
                return np.array([1.0, 2.0], dtype=np.float32)

        class FakeDemonstration:
            def __init__(self):
                self.calls = []

            def get_image(self, timestep, camera, transform):
                self.calls.append((timestep, camera, transform))
                return torch.tensor([[len(self.calls)]], dtype=torch.float32), None

        processor = RecordingProcessor()
        demonstration = FakeDemonstration()

        result = processor.process_demonstration(
            demonstration, 7, ["camera-2", "camera-1"], transform="transform"
        )

        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))
        self.assertEqual(
            demonstration.calls,
            [(7, "camera-2", "transform"), (7, "camera-1", "transform")],
        )
        self.assertEqual(
            [view.item() for view in processor.received_views], [1.0, 2.0]
        )
        self.assertFalse(callable(processor.process_file))

    def test_rejects_an_incomplete_camera_set(self):
        class RecordingProcessor(sensor_processing.MultiViewSensorProcessing):
            def process(self, _views):
                raise AssertionError("process should not be called")

        processor = RecordingProcessor(
            {"latent_size": 2, "num_views": 2, "image_size": [8, 8]}
        )
        with self.assertRaisesRegex(ValueError, "Expected 2 camera views"):
            processor.process_demonstration(object(), 7, ["camera-1"])


class TestEncoderSensorProcessing(unittest.TestCase):
    def test_loads_checkpoint_and_processes_with_encoder_method(self):
        class ToyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def encode(self, inputs):
                return inputs * self.scale

        class ToyProcessor(sensor_processing.SingleViewEncoderSensorProcessing):
            def __init__(self, exp):
                super().__init__(exp)
                self.enc = ToyEncoder()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "encoder.pth"
            source_encoder = ToyEncoder()
            source_encoder.scale.data.fill_(3.0)
            torch.save(source_encoder.state_dict(), checkpoint_path)

            exp = {
                "data_dir": temporary_directory,
                "proprioception_mlp_model_file": checkpoint_path.name,
                "latent_size": 1,
                "image_size": [8, 8],
            }
            processor = ToyProcessor(exp)

            loaded_path = processor.load_encoder_checkpoint(required=True)

            self.assertEqual(loaded_path, checkpoint_path)
            self.assertFalse(processor.enc.training)
            self.assertEqual(
                np.asarray(processor.process(torch.tensor([[2.0]]))).item(), 6.0
            )


class TestConcatConvVaeSensorProcessing(unittest.TestCase):
    @staticmethod
    def _processor(model, latent_size=2):
        processor = object.__new__(
            sp_conv_vae_concat_multiview.ConcatConvVaeSensorProcessing
        )
        processor.model = model
        processor.latent_size = latent_size
        processor.debug = False
        return processor

    def test_process_returns_the_vae_latent_mean(self):
        class Encoder:
            @staticmethod
            def encode(inputs):
                return torch.tensor(
                    [[1.0, 2.0]], device=inputs.device
                ), torch.zeros(1, 2, device=inputs.device)

        result = self._processor(Encoder()).process(torch.zeros(1, 3, 64, 64))

        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))

    def test_process_raises_instead_of_using_a_synthetic_latent(self):
        class BrokenEncoder:
            @staticmethod
            def encode(_inputs):
                raise RuntimeError("checkpoint does not match the architecture")

        processor = self._processor(BrokenEncoder())
        with self.assertRaisesRegex(
            RuntimeError, "Conv-VAE encoding failed for composite input"
        ) as error:
            processor.process(torch.zeros(1, 3, 64, 64))

        self.assertIsInstance(error.exception.__cause__, RuntimeError)

    def test_process_rejects_an_unexpected_latent_shape(self):
        class WrongShapeEncoder:
            @staticmethod
            def encode(inputs):
                return (torch.zeros(inputs.size(0), 3, device=inputs.device),)

        with self.assertRaisesRegex(ValueError, r"expected \(1, 2\)"):
            self._processor(WrongShapeEncoder()).process(
                torch.zeros(1, 3, 64, 64)
            )


class TestViTHelpers(unittest.TestCase):
    def test_shared_heads_and_image_preprocessing(self):
        exp = {"projection_hidden_dim": 6, "proprio_step_1": 5, "proprio_step_2": 4}
        projection, hidden_size = vit_helper.create_projection(8, 3, exp)
        proprioceptor = vit_helper.create_proprioceptor(3, 2, exp)
        normalize, resize = vit_helper.create_image_preprocessing([4, 6])

        self.assertEqual(hidden_size, 6)
        self.assertEqual(projection(torch.ones(2, 8)).shape, (2, 3))
        self.assertEqual(proprioceptor(torch.ones(2, 3)).shape, (2, 2))

        image = torch.ones(1, 3, 2, 3)
        resized = vit_helper.resize_if_needed(image, resize)
        self.assertEqual(tuple(resized.shape[-2:]), (4, 6))
        self.assertFalse(torch.equal(vit_helper.normalize_if_unit_interval(resized, normalize), resized))
        normalized = torch.full_like(resized, -2.0)
        self.assertIs(vit_helper.normalize_if_unit_interval(normalized, normalize), normalized)

    def test_rejects_unknown_vit_backbone(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ViT model type"):
            vit_helper.create_vit_backbone(
                {"vit_model": "not-a-vit", "vit_weights": "DEFAULT"}
            )


class TestProprioTunedCNNRegression(unittest.TestCase):
    def test_common_pipeline_encodes_and_predicts(self):
        class TinyRegression(sp_propriotuned_cnn._ProprioTunedCNNRegression):
            def create_feature_extractor(self):
                return torch.nn.Linear(2, 2, bias=False)

            def create_heads(self, _exp):
                self.reductor = torch.nn.Linear(2, 1, bias=False)
                self.proprioceptor = torch.nn.Linear(1, 1, bias=False)
                self.feature_extractor.weight.data.copy_(torch.eye(2))
                self.reductor.weight.data.copy_(torch.tensor([[2.0, 3.0]]))
                self.proprioceptor.weight.data.fill_(4.0)

            def encode_flat_features(self, flat_features):
                return self.reductor(flat_features)

            def predict_from_latent(self, latent):
                return self.proprioceptor(latent)

        frozen = TinyRegression(
            {"latent_size": 1, "output_size": 1, "freeze_feature_extractor": True}
        )
        trainable = TinyRegression(
            {"latent_size": 1, "output_size": 1, "freeze_feature_extractor": False}
        )

        inputs = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.equal(frozen.encode(inputs), torch.tensor([[8.0]])))
        self.assertTrue(torch.equal(frozen(inputs), torch.tensor([[32.0]])))
        self.assertFalse(frozen.feature_extractor.weight.requires_grad)
        self.assertTrue(trainable.feature_extractor.weight.requires_grad)


class TestSensorProcessingFactory(unittest.TestCase):
    def test_multiview_class_registry_and_unknown_processor(self):
        self.assertTrue(sp_factory.is_multiview_sp({"class": "Vit_multiview"}))
        self.assertTrue(sp_factory.is_multiview_sp({"class": "Vit", "num_views": 2}))
        self.assertFalse(sp_factory.is_multiview_sp({"class": "Vit"}))
        with self.assertRaisesRegex(Exception, "Unknown sensor processing class"):
            sp_factory.create_sp({"class": "unknown"})

    def test_multiview_cnn_factory_uses_the_configured_or_legacy_model(self):
        with patch.object(
            sp_factory.sp_propriotuned_cnn_multiview,
            "MultiViewCNNSensorProcessing",
        ) as processor_class:
            sp_factory.create_sp(
                {"class": "MultiViewCNN", "model": "MultiViewResNetModel"}
            )
            sp_factory.create_sp(
                {"class": "VGG19ProprioTunedSensorProcessing_multiview"}
            )

        self.assertEqual(processor_class.call_count, 2)
        self.assertEqual(
            processor_class.call_args_list[0].args[0]["model"], "MultiViewResNetModel"
        )
        self.assertEqual(
            processor_class.call_args_list[1].args[0]["model"], "MultiViewVGG19Model"
        )

    def test_singleview_cnn_factory_uses_the_configured_or_legacy_model(self):
        with patch.object(
            sp_factory.sp_propriotuned_cnn,
            "ProprioTunedCNNSensorProcessing",
        ) as processor_class:
            sp_factory.create_sp(
                {
                    "class": "ProprioTunedCNN",
                    "model": "ResNetProprioTunedRegression",
                }
            )
            sp_factory.create_sp(
                {"class": "VGG19ProprioTunedSensorProcessing"}
            )

        self.assertEqual(processor_class.call_count, 2)
        self.assertEqual(
            processor_class.call_args_list[0].args[0]["model"],
            "ResNetProprioTunedRegression",
        )
        self.assertEqual(
            processor_class.call_args_list[1].args[0]["model"],
            "VGG19ProprioTunedRegression",
        )

    def test_cnn_factory_accepts_experiment_objects_without_copying_them(self):
        generic_exp = Experiment(
            {
                "class": "ProprioTunedCNN",
                "model": "ResNetProprioTunedRegression",
            }
        )
        legacy_exp = Experiment(
            {
                "class": "VGG19ProprioTunedSensorProcessing_multiview",
                "model": "unchanged",
            }
        )

        with (
            patch.object(
                sp_factory.sp_propriotuned_cnn,
                "ProprioTunedCNNSensorProcessing",
            ) as singleview_processor,
            patch.object(
                sp_factory.sp_propriotuned_cnn_multiview,
                "MultiViewCNNSensorProcessing",
            ) as multiview_processor,
        ):
            sp_factory.create_sp(generic_exp)
            sp_factory.create_sp(legacy_exp)

        self.assertIs(singleview_processor.call_args.args[0], generic_exp)
        legacy_values = multiview_processor.call_args.args[0]
        self.assertIsInstance(legacy_values, dict)
        self.assertEqual(legacy_values["model"], "MultiViewVGG19Model")
        self.assertEqual(legacy_exp["model"], "unchanged")


class TestHelperTrainingData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.exp = {
            "data_dir": str(self.data_dir),
            "proprioception_input_file": "inputs.pt",
            "proprioception_target_file": "targets.pt",
            "training_data": [["demo-run", "demo-name", "camera-1"]],
            "image_size": [8, 8],
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_complete_cache_without_accessing_demonstrations(self):
        inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        targets = torch.arange(5, dtype=torch.float32).reshape(5, 1)
        torch.save(inputs, self.data_dir / "inputs.pt")
        torch.save(targets, self.data_dir / "targets.pt")

        with patch.object(
            helper_training_data.Config,
            "get_experiment",
            side_effect=AssertionError(
                "complete cache should avoid demonstration loading"
            ),
        ):
            result = helper_training_data.load_images_as_proprioception_training(
                self.exp, {}, generator=torch.Generator().manual_seed(7)
            )

        self.assertTrue(torch.equal(result["inputs"], inputs))
        self.assertTrue(torch.equal(result["targets"], targets))
        self.assertEqual(len(result["inputs_training"]), 3)
        self.assertEqual(len(result["inputs_validation"]), 2)
        self.assertEqual(
            len(result["inputs_training"]) + len(result["inputs_validation"]),
            5,
        )

    def test_rebuilds_both_tensors_when_cache_is_incomplete(self):
        torch.save(torch.full((2, 1), -1.0), self.data_dir / "inputs.pt")
        robot_exp = {"robot": "configuration"}

        class FakePosition:
            def __init__(self, index):
                self.index = index

            def to_normalized_vector(self, received_robot_exp):
                assert received_robot_exp is robot_exp
                return np.array(
                    [self.index, self.index + 0.5], dtype=np.float32
                )

        class FakeDemonstration:
            metadata = {"maxsteps": 4}

            def __init__(self, exp_demo, demo_name):
                assert exp_demo == {
                    "experiment": "demonstration",
                    "run": "demo-run",
                }
                assert demo_name == "demo-name"

            def get_image(self, index, transform, camera):
                assert transform == "transform"
                assert camera == "camera-1"
                return torch.tensor([[float(index)]]), None

            def get_action(self, index, action_name, received_robot_exp):
                assert action_name == "rc-position-target"
                assert received_robot_exp is robot_exp
                return FakePosition(index)

        def fake_get_experiment(_config, experiment, run):
            return {"experiment": experiment, "run": run}

        with (
            patch.object(
                helper_training_data.Config,
                "get_experiment",
                new=fake_get_experiment,
            ),
            patch.object(
                helper_training_data,
                "get_transform_to_sp",
                return_value="transform",
            ),
            patch.object(
                helper_training_data,
                "Demonstration",
                FakeDemonstration,
            ),
        ):
            result = helper_training_data.load_images_as_proprioception_training(
                self.exp,
                robot_exp,
                generator=torch.Generator().manual_seed(2),
            )

        self.assertEqual(result["inputs"].shape, (4, 1))
        self.assertEqual(result["targets"].shape, (4, 2))
        self.assertEqual(len(result["inputs_training"]), 2)
        self.assertEqual(len(result["inputs_validation"]), 2)
        for input_split, target_split in (
            (result["inputs_training"], result["targets_training"]),
            (result["inputs_validation"], result["targets_validation"]),
        ):
            self.assertTrue(
                torch.equal(input_split.flatten(), target_split[:, 0])
            )
        self.assertTrue(
            torch.equal(
                torch.load(self.data_dir / "inputs.pt", weights_only=True),
                result["inputs"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.load(self.data_dir / "targets.pt", weights_only=True),
                result["targets"],
            )
        )

    def test_split_is_reproducible_and_preserves_every_sample(self):
        inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        targets = torch.arange(10, dtype=torch.float32).reshape(10, 1)
        torch.save(inputs, self.data_dir / "inputs.pt")
        torch.save(targets, self.data_dir / "targets.pt")

        first = helper_training_data.load_images_as_proprioception_training(
            self.exp, {}, generator=torch.Generator().manual_seed(19)
        )
        second = helper_training_data.load_images_as_proprioception_training(
            self.exp, {}, generator=torch.Generator().manual_seed(19)
        )

        self.assertTrue(
            torch.equal(first["inputs_training"], second["inputs_training"])
        )
        combined_targets = torch.cat(
            [first["targets_training"], first["targets_validation"]]
        )
        self.assertEqual(
            sorted(combined_targets.flatten().tolist()), list(range(10))
        )

    def test_rejects_mismatched_cached_sample_counts(self):
        torch.save(torch.zeros((3, 1)), self.data_dir / "inputs.pt")
        torch.save(torch.zeros((2, 1)), self.data_dir / "targets.pt")

        with self.assertRaisesRegex(ValueError, "different sample counts"):
            helper_training_data.load_images_as_proprioception_training(
                self.exp, {}
            )

    def test_rejects_invalid_training_fraction(self):
        for fraction in [0.0, 1.0, -0.1, 1.1]:
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "training_fraction"):
                    helper_training_data.load_images_as_proprioception_training(
                        self.exp, {}, training_fraction=fraction
                    )


class TestMultiviewHelperTrainingData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.robot_exp = {"robot": "configuration"}
        self.exp = {
            "data_dir": str(self.data_dir),
            "proprioception_input_file": "train-inputs.pt",
            "proprioception_target_file": "train-targets.pt",
            "proprioception_test_input_file": "validation-inputs.pt",
            "proprioception_test_target_file": "validation-targets.pt",
            "training_data": [
                ["demo-run", "training-demo", ["camera-2", "camera-1"]]
            ],
            "validation_data": [
                ["demo-run", "validation-demo", ["camera-2", "camera-1"]]
            ],
            "image_size": [8, 8],
            "num_views": 2,
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _patch_demonstrations(self):
        robot_exp = self.robot_exp

        class FakePosition:
            def __init__(self, index):
                self.index = index

            def to_normalized_vector(self, received_robot_exp):
                assert received_robot_exp is robot_exp
                return np.array(
                    [self.index, self.index + 0.5], dtype=np.float32
                )

        class FakeDemonstration:
            def __init__(self, _exp_demo, demo_name):
                self.demo_name = demo_name
                self.metadata = {
                    "cameras": ["camera-1", "camera-2"],
                    "maxsteps": 2 if demo_name == "training-demo" else 1,
                }

            def get_image(self, index, transform, camera):
                assert transform == "transform"
                camera_value = 20 if camera == "camera-2" else 10
                return torch.tensor([[camera_value + index]], dtype=torch.float32), None

            def get_action(self, index, action_name, received_robot_exp):
                assert action_name == "rc-position-target"
                assert received_robot_exp is robot_exp
                return FakePosition(index)

        def fake_get_experiment(_config, experiment, run):
            return {"experiment": experiment, "run": run}

        return (
            patch.object(
                helper_training_data.Config,
                "get_experiment",
                new=fake_get_experiment,
            ),
            patch.object(
                helper_training_data,
                "get_transform_to_sp",
                return_value="transform",
            ),
            patch.object(
                helper_training_data,
                "Demonstration",
                FakeDemonstration,
            ),
        )

    def test_builds_ordered_training_and_explicit_validation_sets(self):
        config_patch, transform_patch, demonstration_patch = (
            self._patch_demonstrations()
        )
        with config_patch, transform_patch, demonstration_patch:
            result = (
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp, self.robot_exp
                )
            )

        self.assertEqual(len(result["view_inputs_training"]), 2)
        self.assertEqual(len(result["targets_training"]), 2)
        self.assertEqual(len(result["targets_validation"]), 1)
        self.assertEqual(
            result["view_inputs_training"][0].flatten().tolist(),
            [20.0, 21.0],
        )
        self.assertEqual(
            result["view_inputs_training"][1].flatten().tolist(),
            [10.0, 11.0],
        )
        for filename in (
            "train-inputs.pt",
            "train-inputs.pt.manifest.json",
            "train-targets.pt",
            "validation-inputs.pt",
            "validation-inputs.pt.manifest.json",
            "validation-targets.pt",
        ):
            self.assertTrue((self.data_dir / filename).exists())

    def test_cached_training_data_can_use_reproducible_fallback_split(self):
        self.exp.pop("validation_data")
        views = [
            torch.arange(6, dtype=torch.float32).reshape(6, 1),
            torch.arange(10, 16, dtype=torch.float32).reshape(6, 1),
        ]
        targets = torch.arange(6, dtype=torch.float32).reshape(6, 1)
        torch.save(views, self.data_dir / "train-inputs.pt")
        torch.save(targets, self.data_dir / "train-targets.pt")
        manifest = helper_training_data._multiview_cache_manifest(
            self.exp, "training_data"
        )
        with (self.data_dir / "train-inputs.pt.manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle)

        with patch.object(
            helper_training_data,
            "get_transform_to_sp",
            return_value="unused",
        ):
            first = (
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp,
                    self.robot_exp,
                    generator=torch.Generator().manual_seed(17),
                )
            )
            second = (
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp,
                    self.robot_exp,
                    generator=torch.Generator().manual_seed(17),
                )
            )

        self.assertTrue(
            torch.equal(
                first["targets_training"], second["targets_training"]
            )
        )
        for view, target_offset in zip(
            first["view_inputs_training"], (0, 10)
        ):
            self.assertTrue(
                torch.equal(
                    view.flatten() - target_offset,
                    first["targets_training"].flatten(),
                )
            )

    def test_rebuilds_legacy_cache_without_manifest(self):
        torch.save(
            [torch.full((1, 1), -1.0), torch.full((1, 1), -1.0)],
            self.data_dir / "train-inputs.pt",
        )
        torch.save(
            torch.full((1, 2), -1.0), self.data_dir / "train-targets.pt"
        )

        config_patch, transform_patch, demonstration_patch = (
            self._patch_demonstrations()
        )
        with config_patch, transform_patch, demonstration_patch:
            result = (
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp, self.robot_exp
                )
            )

        self.assertEqual(len(result["targets_training"]), 2)
        self.assertNotEqual(
            result["view_inputs_training"][0][0].item(), -1.0
        )

    def test_requires_an_ordered_camera_list(self):
        self.exp["training_data"][0][2] = "camera-2,camera-1"
        with patch.object(
            helper_training_data,
            "get_transform_to_sp",
            return_value="transform",
        ):
            with self.assertRaisesRegex(ValueError, "ordered list"):
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp, self.robot_exp
                )

    def test_rejects_cached_view_count_mismatch(self):
        torch.save(
            [torch.zeros((2, 1))], self.data_dir / "train-inputs.pt"
        )
        torch.save(
            torch.zeros((2, 1)), self.data_dir / "train-targets.pt"
        )
        manifest = helper_training_data._multiview_cache_manifest(
            self.exp, "training_data"
        )
        with (self.data_dir / "train-inputs.pt.manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle)
        with patch.object(
            helper_training_data,
            "get_transform_to_sp",
            return_value="unused",
        ):
            with self.assertRaisesRegex(ValueError, "expected 2"):
                helper_training_data.load_multiview_images_as_proprioception_training(
                    self.exp, self.robot_exp
                )


if __name__ == "__main__":
    unittest.main()
