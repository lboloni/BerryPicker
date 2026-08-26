"""
test_multiview.py

Tests for the shared multi-view fusion head, the backbone container and the
three multi-view encoders (ViT, CNN, conv encoder). Pretrained torchvision
constructors are replaced by tiny stand-ins so no weights are downloaded.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sensorprocessing import (
    multiview_backbones,
    multiview_fusion,
    sp_conv_vae_multiview,
    sp_factory,
    sp_propriotuned_cnn_multiview,
    sp_vit_multiview,
    vit_helper,
)
from sensorprocessing.helper_training_data import (
    MultiViewDataset,
    collate_multiview,
    make_multiview_loaders,
)


FUSIONS = list(multiview_fusion.FUSION_TYPES)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 4, 4)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(16, 24)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def fake_vit(exp):
    return TinyViT(), 24


class TinyVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 4, 4), nn.ReLU(), nn.AdaptiveAvgPool2d(2))


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, 4, 4), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(4, 1)

    def children(self):
        return iter([self.stem, self.fc])


def base_exp(**overrides):
    exp = {
        "latent_size": 8,
        "output_size": 6,
        "num_views": 2,
        "cameras": ["dev2", "dev3"],
        "image_size": [32, 32],
        "vit_model": "vit_b_16",
        "vit_weights": "DEFAULT",
        "freeze_feature_extractor": False,
    }
    exp.update(overrides)
    return exp


def views(batch=4, size=32):
    return [torch.rand(batch, 3, size, size) for _ in range(2)]


class TestMultiViewFusion(unittest.TestCase):
    def test_every_fusion_type_maps_view_features_to_the_latent(self):
        for fusion_type in FUSIONS:
            with self.subTest(fusion=fusion_type):
                head = multiview_fusion.MultiViewFusion(12, 2, 5, fusion_type)
                head.train()
                features = [torch.randn(6, 12), torch.randn(6, 12)]
                latent = head(features)
                self.assertEqual(latent.shape, (6, 5))
                latent.sum().backward()
                head.eval()
                with torch.no_grad():
                    self.assertEqual(head([f[:1] for f in features]).shape, (1, 5))

    def test_rejects_wrong_view_count_and_width(self):
        head = multiview_fusion.MultiViewFusion(12, 2, 5, "concat_proj")
        with self.assertRaisesRegex(ValueError, "Expected 2 view feature tensors"):
            head([torch.randn(2, 12)])
        with self.assertRaisesRegex(ValueError, "must have shape"):
            head([torch.randn(2, 12), torch.randn(2, 7)])
        with self.assertRaisesRegex(ValueError, "Unknown fusion type"):
            multiview_fusion.MultiViewFusion(12, 2, 5, "average")

    def test_view_scores_sum_to_one_for_weighted_and_gated_heads(self):
        features = [torch.randn(3, 12), torch.randn(3, 12)]
        for fusion_type in ("weighted_sum", "gated"):
            head = multiview_fusion.MultiViewFusion(12, 2, 5, fusion_type).eval()
            scores = head.view_scores(features)
            self.assertEqual(scores.shape, (3, 2))
            self.assertTrue(torch.allclose(scores.sum(dim=1), torch.ones(3), atol=1e-5))
        self.assertIsNone(
            multiview_fusion.MultiViewFusion(12, 2, 5, "concat_proj").view_scores(features)
        )

    def test_attention_view_embedding_breaks_view_permutation_symmetry(self):
        torch.manual_seed(0)
        head = multiview_fusion.MultiViewFusion(12, 2, 5, "attention").eval()
        a, b = torch.randn(2, 12), torch.randn(2, 12)
        with torch.no_grad():
            self.assertFalse(torch.allclose(head([a, b]), head([b, a])))
        symmetric = multiview_fusion.MultiViewFusion(
            12, 2, 5, "attention", view_embedding=False
        ).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(symmetric([a, b]), symmetric([b, a]), atol=1e-5))

    def test_fusion_from_exp_reads_the_configuration_keys(self):
        head = multiview_fusion.fusion_from_exp(
            {"fusion_type": "attention", "attention_heads": 2, "fusion_dropout": 0.0},
            12, 2, 5,
        )
        self.assertEqual(head.attention.num_heads, 2)
        self.assertEqual(head.fusion_type, "attention")
        self.assertEqual(multiview_fusion.default_attention_heads(768), 8)
        self.assertEqual(multiview_fusion.default_attention_heads(6), 2)


class TestViewBackbones(unittest.TestCase):
    def test_shared_batched_forward_matches_per_view_forward(self):
        torch.manual_seed(0)
        container = multiview_backbones.ViewBackbones(TinyViT, 2, shared=True).eval()
        inputs = views()
        with torch.no_grad():
            batched = container(inputs)
            container.batched = False
            separate = container(inputs)
        for x, y in zip(batched, separate):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_separate_backbones_are_distinct_modules(self):
        container = multiview_backbones.ViewBackbones(TinyViT, 2, shared=False)
        self.assertIsNone(container.backbone)
        self.assertEqual(len(container.backbones), 2)
        self.assertIsNot(container.backbones[0], container.backbones[1])
        with self.assertRaisesRegex(ValueError, "Expected 2 views"):
            container([torch.rand(1, 3, 32, 32)])

    def test_frozen_backbone_stays_in_eval_mode_during_training(self):
        container = multiview_backbones.ViewBackbones(
            lambda: nn.Sequential(nn.Conv2d(3, 2, 3), nn.BatchNorm2d(2)), 2, shared=True, freeze=True
        )
        container.train()
        batch_norm = container.backbone[1]
        self.assertFalse(batch_norm.training)
        self.assertFalse(any(p.requires_grad for p in container.parameters()))
        self.assertEqual(container.trainable_parameters(), [])


class TestMultiViewEncoders(unittest.TestCase):
    def _check_encoder(self, model, image_size=32):
        inputs = views(size=image_size)
        model.train()
        output = model(inputs)
        latent = model.encode_views(inputs)
        self.assertEqual(output.shape, (4, 6))
        self.assertEqual(latent.shape, (4, 8))
        output.sum().backward()
        model.eval()
        with torch.no_grad():
            self.assertEqual(model.encode_views([v[:1] for v in inputs]).shape, (1, 8))
        with self.assertRaisesRegex(ValueError, "Expected 2 views"):
            model.encode_views(inputs[:1])

    def test_vit_encoder_all_fusions_shared_and_separate(self):
        with patch.object(vit_helper, "create_vit_backbone", fake_vit), patch.object(
            sp_vit_multiview, "create_vit_backbone", fake_vit
        ):
            for fusion_type in FUSIONS:
                for shared in (True, False):
                    with self.subTest(fusion=fusion_type, shared=shared):
                        model = sp_vit_multiview.MultiViewViTEncoder(
                            base_exp(fusion_type=fusion_type, shared_backbone=shared)
                        )
                        self._check_encoder(model)
                        self.assertEqual(model.encode(views()).shape, (4, 8))

    def test_cnn_encoders_derive_the_feature_width_from_the_image_size(self):
        with patch.object(models, "vgg19", lambda weights=None: TinyVGG()), patch.object(
            models, "resnet50", lambda weights=None: TinyResNet()
        ):
            vgg = sp_propriotuned_cnn_multiview.MultiViewVGG19Model(
                base_exp(fusion_type="gated", image_size=[48, 48])
            )
            self.assertEqual(vgg.feature_size, 4 * 2 * 2)
            self._check_encoder(vgg, image_size=48)

            resnet = sp_propriotuned_cnn_multiview.MultiViewResNetModel(
                base_exp(fusion_type="attention", freeze_feature_extractor=True)
            )
            self.assertEqual(resnet.feature_size, 4)
            resnet.train()
            self.assertFalse(resnet.backbones.backbone[0][1].training)  # frozen BN stays eval
            self._check_encoder(resnet)

            with self.assertRaisesRegex(ValueError, "Unknown multi-view CNN model"):
                sp_propriotuned_cnn_multiview.create_multiview_cnn_model(
                    base_exp(model="MultiViewDenseNet")
                )
            legacy = sp_propriotuned_cnn_multiview.create_multiview_cnn_model(
                base_exp(model="ResNetProprioTunedRegression_multiview")
            )
            self.assertIsInstance(legacy, sp_propriotuned_cnn_multiview.MultiViewResNetModel)

    def test_conv_encoder_works_at_several_image_sizes(self):
        for size in (32, 64):
            with self.subTest(size=size):
                model = sp_conv_vae_multiview.MultiViewConvVAEModel(
                    base_exp(fusion_type="indiv_proj", image_size=[size, size], encoder_channels=[4, 8])
                )
                self.assertEqual(model.backbones.backbone.flatten_size, 8 * (size // 4) ** 2)
                self._check_encoder(model, image_size=size)


class TestMultiViewSensorProcessingWrappers(unittest.TestCase):
    def test_wrapper_loads_trained_weights_and_processes_views(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            vit_helper, "create_vit_backbone", fake_vit
        ), patch.object(sp_vit_multiview, "create_vit_backbone", fake_vit):
            exp = base_exp(
                data_dir=directory,
                proprioception_mlp_model_file="encoder.pth",
                fusion_type="weighted_sum",
            )
            trained = sp_vit_multiview.MultiViewViTEncoder(exp)
            torch.save({"model_state_dict": trained.state_dict()}, Path(directory) / "encoder.pth")

            processor = sp_vit_multiview.MultiViewVitSensorProcessing(exp)
            self.assertEqual(processor.cameras, ["dev2", "dev3"])
            self.assertFalse(processor.enc.training)
            inputs = [v[:1] for v in views()]
            latent = processor.process(inputs)
            self.assertEqual(np.asarray(latent).shape, (8,))
            trained.eval()
            with torch.no_grad():
                expected = trained.encode(inputs).squeeze(0).numpy()
            self.assertTrue(np.allclose(latent, expected, atol=1e-6))

    def test_camera_order_mismatch_is_reported(self):
        with patch.object(vit_helper, "create_vit_backbone", fake_vit), patch.object(
            sp_vit_multiview, "create_vit_backbone", fake_vit
        ), tempfile.TemporaryDirectory() as directory:
            processor = sp_vit_multiview.MultiViewVitSensorProcessing(
                base_exp(data_dir=directory, proprioception_mlp_model_file="none.pth")
            )

            class FakeDemonstration:
                def get_image(self, timestep, camera, transform):
                    return torch.rand(1, 3, 32, 32), None

            with patch("builtins.print") as printed:
                processor.process_demonstration(FakeDemonstration(), 0, ["dev3", "dev2"])
            messages = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
            self.assertIn("camera order", messages)

    def test_factory_registers_the_conv_encoder_processor(self):
        with patch.object(
            sp_factory.sp_conv_vae_multiview, "MultiViewConvVAESensorProcessing"
        ) as processor_class:
            sp_factory.create_sp({"class": "ConvVaeSensorProcessing_multiview"})
            sp_factory.create_sp({"class": "MultiViewConvVAESensorProcessing"})
        self.assertEqual(processor_class.call_count, 2)
        self.assertTrue(sp_factory.is_multiview_sp({"class": "MultiViewConvVAESensorProcessing"}))


class TestMultiViewDataHelpers(unittest.TestCase):
    def test_dataset_collate_and_loaders(self):
        tr = {
            "view_inputs_training": [torch.arange(10.0).view(10, 1), torch.arange(10.0, 20.0).view(10, 1)],
            "targets_training": torch.arange(10.0).view(10, 1),
            "view_inputs_validation": [torch.zeros(3, 1), torch.ones(3, 1)],
            "targets_validation": torch.zeros(3, 1),
        }
        dataset = MultiViewDataset(tr["view_inputs_training"], tr["targets_training"])
        item_views, item_target = dataset[3]
        self.assertEqual([v.item() for v in item_views], [3.0, 13.0])
        self.assertEqual(item_target.item(), 3.0)

        batched_views, targets = collate_multiview([dataset[0], dataset[1]])
        self.assertEqual(len(batched_views), 2)
        self.assertEqual(batched_views[1].tolist(), [[10.0], [11.0]])
        self.assertEqual(targets.tolist(), [[0.0], [1.0]])

        train_loader, validation_loader = make_multiview_loaders(tr, batch_size=4, shuffle=False)
        self.assertEqual(len(train_loader), 2)  # drop_last: 10 samples, batch 4 -> 2 full batches
        self.assertEqual(len(validation_loader), 1)
        with self.assertRaisesRegex(ValueError, "targets have"):
            MultiViewDataset([torch.zeros(2, 1), torch.zeros(3, 1)], torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
