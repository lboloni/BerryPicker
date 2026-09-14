"""Tests for training-free random-projection CNN sensor processing."""

import math
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
from sensorprocessing import sp_factory
from sensorprocessing.sp_random_projection_cnn import (
    RandomProjectionCNNSensorProcessing,
    _RandomProjectionCNN,
)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images):
        return images * self.scale


class TinyRandomProjection(_RandomProjectionCNN):
    def create_feature_extractor(self, _weights):
        return TinyFeatures()


def model_exp(**overrides):
    exp = {
        "image_size": [4, 6],
        "latent_size": 5,
        "backbone_weights": "NONE",
        "feature_pool_size": [2, 3],
        "feature_normalization": "l2",
        "projection": "rademacher",
        "projection_seed": 73017,
    }
    exp.update(overrides)
    return exp


class TestRandomProjectionCNN(unittest.TestCase):
    def test_derives_feature_width_and_returns_configured_latent_shape(self):
        with patch.dict(Config().runtime, {"device": "cpu"}):
            model = TinyRandomProjection(model_exp())
            latent = model.encode(torch.rand(2, 3, 4, 6))

        self.assertEqual(model.feature_size, 3 * 2 * 3)
        self.assertEqual(model.projection.shape, (5, 18))
        self.assertEqual(latent.shape, (2, 5))
        self.assertTrue(torch.isfinite(latent).all())

    def test_rademacher_projection_is_seeded_scaled_and_not_trainable(self):
        with patch.dict(Config().runtime, {"device": "cpu"}):
            torch.manual_seed(1)
            first = TinyRandomProjection(model_exp())
            torch.manual_seed(999)
            second = TinyRandomProjection(model_exp())
            different = TinyRandomProjection(model_exp(projection_seed=73018))

        self.assertTrue(torch.equal(first.projection, second.projection))
        self.assertFalse(torch.equal(first.projection, different.projection))
        expected = torch.tensor(1.0 / math.sqrt(first.latent_size))
        self.assertTrue(torch.allclose(first.projection.abs(), expected))
        self.assertIn("projection", dict(first.named_buffers()))
        self.assertNotIn("projection", dict(first.named_parameters()))
        self.assertFalse(
            any(p.requires_grad for p in first.feature_extractor.parameters())
        )

    def test_frozen_backbone_remains_in_evaluation_mode(self):
        with patch.dict(Config().runtime, {"device": "cpu"}):
            model = TinyRandomProjection(model_exp())
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.feature_extractor.training)

    def test_l2_and_none_feature_normalization(self):
        image = torch.tensor([[[[3.0]], [[4.0]], [[0.0]]]])
        common = dict(image_size=[1, 1], latent_size=3, feature_pool_size=[1, 1])
        with patch.dict(Config().runtime, {"device": "cpu"}):
            normalized = TinyRandomProjection(
                model_exp(feature_normalization="l2", **common)
            )
            unnormalized = TinyRandomProjection(
                model_exp(feature_normalization="none", **common)
            )
        normalized.projection.copy_(torch.eye(3))
        unnormalized.projection.copy_(torch.eye(3))

        self.assertTrue(
            torch.allclose(normalized.encode(image), torch.tensor([[0.6, 0.8, 0.0]]))
        )
        self.assertTrue(
            torch.equal(unnormalized.encode(image), torch.tensor([[6.0, 8.0, 0.0]]))
        )

    def test_gaussian_projection_is_repeatable(self):
        exp = model_exp(projection="gaussian")
        with patch.dict(Config().runtime, {"device": "cpu"}):
            first = TinyRandomProjection(exp)
            second = TinyRandomProjection(exp)
        self.assertTrue(torch.equal(first.projection, second.projection))
        self.assertTrue(torch.isfinite(first.projection).all())


class TinyEncoder(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]

    def encode(self, images):
        return torch.zeros(images.shape[0], self.latent_size)


class TestRandomProjectionCNNIntegration(unittest.TestCase):
    def test_factory_constructs_processor_without_loading_a_checkpoint(self):
        exp = {
            "class": "RandomProjectionCNNSensorProcessing",
            "model": "TinyEncoder",
            "image_size": [4, 4],
            "latent_size": 7,
        }
        with patch.dict(
            RandomProjectionCNNSensorProcessing.encoder_classes,
            {"TinyEncoder": TinyEncoder},
        ), patch.dict(Config().runtime, {"device": "cpu"}), patch.object(
            RandomProjectionCNNSensorProcessing,
            "load_encoder_checkpoint",
            side_effect=AssertionError("checkpoint loading is not part of this SP"),
        ):
            processor = sp_factory.create_sp(exp)
            latent = processor.process(torch.rand(1, 3, 4, 4))

        self.assertIsInstance(processor, RandomProjectionCNNSensorProcessing)
        self.assertEqual(latent.shape, (7,))
        self.assertTrue(np.all(np.isfinite(latent)))


if __name__ == "__main__":
    unittest.main()
