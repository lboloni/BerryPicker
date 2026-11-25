"""Check the assignment and gradient paths specific to vector quantization."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from sensorprocessing.vq_vae import VQVAE, VQVAELoss


def test_group_assignments_and_training_gradients():
    exp = dict(image_size=[16, 16], latent_size=8, base_channels=4,
               num_codes_per_image=4, codebook_size=16, commitment_weight=0.25)
    model = VQVAE(exp)
    images = torch.rand(3, 3, 16, 16)
    output = model(images)
    reconstruction, vectors, embeddings, ids = output
    distances = (vectors.unsqueeze(2) - model.codebook.weight).square().sum(-1)
    assert ids.shape == (3, 4)
    assert torch.equal(ids, distances.argmin(2))
    assert torch.equal(model.encode(images), embeddings.flatten(1))

    reconstruction.mean().backward(retain_graph=True)
    assert model.projection.weight.grad.abs().sum() > 0
    assert model.codebook.weight.grad is None

    model.zero_grad()
    VQVAELoss(exp)(output, images).backward()
    assert model.codebook.weight.grad.abs().sum() > 0
