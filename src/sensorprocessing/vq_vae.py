"""Single-view autoencoder with a shared codebook for grouped image latents."""

import torch
from torch import nn
from torch.nn import functional as F

from .conv_vae_neo import ConvAutoencoderBase, train_autoencoder


class VQVAE(ConvAutoencoderBase):
    def _build_latent(self, exp):
        self.num_groups = exp["num_codes_per_image"]
        self.projection = nn.Linear(self.flatten_size, self.latent_size)
        self.codebook = nn.Embedding(exp["codebook_size"], self.latent_size // self.num_groups)

    def quantize(self, images):
        """Return projected vectors, nearest embeddings, and their IDs per group."""
        features = self.encoder(images).flatten(1)
        vectors = self.projection(features).reshape(
            len(images), self.num_groups, self.codebook.embedding_dim
        )
        # The squared norm of each query is constant across candidate codes.
        weights = self.codebook.weight
        distances = weights.square().sum(1) - 2 * (vectors @ weights.T)
        ids = distances.argmin(dim=-1)
        return vectors, self.codebook(ids), ids

    def encode(self, images):
        _, embeddings, _ = self.quantize(images)
        return embeddings.flatten(1)

    def forward(self, images):
        vectors, embeddings, ids = self.quantize(images)
        straight_through = vectors + (embeddings - vectors).detach()
        return self.decode(straight_through.flatten(1)), vectors, embeddings, ids


class VQVAELoss(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.commitment_weight = exp["commitment_weight"]

    def forward(self, output, images):
        reconstruction, vectors, embeddings, _ = output
        return (
            F.mse_loss(reconstruction, images)
            + F.mse_loss(embeddings, vectors.detach())
            + self.commitment_weight * F.mse_loss(vectors, embeddings.detach())
        )


def train(exp, *, epochs=None):
    return train_autoencoder(exp, VQVAE, VQVAELoss, epochs=epochs)
