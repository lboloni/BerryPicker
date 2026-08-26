"""
multiview_fusion.py

One implementation of the multi-view fusion heads, shared by every
multi-view sensor-processing backbone (ViT, propriotuned CNN, Conv encoder).

A fusion head receives a list of per-view feature vectors, one tensor of shape
``[batch, feature_dim]`` per camera, in the camera order the model was trained
with, and returns one latent tensor of shape ``[batch, latent_size]``.

Fusion types (``exp["fusion_type"]``):

- ``concat_proj``   concatenate the view features and project them with a
                    three-layer MLP whose widths shrink gradually
                    (``V*D -> V*D/2 -> V*D/4 -> latent``).
- ``indiv_proj``    project every view to the latent size with its own head,
                    concatenate the per-view latents and fuse them with a
                    small MLP.
- ``attention``     treat the views as a token sequence, add a learned view
                    embedding (so the head knows which camera a token came
                    from), run multi-head self-attention with a residual
                    connection and LayerNorm, mean-pool over the views and
                    project to the latent size.
- ``weighted_sum``  per-view projection to the latent size followed by a
                    softmax-normalised, learned, input-independent weighting.
- ``gated``         per-view projection to the latent size followed by an
                    input-dependent gate (softmax over views) computed from
                    the concatenated view features.

The widths are derived from ``feature_dim`` so the same head is used at every
backbone width, which keeps the backbone comparison clean: only the backbone
changes between the ViT, CNN and Conv encoder multi-view models.

Configuration keys read by :func:`fusion_from_exp` (all optional):

- ``fusion_type``        one of the five names above (default ``concat_proj``)
- ``fusion_hidden_dim``  first hidden width of ``concat_proj`` / per-view
                         projection width; defaults derive from ``feature_dim``
- ``fusion_dropout``     dropout inside the heads (default 0.1)
- ``attention_heads``    number of attention heads (default: the largest of
                         8, 4, 2, 1 that divides ``feature_dim``)
- ``attention_residual`` residual + LayerNorm around the attention (default on)
- ``view_embedding``     learned per-view embedding for attention (default on)
"""

import torch
import torch.nn as nn


FUSION_TYPES = ("concat_proj", "indiv_proj", "attention", "weighted_sum", "gated")


def _mlp(in_dim, hidden_dim, out_dim, dropout):
    """``in -> hidden -> out`` with BatchNorm, ReLU and dropout in between."""
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def default_attention_heads(feature_dim):
    """Largest of 8, 4, 2, 1 that divides ``feature_dim``."""
    for heads in (8, 4, 2, 1):
        if feature_dim % heads == 0:
            return heads
    return 1


class MultiViewFusion(nn.Module):
    """Fuse ordered per-view features into one latent vector."""

    def __init__(
        self,
        feature_dim,
        num_views,
        latent_size,
        fusion_type="concat_proj",
        *,
        hidden_dim=None,
        dropout=0.1,
        attention_heads=None,
        attention_residual=True,
        view_embedding=True,
    ):
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(
                f"Unknown fusion type: {fusion_type!r}. "
                f"Must be one of: {', '.join(FUSION_TYPES)}"
            )
        if num_views < 1:
            raise ValueError("num_views must be at least 1")

        self.feature_dim = feature_dim
        self.num_views = num_views
        self.latent_size = latent_size
        self.fusion_type = fusion_type
        self.dropout = dropout
        concat_dim = feature_dim * num_views

        if fusion_type == "concat_proj":
            hidden_1 = hidden_dim or max(concat_dim // 2, latent_size)
            hidden_2 = max(hidden_1 // 2, latent_size)
            if hidden_1 < concat_dim // 4:
                print(
                    f"WARNING: fusion_hidden_dim={hidden_1} is small for a "
                    f"{concat_dim}-wide concatenated input"
                )
            layers = [
                nn.Linear(concat_dim, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.BatchNorm1d(hidden_2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_2, latent_size),
            ]
            self.projection = nn.Sequential(*layers)
            self.description = (
                f"concat_proj: {concat_dim} -> {hidden_1} -> {hidden_2} -> {latent_size}"
            )

        elif fusion_type in ("indiv_proj", "weighted_sum", "gated"):
            view_hidden = hidden_dim or max(feature_dim // 2, latent_size)
            self.view_projections = nn.ModuleList(
                _mlp(feature_dim, view_hidden, latent_size, dropout)
                for _ in range(num_views)
            )
            per_view = f"{feature_dim} -> {view_hidden} -> {latent_size}"

            if fusion_type == "indiv_proj":
                fusion_input = latent_size * num_views
                fusion_hidden = max(fusion_input // 2, latent_size)
                self.fusion_layer = nn.Sequential(
                    nn.Linear(fusion_input, fusion_hidden),
                    nn.BatchNorm1d(fusion_hidden),
                    nn.ReLU(),
                    nn.Linear(fusion_hidden, latent_size),
                )
                self.description = (
                    f"indiv_proj: per view {per_view}; fusion "
                    f"{fusion_input} -> {fusion_hidden} -> {latent_size}"
                )
            elif fusion_type == "weighted_sum":
                self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
                self.description = (
                    f"weighted_sum: per view {per_view}; learned softmax weights"
                )
            else:  # gated
                gate_hidden = max(concat_dim // 4, num_views)
                self.gate_network = nn.Sequential(
                    nn.Linear(concat_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden, num_views),
                    nn.Softmax(dim=1),
                )
                self.description = (
                    f"gated: per view {per_view}; gate "
                    f"{concat_dim} -> {gate_hidden} -> {num_views}"
                )

        else:  # attention
            heads = attention_heads or default_attention_heads(feature_dim)
            if feature_dim % heads != 0:
                raise ValueError(
                    f"attention_heads={heads} must divide feature_dim={feature_dim}"
                )
            self.attention_residual = attention_residual
            self.view_embedding = (
                nn.Parameter(torch.zeros(num_views, feature_dim))
                if view_embedding
                else None
            )
            if self.view_embedding is not None:
                nn.init.normal_(self.view_embedding, std=0.02)
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
            final_hidden = hidden_dim or max(feature_dim // 2, latent_size)
            self.final_projection = _mlp(feature_dim, final_hidden, latent_size, dropout)
            self.description = (
                f"attention: {heads} heads over {num_views} view tokens of "
                f"width {feature_dim}"
                f"{' (+view embedding)' if view_embedding else ''}"
                f"{' (+residual/LayerNorm)' if attention_residual else ''}; "
                f"final {feature_dim} -> {final_hidden} -> {latent_size}"
            )

    def _check_inputs(self, features_list):
        if len(features_list) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} view feature tensors, got {len(features_list)}"
            )
        for index, features in enumerate(features_list):
            if features.dim() != 2 or features.size(1) != self.feature_dim:
                raise ValueError(
                    f"View {index} features must have shape [batch, {self.feature_dim}], "
                    f"got {tuple(features.shape)}"
                )

    def view_scores(self, features_list):
        """Return per-view weights for the weighted_sum and gated heads.

        Useful for analysing which camera the model relies on. Returns
        ``None`` for the other fusion types.
        """
        if self.fusion_type == "weighted_sum":
            return torch.softmax(self.view_weights, dim=0).expand(
                features_list[0].size(0), -1
            )
        if self.fusion_type == "gated":
            return self.gate_network(torch.cat(features_list, dim=1))
        return None

    def forward(self, features_list):
        self._check_inputs(list(features_list))

        if self.fusion_type == "concat_proj":
            return self.projection(torch.cat(features_list, dim=1))

        if self.fusion_type == "attention":
            tokens = torch.stack(features_list, dim=1)  # [B, V, D]
            if self.view_embedding is not None:
                tokens = tokens + self.view_embedding.unsqueeze(0)
            attended, _ = self.attention(tokens, tokens, tokens)
            if self.attention_residual:
                attended = self.attention_norm(attended + tokens)
            return self.final_projection(attended.mean(dim=1))

        latents = [
            projection(features)
            for projection, features in zip(self.view_projections, features_list)
        ]

        if self.fusion_type == "indiv_proj":
            return self.fusion_layer(torch.cat(latents, dim=1))

        scores = self.view_scores(features_list)  # [B, V]
        fused = torch.zeros_like(latents[0])
        for index, latent in enumerate(latents):
            fused = fused + scores[:, index : index + 1] * latent
        return fused

    def extra_repr(self):
        return self.description


def fusion_from_exp(exp, feature_dim, num_views, latent_size):
    """Build a :class:`MultiViewFusion` from the fusion keys of an exp/run."""
    fusion = MultiViewFusion(
        feature_dim,
        num_views,
        latent_size,
        exp.get("fusion_type", "concat_proj"),
        hidden_dim=exp.get("fusion_hidden_dim"),
        dropout=exp.get("fusion_dropout", 0.1),
        attention_heads=exp.get("attention_heads"),
        attention_residual=exp.get("attention_residual", True),
        view_embedding=exp.get("view_embedding", True),
    )
    print(f"Created fusion head ({fusion.description})")
    return fusion
