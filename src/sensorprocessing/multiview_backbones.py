"""
multiview_backbones.py

Backbone bookkeeping shared by the multi-view encoders.

``ViewBackbones`` holds either one backbone shared by all camera views
(``shared_backbone: true``, the default) or one backbone per view
(``shared_backbone: false``). It also takes care of two details that are easy
to get wrong:

- A frozen backbone stays in eval mode even while the rest of the model is
  training. Freezing the parameters does not stop BatchNorm layers (ResNet)
  from updating their running statistics, which silently changes the
  "frozen" features from epoch to epoch.
- With a shared backbone, all views are pushed through the backbone in one
  batched call (``batched_backbone: true``), which is faster on the GPU than
  one call per view and gives the same result for backbones without batch
  statistics (ViT, VGG, or a frozen ResNet).
"""

import torch
import torch.nn as nn


class ViewBackbones(nn.Module):
    """One shared backbone, or one backbone per view."""

    def __init__(self, make_backbone, num_views, *, shared=True, freeze=False, batched=True):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be at least 1")
        self.num_views = num_views
        self.shared = shared
        self.freeze = freeze
        self.batched = batched and shared

        if shared:
            self.backbone = make_backbone()
            self.backbones = None
        else:
            self.backbone = None
            self.backbones = nn.ModuleList(make_backbone() for _ in range(num_views))

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self._backbones_eval()

    def _all_backbones(self):
        return [self.backbone] if self.shared else list(self.backbones)

    def _backbones_eval(self):
        for backbone in self._all_backbones():
            backbone.eval()

    def train(self, mode=True):
        """Keep frozen backbones in eval mode so BatchNorm statistics stay put."""
        super().train(mode)
        if self.freeze:
            self._backbones_eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, views_list):
        """Return one feature tensor per view, in the given view order."""
        views_list = list(views_list)
        if len(views_list) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views_list)}")

        if not self.shared:
            return [
                backbone(view) for backbone, view in zip(self.backbones, views_list)
            ]

        if self.batched:
            batch_size = views_list[0].size(0)
            for index, view in enumerate(views_list):
                if view.size(0) != batch_size:
                    raise ValueError(
                        f"View {index} has batch size {view.size(0)}; expected {batch_size}"
                    )
            features = self.backbone(torch.cat(views_list, dim=0))
            return list(features.split(batch_size, dim=0))

        return [self.backbone(view) for view in views_list]

    def extra_repr(self):
        kind = "shared" if self.shared else "per-view"
        return f"{kind} backbone x{1 if self.shared else self.num_views}, frozen={self.freeze}, batched={self.batched}"


def count_parameters(module):
    """Return (total, trainable) parameter counts."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def describe_multiview_model(model, exp, name):
    """Print a short, uniform summary of a multi-view encoder."""
    total, trainable = count_parameters(model)
    print(f"Created {name}:")
    print(f"  Views: {model.num_views} (cameras: {exp.get('cameras', 'unspecified')})")
    print(f"  Fusion: {model.fusion.description}")
    print(f"  Shared backbone: {model.backbones.shared}, frozen: {model.backbones.freeze}")
    print(f"  Latent size: {model.latent_size}")
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
