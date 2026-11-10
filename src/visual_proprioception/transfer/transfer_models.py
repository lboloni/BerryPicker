"""
transfer_models.py

The models of the transfer experiments (DESIGN-Transfer.md):

* TransferEncoder: a frozen pretrained backbone, a trainable projection to the
  latent, and one regression head per training robot. With one robot it is the
  single-source encoder of the paper; with several it is design D1; with the
  vgg19_keypoints backbone it is design D2.
* ViTRandomProjection: the frozen ImageNet ViT with a fixed random projection,
  the reference R1 of the paper (the CNN version is
  sensorprocessing/sp_random_projection_cnn.py).
* MetaRidgeEncoder, ridge_fit, ridge_predict, fomaml_task: design D5.
* adapt_batchnorm: design D4a.
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision import models

from sensorprocessing.vit_helper import create_projection, create_proprioceptor


class ImageNetNormalize(nn.Module):
    """Normalizes [0, 1] images with the ImageNet statistics"""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SpatialSoftmax(nn.Module):
    """[B, K, H, W] heatmaps -> [B, 2K] expected (x, y) positions in [-1, 1]"""

    def __init__(self, height, width):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1))
        self.register_buffer("ys", ys.reshape(-1))
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, heatmaps):
        b, k, h, w = heatmaps.shape
        p = torch.softmax(heatmaps.reshape(b, k, h * w) / self.log_temperature.exp(), dim=-1)
        return torch.cat([(p * self.xs).sum(-1), (p * self.ys).sum(-1)], dim=-1)


class KeypointProjection(nn.Module):
    """A 1x1 convolution to latent_size/2 heatmaps, then their spatial softmax
    (pooling "softmax"), or their mean and max (pooling "avg", the ablation)"""

    def __init__(self, channels, latent_size, height, width, pooling):
        super().__init__()
        self.keypoints = nn.Conv2d(channels, latent_size // 2, kernel_size=1)
        self.pooling = pooling
        self.spatial_softmax = SpatialSoftmax(height, width)

    def forward(self, features):
        heatmaps = self.keypoints(features)
        if self.pooling == "softmax":
            return self.spatial_softmax(heatmaps)
        return torch.cat([heatmaps.mean(dim=(2, 3)), heatmaps.amax(dim=(2, 3))], dim=1)


def vgg19_features(pretrained):
    return models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None).features


def vit_b_16(pretrained):
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
    vit.heads = nn.Identity()
    return vit


def create_backbone_and_projection(exp):
    """The frozen backbone and the trainable projection of a TransferEncoder"""
    backbone, latent = exp["backbone"], exp["latent_size"]
    pretrained = exp.get("pretrained_backbone", True)
    height, width = exp["image_size"]
    if backbone == "vgg19":
        # the encoder of VGG19ProprioTunedRegression (sp_propriotuned_cnn.py)
        features = vgg19_features(pretrained)
        projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * (height // 32) * (width // 32), latent),
            nn.ReLU(),
            nn.Linear(latent, latent))
    elif backbone == "vgg19_keypoints":
        # up to relu4_4: [512, H/8, W/8]
        features = vgg19_features(pretrained)[:27]
        projection = KeypointProjection(512, latent, height // 8, width // 8, exp.get("keypoint_pooling", "softmax"))
    elif backbone == "vit_b_16":
        # the CLS token, then the projection of the ViT encoder (with BatchNorm)
        features = vit_b_16(pretrained)
        projection, _ = create_projection(768, latent, exp)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    return features, projection


class TransferEncoder(nn.Module):
    """A frozen backbone, a trainable projection to the latent, one head per robot.

    exp["robots"] is a list of {name, output_size, ...}; forward(x, name) gives
    the prediction of the head of robot name. Only encode() is used at inference.
    """

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.normalize = ImageNetNormalize()
        self.backbone, self.projection = create_backbone_and_projection(exp)
        self.backbone.requires_grad_(False)
        hidden = {"proprio_step_1": exp["head_hidden_size"], "proprio_step_2": exp["head_hidden_size"]}
        self.heads = nn.ModuleDict({
            robot["name"]: create_proprioceptor(self.latent_size, robot["output_size"], hidden)
            for robot in exp["robots"]})
        self.to(Config().runtime["device"])

    def train(self, mode=True):
        super().train(mode)
        # the frozen backbone always uses its pretrained statistics
        self.backbone.eval()
        return self

    def encode(self, images):
        with torch.no_grad():
            features = self.backbone(self.normalize(images))
        return self.projection(features)

    def forward(self, images, robot):
        return self.heads[robot](self.encode(images))


class ViTRandomProjection(nn.Module):
    """The frozen ImageNet ViT-B/16 (CLS token) and a fixed random projection,
    generated as in sp_random_projection_cnn.py"""

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.feature_normalization = exp["feature_normalization"]
        self.normalize = ImageNetNormalize()
        self.backbone = vit_b_16(exp.get("pretrained_backbone", True))
        self.backbone.requires_grad_(False)
        generator = torch.Generator(device="cpu").manual_seed(exp["projection_seed"])
        projection = torch.empty(self.latent_size, 768)
        if exp["projection"] == "rademacher":
            projection.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        elif exp["projection"] == "gaussian":
            projection.normal_(generator=generator)
        else:
            raise ValueError(f"Unknown projection {exp['projection']}")
        self.register_buffer("projection", projection / math.sqrt(self.latent_size))
        self.to(Config().runtime["device"])
        self.eval()

    def encode(self, images):
        features = self.backbone(self.normalize(images))
        if self.feature_normalization == "l2":
            features = F.normalize(features, p=2, dim=1)
        return F.linear(features, self.projection)

    def forward(self, images):
        return self.encode(images)


# ---------------------------------------------------------------------------
# D5: meta-learning with a closed-form ridge head
# ---------------------------------------------------------------------------

class MetaRidgeEncoder(nn.Module):
    """Frozen VGG19 features, pooled to exp["feature_pool_size"], and a trainable
    projection with LayerNorm (support sets of a few frames make BatchNorm
    useless). log_lambda is the meta-learned ridge regularization."""

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.normalize = ImageNetNormalize()
        self.backbone = vgg19_features(exp.get("pretrained_backbone", True))
        self.backbone.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d(tuple(exp["feature_pool_size"]))
        feature_size = 512 * exp["feature_pool_size"][0] * exp["feature_pool_size"][1]
        hidden = exp["projection_hidden_size"]
        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, self.latent_size))
        self.log_lambda = nn.Parameter(torch.tensor(float(exp["initial_log_lambda"])))
        self.to(Config().runtime["device"])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        """The frozen part: [B, feature_size]"""
        with torch.no_grad():
            return torch.flatten(self.pool(self.backbone(self.normalize(images))), 1)

    def encode(self, images):
        return self.projection(self.features(images))


def with_bias(z):
    return torch.cat([z, torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)], dim=1)


def solve_on_cpu(a, b):
    """torch.linalg.solve on the CPU: its backward pass is not implemented on
    MPS. The matrices are small, at most the latent size + 1 square."""
    return torch.linalg.solve(a.cpu(), b.cpu()).to(a.device)


def ridge_fit(z, y, log_lambda):
    """The closed-form ridge regression of y [n, k] on z [n, d], with a bias:
    W [d+1, k]. Uses the Woodbury form when n < d+1."""
    zb = with_bias(z)
    n, d = zb.shape
    lam = log_lambda.exp() if torch.is_tensor(log_lambda) else math.exp(log_lambda)
    if n < d:
        a = zb @ zb.T + lam * torch.eye(n, device=z.device, dtype=z.dtype)
        return zb.T @ solve_on_cpu(a, y)
    a = zb.T @ zb + lam * torch.eye(d, device=z.device, dtype=z.dtype)
    return solve_on_cpu(a, zb.T @ y)


def ridge_predict(z, w):
    return with_bias(z) @ w


def fomaml_task(projection, f_s, y_s, f_q, y_q, inner_steps, inner_lr):
    """First-order MAML on one task: inner_steps SGD steps on the support set,
    over the projection and a zero-initialized linear head. Returns the query
    loss and the gradients of the query loss at the adapted weights, keyed by
    the parameter names of the projection (the first-order approximation)."""
    fast = {name: p.detach().clone().requires_grad_(True) for name, p in projection.named_parameters()}
    latent = fast[list(fast)[-1]].shape[0]
    head = torch.zeros(latent + 1, y_s.shape[1], device=f_s.device, requires_grad=True)
    for _ in range(inner_steps):
        loss = F.mse_loss(ridge_predict(functional_call(projection, fast, (f_s,)), head), y_s)
        grads = torch.autograd.grad(loss, list(fast.values()) + [head])
        fast = {name: (p - inner_lr * g).detach().requires_grad_(True) for (name, p), g in zip(fast.items(), grads)}
        head = (head - inner_lr * grads[-1]).detach().requires_grad_(True)
    query_loss = F.mse_loss(ridge_predict(functional_call(projection, fast, (f_q,)), head), y_q)
    grads = torch.autograd.grad(query_loss, list(fast.values()))
    return query_loss.detach(), dict(zip(fast.keys(), grads))


# ---------------------------------------------------------------------------
# D4a: re-estimating the BatchNorm statistics on unlabeled target images
# ---------------------------------------------------------------------------

def adapt_batchnorm(network, modules, forward, batches):
    """Recomputes the running statistics of the BatchNorm layers among modules
    as the cumulative average over the batches, passed through forward. The
    network is left in eval mode; dropout is never active."""
    bns = [m for m in modules if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    network.eval()
    for m in bns:
        m.train()
    with torch.no_grad():
        for batch in batches:
            forward(batch)
    network.eval()
    return len(bns)


class RidgeRegressor(nn.Module):
    """The ridge regressor of the regressor stage: [z, 1] @ W"""

    def __init__(self, w):
        super().__init__()
        self.register_buffer("w", w)

    def forward(self, z):
        return ridge_predict(z, self.w)
