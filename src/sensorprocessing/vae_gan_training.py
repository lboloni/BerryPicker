"""Three-optimizer VAE-GAN training with epoch-boundary recovery.

The last checkpoint is authoritative: exports, metrics and status are derived
from it on resume. Interrupting an epoch repeats that epoch, not partial updates.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from exp_run_config import Config
from training_harness.checkpoints import model_file
from .conv_vae_neo import make_dataloaders, seed_everything, _finite_nonnegative
from .vae_gan import VAEGAN, kl_loss


@contextmanager
def frozen(module):
    """Freeze parameters without detaching the module's input gradients."""
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class VAEGANTrainer:
    def __init__(self, model, exp):
        self.model, self.exp = model, exp
        vae = model.vae
        self.encoder = nn.ModuleList([vae.encoder, vae.fc_mu, vae.fc_logvar])
        self.generator = nn.ModuleList([vae.fc_decode, vae.decoder, vae.output_layer])
        self.groups = {"encoder": self.encoder, "generator": self.generator,
                       "discriminator": model.discriminator}
        self.optimizers = {}
        for name, module in self.groups.items():
            lr = _finite_nonnegative(exp.get(name + "_learning_rate", 0.0002), name + "_learning_rate")
            if lr == 0:
                raise ValueError("Learning rates must be positive")
            self.optimizers[name] = torch.optim.Adam(
                module.parameters(), lr=lr, betas=(0.5, 0.999),
                weight_decay=_finite_nonnegative(exp.get("weight_decay", 0.0), "weight_decay"))
        self.weights = {name: _finite_nonnegative(exp.get(name + "_weight", default), name + "_weight")
                        for name, default in (("kl", .001), ("feature", 1.), ("adversarial", .1), ("pixel", 0.))}
        self.clip = _finite_nonnegative(exp.get("grad_clip_norm", 1.), "grad_clip_norm")

    def _step(self, name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        loss.backward()
        nn.utils.clip_grad_norm_(self.groups[name].parameters(), self.clip or float("inf"), error_if_nonfinite=True)
        self.optimizers[name].step()

    def update(self, images, part):
        """One isolated update; exposed separately for gradient-boundary tests."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        vae, discriminator = self.model.vae, self.model.discriminator
        w = self.weights
        if part == "discriminator":
            with torch.no_grad():
                mu, logvar = vae.encode_distribution(images)
                reconstruction = vae.decode(vae.reparameterize(mu, logvar))
                prior = vae.sample(len(images))
            real, _ = discriminator(images)
            rec, _ = discriminator(reconstruction)
            generated, _ = discriminator(prior)
            loss = (F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                    + .5 * (F.binary_cross_entropy_with_logits(rec, torch.zeros_like(rec))
                            + F.binary_cross_entropy_with_logits(generated, torch.zeros_like(generated))))
            self._step(part, loss)
            return {"discriminator": loss.item()}
        if part not in {"encoder", "generator"}:
            raise ValueError(f"Unknown update: {part}")
        other = self.generator if part == "encoder" else self.encoder
        with frozen(discriminator), frozen(other):
            with torch.no_grad():
                _, target_features = discriminator(images)
            mu, logvar = vae.encode_distribution(images)
            reconstruction = vae.decode(vae.reparameterize(mu, logvar))
            logits, features = discriminator(reconstruction)
            feature = F.mse_loss(features, target_features)
            pixel = F.mse_loss(reconstruction, images)
            loss = w["feature"] * feature + w["pixel"] * pixel
            metrics = {"feature_" + part: feature.item(), "pixel_" + part: pixel.item()}
            if part == "encoder":
                kl = kl_loss(mu, logvar)
                loss = loss + w["kl"] * kl
                metrics["kl"] = kl.item()
            else:
                prior_logits, _ = discriminator(vae.sample(len(images)))
                adversarial = .5 * (F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                                    + F.binary_cross_entropy_with_logits(prior_logits, torch.ones_like(prior_logits)))
                loss = loss + w["adversarial"] * adversarial
                metrics["adversarial"] = adversarial.item()
            self._step(part, loss)
            metrics[part] = loss.item()
            return metrics

    def batch(self, images, warmup=False):
        self.model.train()
        metrics = self.update(images, "discriminator")
        if not warmup:
            metrics.update(self.update(images, "encoder"))
            metrics.update(self.update(images, "generator"))
        return metrics


def initialize_vae(model, exp):
    initialization = exp.get("initialization", "random")
    if initialization == "random":
        return None
    if initialization != "conv_vae_neo":
        raise ValueError("initialization must be random or conv_vae_neo")
    source = Config().get_experiment(exp["initialization_experiment"], exp["initialization_run"])
    for key, default in (("architecture_version", 1), ("image_size", None), ("latent_size", None),
                         ("input_channels", 3), ("base_channels", 32), ("max_channels", 512),
                         ("bottleneck_max_size", 8), ("group_norm_groups", 8)):
        if source.get(key, default) != exp.get(key, default):
            raise ValueError(f"Incompatible Neo initialization: {key}")
    path = model_file(source)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.vae.load_state_dict(state.get("model_state_dict", state), strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(descriptor)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_text(path, text):
    def writer(temporary):
        with open(temporary, "w") as stream:
            stream.write(text)
    _atomic(path, writer)


def _rng_state(loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state() if loader.generator is not None else None}


def _restore_rng(state, loader):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state["loader"] is not None:
        loader.generator.set_state(state["loader"].cpu())


def _fingerprint(exp):
    values = getattr(exp, "values", exp)
    if callable(values):
        values = exp
    ignored = {"data_dir", "epochs", "keep_checkpoints", "reload_existing_model",
               "time_started", "time_done", "exp_run_sys_indep_file",
               "exp_run_sys_dep_file", "subrun_name"}
    return hashlib.sha256(json.dumps({k: v for k, v in values.items() if k not in ignored},
                                    sort_keys=True, default=str).encode()).hexdigest()


def _publish(exp, checkpoint):
    directory = Path(exp["data_dir"])
    if checkpoint["best_vae"] is not None:
        _atomic(model_file(exp), lambda path: torch.save(checkpoint["best_vae"], path))
    _write_text(directory / "metrics.jsonl", "".join(json.dumps(row) + "\n" for row in checkpoint["history"]))
    _write_text(directory / "training_manifest.json", json.dumps({
        "format_version": 1, "next_epoch": checkpoint["next_epoch"], "phase": checkpoint["phase"],
        "best_epoch": checkpoint["best_epoch"], "best_validation_mse": checkpoint["best_metric"],
        "fingerprint": checkpoint["fingerprint"], "initialization": checkpoint["initialization"],
        "resume_checkpoint": "checkpoints/last.pt", "model_file": exp["model_file"],
    }, indent=2))


def train(exp, *, epochs=None, loaders=None, callback=None, device=None):
    """Train/resume to a total epoch count. Callback receives (model, history).

    Callbacks run after a committed epoch and cannot perturb training RNG.
    Injected loaders must yield image tensors (not image/label pairs).
    """
    total = exp["epochs"] if epochs is None else epochs
    warmup = exp.get("discriminator_warmup_epochs", 0)
    keep = exp.get("keep_checkpoints", 2)
    if type(total) is not int or total <= 0 or type(warmup) is not int or warmup < 0:
        raise ValueError("epochs must be positive and discriminator_warmup_epochs nonnegative")
    if type(keep) is not int or keep < 1:
        raise ValueError("keep_checkpoints must be positive")
    training_ids = {tuple(entry[:2]) for entry in exp["training_data"]}
    validation_ids = {tuple(entry[:2]) for entry in exp["validation_data"]}
    if training_ids & validation_ids:
        raise ValueError("Whole demonstrations must be disjoint between training and validation")
    seed_everything(exp.get("random_seed", 0))
    train_loader, val_loader = make_dataloaders(exp) if loaders is None else loaders
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation loaders must be nonempty")
    device = torch.device(device or Config().runtime["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Resumable VAE-GAN training currently supports CPU and CUDA devices")
    model = VAEGAN(exp).to(device)
    trainer = VAEGANTrainer(model, exp)
    directory = Path(exp["data_dir"]) / "checkpoints"
    last = directory / "last.pt"
    fingerprint = _fingerprint(exp)
    history, start, best, best_epoch, best_vae = [], 0, float("inf"), None, None
    initialization = None
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True)
        if checkpoint["format_version"] != 1 or checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint format/configuration mismatch; use a new exp/run")
        model.load_state_dict(checkpoint["model_state_dict"])
        for name, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
        start, history = checkpoint["next_epoch"], checkpoint["history"]
        best, best_epoch, best_vae = checkpoint["best_metric"], checkpoint["best_epoch"], checkpoint["best_vae"]
        initialization = checkpoint["initialization"]
        _restore_rng(checkpoint["rng"], train_loader)
        _publish(exp, checkpoint)
    else:
        if Path(model_file(exp)).exists():
            raise FileExistsError("VAE export exists without a resume checkpoint; use a new exp/run")
        initialization = initialize_vae(model, exp)
    for epoch in range(start, total):
        sums, count = {}, 0
        for images in train_loader:
            images = images.to(device)
            metrics = trainer.batch(images, warmup=epoch < warmup)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.) + len(images) * value
            count += len(images)
        model.eval()
        validation, kl, feature, samples = 0., 0., 0., 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                _, real_features = model.discriminator(images)
                _, rec_features = model.discriminator(recon)
                validation += F.mse_loss(recon, images).item() * len(images)
                kl += kl_loss(mu, logvar).item() * len(images)
                feature += F.mse_loss(rec_features, real_features).item() * len(images)
                samples += len(images)
        row = {"epoch": epoch + 1, "phase": "discriminator_warmup" if epoch < warmup else "joint",
               **{key: value / count for key, value in sums.items()},
               "validation_mse": validation / samples, "validation_kl": kl / samples,
               "validation_feature": feature / samples}
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise FloatingPointError("Non-finite epoch metrics")
        history.append(row)
        if row["validation_mse"] < best:
            best, best_epoch = row["validation_mse"], epoch + 1
            best_vae = {key: value.detach().cpu().clone() for key, value in model.vae.state_dict().items()}
        checkpoint = {"format_version": 1, "fingerprint": fingerprint, "next_epoch": epoch + 1,
                      "phase": "discriminator_warmup" if epoch + 1 < warmup else "joint",
                      "model_state_dict": model.state_dict(),
                      "optimizers": {key: opt.state_dict() for key, opt in trainer.optimizers.items()},
                      "best_metric": best, "best_epoch": best_epoch, "best_vae": best_vae,
                      "history": history, "rng": _rng_state(train_loader), "initialization": initialization}
        _atomic(last, lambda path: torch.save(checkpoint, path))
        _atomic(directory / f"epoch_{epoch + 1:06d}.pt", lambda path: torch.save(checkpoint, path))
        _publish(exp, checkpoint)
        for obsolete in sorted(directory.glob("epoch_*.pt"))[:-keep]:
            obsolete.unlink()
        if callback is not None:
            try:
                callback(model, history)
            finally:
                _restore_rng(checkpoint["rng"], train_loader)
    model.eval()
    return model, history
