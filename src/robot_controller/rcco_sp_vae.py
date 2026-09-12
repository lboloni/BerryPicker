"""Deterministic Conv-VAE-Neo encoder component for controller graphs."""

from pathlib import Path

import torch
from torch import nn

from exp_run_config import Config
from robot_controller.abstract_rcco import AbstractRCComponent
from sensorprocessing.conv_vae_neo import ARCHITECTURE_VERSION, ConvVAENeo
from sensorprocessing.sp_factory import create_sp
from sensorprocessing.sp_helper import SensorPreprocessor
from training_harness.checkpoints import model_file


Config.PROJECTNAME = "BerryPicker"


def _unwrap_state(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


class ConvVAENeoEncoder(nn.Module):
    """The deployable, trainable subset of a :class:`ConvVAENeo`."""

    def __init__(self, exp_sp):
        super().__init__()
        vae = ConvVAENeo(exp_sp)
        self.encoder = vae.encoder
        self.fc_mu = vae.fc_mu
        self.image_size = vae.image_size
        self.input_channels = vae.input_channels
        self.latent_size = vae.latent_size
        self.flatten_size = vae.flatten_size

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("VAE encoder input must be a torch.Tensor")
        expected = (self.input_channels, *self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"VAE encoder expected [batch, {expected[0]}, {expected[1]}, "
                f"{expected[2]}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("VAE encoder input must be floating point")
        features = self.encoder(images).flatten(1)
        if features.size(1) != self.flatten_size:
            raise RuntimeError(
                f"VAE encoder produced {features.size(1)} features; expected "
                f"{self.flatten_size}"
            )
        return self.fc_mu(features)

    encode = forward

    def load_vae_state_dict(self, payload):
        """Load encoder weights from a full VAE or encoder-only checkpoint."""
        state = _unwrap_state(payload)
        if not isinstance(state, dict):
            raise TypeError("VAE checkpoint must contain a state dictionary")
        expected = set(self.state_dict())
        selected = {key: value for key, value in state.items() if key in expected}
        missing = sorted(expected - set(selected))
        if missing:
            raise KeyError(f"VAE checkpoint is missing encoder keys: {missing}")
        self.load_state_dict(selected, strict=True)

    def architecture_signature(self):
        return {
            "type": "ConvVAENeoEncoder",
            "architecture_version": ARCHITECTURE_VERSION,
            "image_size": list(self.image_size),
            "input_channels": self.input_channels,
            "latent_size": self.latent_size,
            "flatten_size": self.flatten_size,
        }


class RCCO_SP_VAE(AbstractRCComponent):
    """Encode a preprocessed image as the deterministic VAE latent mean."""

    def __init__(self, exp_rcco, *, load_state=True, sensor_exp=None):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None
        self.exp_sp = sensor_exp or Config().get_experiment(
            exp_rcco["sp_experiment"], exp_rcco["sp_run"]
        )
        if load_state:
            # Preserve the established sensor-processing construction path for
            # ordinary runtime controllers. Recipe bundles use the smaller
            # encoder-only module below.
            self.sp = create_sp(self.exp_sp)
            self.preprocessor = getattr(self.sp, "pre", None)
            if self.preprocessor is None:
                self.preprocessor = getattr(self.sp, "preprocessor", None)
            self.model = self.sp.enc
        else:
            self.sp = None
            self.preprocessor = SensorPreprocessor(self.exp_sp)
            self.model = ConvVAENeoEncoder(self.exp_sp).to(
                Config().runtime["device"]
            )
        self.latent_size = self.exp_sp["latent_size"]
        self.input_sizes["image"] = None
        self.output_sizes["z"] = self.latent_size

    def _model_path(self):
        return model_file(self.exp_sp)

    def load_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Required VAE model does not exist: {path}")
        payload = torch.load(
            path, map_location=Config().runtime["device"], weights_only=True
        )
        if hasattr(self.model, "load_vae_state_dict"):
            self.model.load_vae_state_dict(payload)
        else:
            state = _unwrap_state(payload)
            encoder_only = state and all(
                key.startswith("encoder.") or key.startswith("fc_mu.")
                for key in state
            )
            result = self.model.load_state_dict(state, strict=not encoder_only)
            if encoder_only and result.unexpected_keys:
                raise KeyError(
                    f"Unexpected VAE encoder keys: {result.unexpected_keys}"
                )
        self.model.eval()
        return path

    def load(self):
        return self.load_state()

    def save_state(self, path=None):
        if path is None:
            raise ValueError(
                "An explicit path is required when saving a controller VAE "
                "encoder so the full source VAE is not overwritten"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.model.state_dict()
        encoder_state = {
            key: value for key, value in state.items()
            if key.startswith("encoder.") or key.startswith("fc_mu.")
        }
        if not encoder_state or not {"fc_mu.weight", "fc_mu.bias"}.issubset(
            encoder_state
        ):
            raise RuntimeError("Could not extract a complete VAE encoder state")
        torch.save(encoder_state, path)
        return path

    def save(self):
        return self.save_state()

    def trainable_parameters(self):
        if isinstance(self.model, ConvVAENeoEncoder):
            return self.model.parameters()
        return (
            parameter
            for module in (self.model.encoder, self.model.fc_mu)
            for parameter in module.parameters()
        )

    def architecture_signature(self):
        if hasattr(self.model, "architecture_signature"):
            return self.model.architecture_signature()
        return {
            "type": "ConvVAENeoEncoder",
            "architecture_version": self.exp_sp.get(
                "architecture_version", ARCHITECTURE_VERSION
            ),
            "image_size": list(self.exp_sp["image_size"]),
            "input_channels": self.exp_sp.get("input_channels", 3),
            "latent_size": self.exp_sp["latent_size"],
            "flatten_size": self.model.flatten_size,
        }

    def preprocess_capture(self, capture):
        if self.preprocessor is None:
            raise RuntimeError("Configured sensor processor has no preprocessor")
        return self.preprocessor.from_capture(capture)

    def propagate(self):
        image = self.inputs["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("RCCO_SP_VAE input image must be a torch.Tensor")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(Config().runtime["device"])
        self.model.eval()
        with torch.inference_mode():
            encode = getattr(self.model, "encode", self.model)
            latent = encode(image)
        if latent.ndim != 2 or latent.size(1) != self.latent_size:
            raise RuntimeError(
                f"VAE encoder returned shape {tuple(latent.shape)}; expected "
                f"[batch, {self.latent_size}]"
            )
        if not torch.isfinite(latent).all():
            raise FloatingPointError("VAE encoder produced non-finite values")
        self.outputs["z"] = latent
        self.dirty = False
        return True
