"""Pretrained CNN encoder component for robot-controller graphs."""

from pathlib import Path

import torch

from exp_run_config import Config
from robot_controller.abstract_rcco import AbstractRCComponent
from sensorprocessing.sp_factory import create_sp
from sensorprocessing.sp_helper import SensorPreprocessor
from sensorprocessing.sp_propriotuned_cnn import (
    ProprioTunedCNNSensorProcessing,
)
from training_harness.checkpoints import model_file


Config.PROJECTNAME = "BerryPicker"


def _unwrap_state(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _create_cnn(exp_sp):
    values = getattr(exp_sp, "values", exp_sp)
    if callable(values):
        values = exp_sp
    values = dict(values)
    # Recipe and bundle construction immediately loads a complete checkpoint;
    # it must not require an ImageNet download merely to create the modules.
    values["pretrained_backbone"] = False
    try:
        model_class = ProprioTunedCNNSensorProcessing.encoder_classes[
            values["model"]
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported controller CNN model {values.get('model')!r}"
        ) from error
    return model_class(values).to(Config().runtime["device"])


class RCCO_SP_CNN(AbstractRCComponent):
    """Encode a preprocessed image without exposing the CNN regression head."""

    def __init__(self, exp_rcco, *, load_state=True, sensor_exp=None):
        super().__init__(exp_rcco)
        self.inputs["image"] = None
        self.outputs["z"] = None
        self.exp_sp = sensor_exp or Config().get_experiment(
            exp_rcco["sp_experiment"], exp_rcco["sp_run"]
        )
        self.latent_size = self.exp_sp["latent_size"]
        if type(self.latent_size) is not int or self.latent_size <= 0:
            raise ValueError("CNN latent_size must be a positive integer")
        if load_state:
            self.sp = create_sp(self.exp_sp)
            self.model = self.sp.enc
            self.preprocessor = getattr(self.sp, "preprocessor", None)
        else:
            self.sp = None
            self.model = _create_cnn(self.exp_sp)
            self.preprocessor = SensorPreprocessor(self.exp_sp)
        if not callable(getattr(self.model, "encode", None)):
            raise TypeError("SP_CNN model must expose encode(tensor)")
        self.input_sizes["image"] = None
        self.output_sizes["z"] = self.latent_size

    def _model_path(self):
        return model_file(self.exp_sp)

    def load_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Required CNN model does not exist: {path}")
        state = torch.load(
            path, map_location=Config().runtime["device"], weights_only=True
        )
        self.model.load_state_dict(_unwrap_state(state), strict=True)
        self.model.eval()
        return path

    def load(self):
        return self.load_state()

    def save_state(self, path=None):
        if path is None:
            raise ValueError(
                "An explicit path is required when saving a controller CNN "
                "so the source sensor-processing model is not overwritten"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    def save(self):
        return self.save_state()

    def trainable_parameters(self):
        return self.model.parameters()

    def architecture_signature(self):
        signature = {
            "type": "ProprioTunedCNNEncoder",
            "model": self.exp_sp["model"],
            "image_size": list(self.exp_sp["image_size"]),
            "latent_size": self.latent_size,
        }
        for key in (
            "reductor_step_1", "proprio_step_1", "proprio_step_2",
            "output_size",
        ):
            if key in self.exp_sp:
                signature[key] = self.exp_sp[key]
        return signature

    def preprocess_capture(self, capture):
        if self.preprocessor is None:
            raise RuntimeError("Configured CNN sensor processor has no preprocessor")
        return self.preprocessor.from_capture(capture)

    def reset_context(self):
        super().reset_context()
        if self.sp is not None:
            self.sp.reset_context()

    def propagate(self):
        image = self.inputs["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("RCCO_SP_CNN input image must be a torch.Tensor")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(
                "RCCO_SP_CNN expected [channels, height, width] or "
                f"[batch, channels, height, width], got {tuple(image.shape)}"
            )
        expected = (3, *self.exp_sp["image_size"])
        if tuple(image.shape[1:]) != expected:
            raise ValueError(
                f"RCCO_SP_CNN expected image shape {expected}, got "
                f"{tuple(image.shape[1:])}"
            )
        if not image.is_floating_point():
            raise TypeError("RCCO_SP_CNN input image must be floating point")
        image = image.to(Config().runtime["device"])
        self.model.eval()
        with torch.inference_mode():
            latent = self.model.encode(image)
        if latent.ndim != 2 or latent.size(1) != self.latent_size:
            raise RuntimeError(
                f"CNN encoder returned shape {tuple(latent.shape)}; expected "
                f"[batch, {self.latent_size}]"
            )
        if not torch.isfinite(latent).all():
            raise FloatingPointError("CNN encoder produced non-finite values")
        self.outputs["z"] = latent
        self.dirty = False
        return True
