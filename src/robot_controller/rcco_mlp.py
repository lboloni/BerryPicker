"""Deterministic MLP action component for robot-controller graphs."""

from pathlib import Path

import torch
from torch import nn

from exp_run_config import Config
from robot_controller.abstract_rcco import AbstractRCComponent


Config.PROJECTNAME = "BerryPicker"


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class MLPController(nn.Module):
    """Map one latent vector to a deterministic normalized action."""

    OUTPUT_ACTIVATIONS = {"sigmoid", "identity"}

    def __init__(self, input_size, hidden_sizes, output_size, output_activation):
        super().__init__()
        self.input_size = _positive_int(input_size, "input_size")
        self.output_size = _positive_int(output_size, "output_size")
        if not isinstance(hidden_sizes, list) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a nonempty list")
        self.hidden_sizes = tuple(
            _positive_int(size, f"hidden_sizes[{index}]")
            for index, size in enumerate(hidden_sizes)
        )
        if output_activation not in self.OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {sorted(self.OUTPUT_ACTIVATIONS)}"
            )
        self.output_activation = output_activation
        layers = []
        previous = self.input_size
        for size in self.hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU()))
            previous = size
        layers.append(nn.Linear(previous, self.output_size))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, latent):
        if not isinstance(latent, torch.Tensor):
            raise TypeError("MLP input must be a torch.Tensor")
        if latent.ndim != 2 or latent.size(1) != self.input_size:
            raise ValueError(
                f"Expected MLP input [batch, {self.input_size}], got "
                f"{tuple(latent.shape)}"
            )
        return self.model(latent)


class RCCO_MLP(AbstractRCComponent):
    """Map the current CNN latent to one normalized robot action."""

    def __init__(self, exp_rcco, *, load_state=True):
        super().__init__(exp_rcco)
        self.input_size = _positive_int(exp_rcco["input_size"], "input_size")
        self.output_size = _positive_int(exp_rcco["output_size"], "output_size")
        self.hidden_sizes = exp_rcco["hidden_sizes"]
        self.output_activation = exp_rcco.get("output_activation", "sigmoid")
        self.inputs["z"] = None
        self.outputs["a"] = None
        self.input_sizes["z"] = self.input_size
        self.output_sizes["a"] = self.output_size
        self.model = MLPController(
            self.input_size, self.hidden_sizes, self.output_size,
            self.output_activation,
        ).to(Config().runtime["device"])
        if load_state:
            self.load()

    def _model_path(self):
        try:
            return Path(self.exp["data_dir"]) / self.exp["model_file"]
        except KeyError as error:
            raise KeyError(
                "MLP component requires 'data_dir' and 'model_file'"
            ) from error

    def load_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Required MLP model does not exist: {path}")
        state = torch.load(
            path, map_location=Config().runtime["device"], weights_only=True
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        return path

    def load(self):
        return self.load_state()

    def save_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    def save(self):
        return self.save_state()

    def trainable_parameters(self):
        return self.model.parameters()

    def architecture_signature(self):
        return {
            "type": "MLPController",
            "input_size": self.input_size,
            "hidden_sizes": list(self.model.hidden_sizes),
            "output_size": self.output_size,
            "output_activation": self.output_activation,
        }

    def propagate(self):
        latent = self.inputs["z"]
        if not isinstance(latent, torch.Tensor):
            raise TypeError("RCCO_MLP input z must be a torch.Tensor")
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        latent = latent.to(Config().runtime["device"])
        self.model.eval()
        with torch.inference_mode():
            action = self.model(latent)
        if action.shape != (latent.size(0), self.output_size):
            raise RuntimeError(
                f"MLP produced shape {tuple(action.shape)}; expected "
                f"[{latent.size(0)}, {self.output_size}]"
            )
        if not torch.isfinite(action).all():
            raise FloatingPointError("MLP produced non-finite action values")
        if self.output_activation == "sigmoid" and not torch.all(
            (0 <= action) & (action <= 1)
        ):
            raise RuntimeError("Sigmoid MLP output is outside [0, 1]")
        self.outputs["a"] = action
        self.dirty = False
        return True
