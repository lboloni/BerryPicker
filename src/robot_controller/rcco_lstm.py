"""Sliding-window LSTM feature component for robot-controller graphs."""

from collections import deque
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


class ResidualLSTM(nn.Module):
    """LSTM stack whose later layers add the preceding layer as a residual."""

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = _positive_int(input_size, "input_size")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer_input = self.input_size if index == 0 else self.hidden_size
            self.layers.append(
                nn.LSTM(layer_input, self.hidden_size, batch_first=True)
            )

    def forward(self, sequence):
        if not isinstance(sequence, torch.Tensor):
            raise TypeError("ResidualLSTM input must be a torch.Tensor")
        if sequence.ndim != 3 or sequence.size(-1) != self.input_size:
            raise ValueError(
                "Expected LSTM input shape [batch, sequence, "
                f"{self.input_size}], got {tuple(sequence.shape)}"
            )
        output = sequence
        for index, layer in enumerate(self.layers):
            next_output, _ = layer(output)
            if index > 0:
                next_output = next_output + output
            output = next_output
        return output[:, -1, :]


class RCCO_LSTM(AbstractRCComponent):
    """Accumulate latent vectors and produce a temporal feature vector."""

    def __init__(self, exp_rcco, *, load_state=True):
        super().__init__(exp_rcco)
        architecture = exp_rcco.get("architecture", "residual")
        if architecture != "residual":
            raise ValueError(
                "RCCO_LSTM currently supports architecture='residual' only"
            )
        context_mode = exp_rcco.get("context_mode", "sliding_window")
        if context_mode != "sliding_window":
            raise ValueError(
                "RCCO_LSTM currently supports context_mode='sliding_window' only"
            )
        self.input_size = _positive_int(exp_rcco["input_size"], "input_size")
        self.hidden_size = _positive_int(exp_rcco["hidden_size"], "hidden_size")
        self.sequence_length = _positive_int(
            exp_rcco["sequence_length"], "sequence_length"
        )
        self.inputs["z"] = None
        self.outputs["h"] = None
        self.input_sizes["z"] = self.input_size
        self.output_sizes["h"] = self.hidden_size
        self.context = deque(maxlen=self.sequence_length)
        self.model = ResidualLSTM(
            self.input_size,
            self.hidden_size,
            _positive_int(exp_rcco["num_layers"], "num_layers"),
        ).to(Config().runtime["device"])
        if load_state:
            self.load()

    def _model_path(self):
        try:
            data_dir = self.exp["data_dir"]
            filename = self.exp["model_file"]
        except KeyError as error:
            raise KeyError(
                "LSTM component requires 'data_dir' and 'model_file'"
            ) from error
        return Path(data_dir) / filename

    def load_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Required LSTM model does not exist: {path}")
        state = torch.load(
            path,
            map_location=Config().runtime["device"],
            weights_only=True,
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
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
            "type": "ResidualLSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.model.num_layers,
            "sequence_length": self.sequence_length,
        }

    def _runtime_latent(self):
        latent = self.inputs["z"]
        if not isinstance(latent, torch.Tensor):
            raise TypeError("RCCO_LSTM input z must be a torch.Tensor")
        if latent.ndim == 2:
            if latent.size(0) != 1:
                raise ValueError("Runtime LSTM input must have batch size 1")
            latent = latent.squeeze(0)
        if latent.ndim != 1 or latent.numel() != self.input_size:
            raise ValueError(
                f"Expected runtime latent shape [{self.input_size}], got "
                f"{tuple(latent.shape)}"
            )
        if not torch.isfinite(latent).all():
            raise ValueError("Runtime LSTM input contains non-finite values")
        return latent.detach().to(Config().runtime["device"])

    def propagate(self):
        self.context.append(self._runtime_latent())
        self.dirty = False
        if len(self.context) < self.sequence_length:
            self.outputs["h"] = None
            return False
        sequence = torch.stack(tuple(self.context)).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            self.outputs["h"] = self.model(sequence)
        return True

    def reset_context(self):
        self.context.clear()
        super().reset_context()
