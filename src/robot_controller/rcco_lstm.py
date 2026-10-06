"""LSTM feature component for robot-controller graphs.

The recurrent network is one of the RECURRENT_CORES. The component owns the
runtime context, which reset_context() clears at an episode boundary, in one
of two modes:

* sliding_window: keep the last sequence_length latents, and re-run the core
  over them from a zero state at every step;
* stateful: carry the recurrent state from step to step, and feed the core
  one latent per step.
"""

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


def _check_sequence(sequence, input_size):
    if not isinstance(sequence, torch.Tensor):
        raise TypeError("LSTM input must be a torch.Tensor")
    if sequence.ndim != 3 or sequence.size(-1) != input_size:
        raise ValueError(
            "Expected LSTM input shape [batch, sequence, "
            f"{input_size}], got {tuple(sequence.shape)}"
        )


def map_state(function, state):
    """Apply function to every tensor of a recurrent state (nested tuples
    and lists of tensors whose batch dimension is the second last)."""
    if isinstance(state, torch.Tensor):
        return function(state)
    return type(state)(map_state(function, item) for item in state)


class PlainLSTM(nn.Module):
    """A stacked nn.LSTM (the bc_LSTM core)."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.input_size = _positive_int(input_size, "input_size")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            batch_first=True, dropout=dropout,
        )

    def forward(self, sequence, state=None):
        """[batch, time, input] -> ([batch, time, hidden], state)"""
        _check_sequence(sequence, self.input_size)
        return self.lstm(sequence, state)


class ResidualLSTM(nn.Module):
    """LSTM stack whose later layers add the preceding layer as a residual
    (the bc_LSTM_Residual and bc_LSTM_MDN core)."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("ResidualLSTM does not support dropout")
        self.input_size = _positive_int(input_size, "input_size")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer_input = self.input_size if index == 0 else self.hidden_size
            self.layers.append(
                nn.LSTM(layer_input, self.hidden_size, batch_first=True)
            )

    def forward(self, sequence, state=None):
        """[batch, time, input] -> ([batch, time, hidden], state), where the
        state is the list of the per-layer (h, c)."""
        _check_sequence(sequence, self.input_size)
        if state is None:
            state = [None] * self.num_layers
        output = sequence
        new_state = []
        for index, layer in enumerate(self.layers):
            next_output, layer_state = layer(output, state[index])
            if index > 0:
                next_output = next_output + output
            output = next_output
            new_state.append(layer_state)
        return output, new_state


# The recurrent cores selectable by the "architecture" of an LSTM component.
# Every core maps (sequence [B, T, input], state) to
# (outputs [B, T, hidden], state); a state of None is the zero state.
RECURRENT_CORES = {"plain": PlainLSTM, "residual": ResidualLSTM}

CONTEXT_MODES = {"sliding_window", "stateful"}


def create_core(exp_rcco):
    """Create the recurrent core configured by an LSTM component exp/run."""
    architecture = exp_rcco.get("architecture", "residual")
    if architecture not in RECURRENT_CORES:
        raise ValueError(
            f"LSTM architecture must be one of {sorted(RECURRENT_CORES)}, "
            f"got {architecture!r}"
        )
    return RECURRENT_CORES[architecture](
        exp_rcco["input_size"], exp_rcco["hidden_size"],
        exp_rcco["num_layers"], exp_rcco.get("dropout", 0.0),
    )


class RCCO_LSTM(AbstractRCComponent):
    """Accumulate latent vectors and produce a temporal feature vector."""

    def __init__(self, exp_rcco, *, load_state=True):
        super().__init__(exp_rcco)
        self.architecture = exp_rcco.get("architecture", "residual")
        self.context_mode = exp_rcco.get("context_mode", "sliding_window")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {sorted(CONTEXT_MODES)}, got "
                f"{self.context_mode!r}"
            )
        self.input_size = _positive_int(exp_rcco["input_size"], "input_size")
        self.hidden_size = _positive_int(exp_rcco["hidden_size"], "hidden_size")
        # the window length; a stateful component has no window
        self.sequence_length = (
            _positive_int(exp_rcco["sequence_length"], "sequence_length")
            if self.context_mode == "sliding_window" else 1
        )
        self.inputs["z"] = None
        self.outputs["h"] = None
        self.input_sizes["z"] = self.input_size
        self.output_sizes["h"] = self.hidden_size
        self.context = deque(maxlen=self.sequence_length)
        self.state = None
        self.model = create_core(exp_rcco).to(Config().runtime["device"])
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
            "type": type(self.model).__name__,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.model.num_layers,
            "context_mode": self.context_mode,
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
        latent = self._runtime_latent()
        self.dirty = False
        self.model.eval()
        if self.context_mode == "stateful":
            with torch.inference_mode():
                output, self.state = self.model(
                    latent.view(1, 1, -1), self.state
                )
            self.outputs["h"] = output[:, -1, :]
            return True
        self.context.append(latent)
        if len(self.context) < self.sequence_length:
            self.outputs["h"] = None
            return False
        sequence = torch.stack(tuple(self.context)).unsqueeze(0)
        with torch.inference_mode():
            output, _ = self.model(sequence)
        self.outputs["h"] = output[:, -1, :]
        return True

    def reset_context(self):
        self.context.clear()
        self.state = None
        super().reset_context()
