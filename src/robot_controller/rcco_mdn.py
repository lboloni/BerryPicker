"""Mixture-density action component for robot-controller graphs."""

from pathlib import Path

import torch

from behavior_cloning.mdn import MDN
from exp_run_config import Config
from robot_controller.abstract_rcco import AbstractRCComponent


Config.PROJECTNAME = "BerryPicker"


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class RCCO_MDN(AbstractRCComponent):
    """Map an LSTM feature to a mixture distribution and one action."""

    ACTION_SELECTIONS = {"sample", "expected_value", "maximum_component"}

    def __init__(self, exp_rcco, *, load_state=True):
        super().__init__(exp_rcco)
        self.input_size = _positive_int(exp_rcco["input_dim"], "input_dim")
        self.output_size = _positive_int(exp_rcco["output_dim"], "output_dim")
        self.num_gaussians = _positive_int(
            exp_rcco["num_gaussians"], "num_gaussians"
        )
        self.hidden_size = _positive_int(exp_rcco["hidden_size"], "hidden_size")
        self.action_selection = exp_rcco.get("action_selection", "sample")
        if self.action_selection not in self.ACTION_SELECTIONS:
            raise ValueError(
                "action_selection must be one of "
                f"{sorted(self.ACTION_SELECTIONS)}, got {self.action_selection!r}"
            )
        self.inputs["h"] = None
        self.outputs.update({"mu": None, "sigma": None, "pi": None, "a": None})
        self.input_sizes["h"] = self.input_size
        for name in ("mu", "sigma", "pi"):
            self.output_sizes[name] = self.output_size * self.num_gaussians
        self.output_sizes["a"] = self.output_size
        self.model = MDN(exp_rcco).to(Config().runtime["device"])
        if load_state:
            self.load()

    def _model_path(self):
        try:
            data_dir = self.exp["data_dir"]
            filename = self.exp["model_file"]
        except KeyError as error:
            raise KeyError(
                "MDN component requires 'data_dir' and 'model_file'"
            ) from error
        return Path(data_dir) / filename

    def load_state(self, path=None):
        path = Path(path) if path is not None else self._model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Required MDN model does not exist: {path}")
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
            "type": "MDN",
            "input_dim": self.input_size,
            "hidden_size": self.hidden_size,
            "output_dim": self.output_size,
            "num_gaussians": self.num_gaussians,
        }

    def _runtime_feature(self):
        feature = self.inputs["h"]
        if not isinstance(feature, torch.Tensor):
            raise TypeError("RCCO_MDN input h must be a torch.Tensor")
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        if feature.ndim != 2 or feature.size(1) != self.input_size:
            raise ValueError(
                f"Expected MDN input shape [batch, {self.input_size}], got "
                f"{tuple(feature.shape)}"
            )
        if not torch.isfinite(feature).all():
            raise ValueError("MDN input contains non-finite values")
        return feature.to(Config().runtime["device"])

    @staticmethod
    def _sample(mu, sigma, pi):
        mixture = torch.distributions.Categorical(probs=pi).sample()
        selected_mu = torch.gather(mu, -1, mixture.unsqueeze(-1)).squeeze(-1)
        selected_sigma = torch.gather(
            sigma, -1, mixture.unsqueeze(-1)
        ).squeeze(-1)
        return torch.distributions.Normal(selected_mu, selected_sigma).sample()

    def _select_action(self, mu, sigma, pi):
        if self.action_selection == "sample":
            return self._sample(mu, sigma, pi)
        if self.action_selection == "expected_value":
            return torch.sum(pi * mu, dim=-1)
        mixture = torch.argmax(pi, dim=-1, keepdim=True)
        return torch.gather(mu, -1, mixture).squeeze(-1)

    def propagate(self):
        feature = self._runtime_feature()
        self.model.eval()
        with torch.inference_mode():
            mu, sigma, pi = self.model(feature)
            action = self._select_action(mu, sigma, pi)
        if not all(torch.isfinite(value).all() for value in (mu, sigma, pi, action)):
            raise FloatingPointError("MDN produced non-finite output")
        if not torch.all(sigma > 0):
            raise FloatingPointError("MDN produced a non-positive standard deviation")
        self.outputs.update({"mu": mu, "sigma": sigma, "pi": pi, "a": action})
        self.dirty = False
        return True
