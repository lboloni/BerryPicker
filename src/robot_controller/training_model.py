"""Batched VAE--LSTM--MDN model used by controller training recipes."""

import torch
from torch import nn

from behavior_cloning.mdn import MDN
from robot_controller.rcco_lstm import ResidualLSTM
from robot_controller.rcco_sp_vae import ConvVAENeoEncoder, _unwrap_state


TRAINABLE_TYPES = ("SP_VAE", "LSTM", "MDN")


class RobotControllerTrainingModel(nn.Module):
    """A differentiable realization of the supported controller graph."""

    def __init__(self, controller_spec):
        super().__init__()
        self.controller_spec = controller_spec
        labels = {}
        for component_type in TRAINABLE_TYPES:
            matches = [
                label for label, item in controller_spec["components"].items()
                if item["type"] == component_type
            ]
            if len(matches) != 1:
                raise ValueError(
                    "The staged recipe requires exactly one "
                    f"{component_type} component, found {matches}"
                )
            labels[component_type] = matches[0]
        self.labels = labels
        connections = {
            (
                item["from_component"], item["from_output"],
                item["to_component"], item["to_input"],
            )
            for item in controller_spec["connections"]
        }
        required_connections = {
            (labels["SP_VAE"], "z", labels["LSTM"], "z"),
            (labels["LSTM"], "h", labels["MDN"], "h"),
        }
        if not required_connections.issubset(connections):
            raise ValueError(
                "The staged recipe requires the graph path SP_VAE.z -> "
                "LSTM.z -> LSTM.h -> MDN.h"
            )

        encoder_item = controller_spec["components"][labels["SP_VAE"]]
        encoder_exp = encoder_item["exp"]
        self.encoder_exp = encoder_exp
        self.sensor_exp = encoder_item["sensor_exp"]
        self.vae_encoder = ConvVAENeoEncoder(self.sensor_exp)

        lstm_exp = controller_spec["components"][labels["LSTM"]]["exp"]
        self.lstm = ResidualLSTM(
            lstm_exp["input_size"], lstm_exp["hidden_size"],
            lstm_exp["num_layers"],
        )
        self.sequence_length = lstm_exp["sequence_length"]
        mdn_exp = controller_spec["components"][labels["MDN"]]["exp"]
        self.mdn = MDN(mdn_exp)
        self._modules_by_label = {
            labels["SP_VAE"]: self.vae_encoder,
            labels["LSTM"]: self.lstm,
            labels["MDN"]: self.mdn,
        }
        self._trainable_labels = set()

    @property
    def component_labels(self):
        return tuple(self._modules_by_label)

    def component_module(self, label):
        try:
            return self._modules_by_label[label]
        except KeyError as error:
            raise KeyError(f"Unknown trainable component {label!r}") from error

    def architecture_signature(self, label):
        module = self.component_module(label)
        if module is self.vae_encoder:
            return module.architecture_signature()
        if module is self.lstm:
            return {
                "type": "ResidualLSTM", "input_size": module.input_size,
                "hidden_size": module.hidden_size,
                "num_layers": module.num_layers,
                "sequence_length": self.sequence_length,
            }
        item = self.controller_spec["components"][label]["exp"]
        return {
            "type": "MDN", "input_dim": item["input_dim"],
            "hidden_size": item["hidden_size"],
            "output_dim": item["output_dim"],
            "num_gaussians": item["num_gaussians"],
        }

    def load_component_state(self, label, payload, *, full_vae=False):
        module = self.component_module(label)
        if module is self.vae_encoder and full_vae:
            module.load_vae_state_dict(payload)
        else:
            module.load_state_dict(_unwrap_state(payload), strict=True)

    def component_state_dict(self, label):
        return self.component_module(label).state_dict()

    def set_trainable(self, labels):
        labels = set(labels)
        unknown = labels - set(self.component_labels)
        if unknown:
            raise KeyError(f"Unknown trainable components: {sorted(unknown)}")
        self._trainable_labels = labels
        for label, module in self._modules_by_label.items():
            enabled = label in labels
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)
            module.train(enabled)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for label, module in self._modules_by_label.items():
                module.train(label in self._trainable_labels)
        return self

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Controller training input must be a torch.Tensor")
        if images.ndim != 5:
            raise ValueError(
                "Controller training input must have shape [batch, time, "
                "channels, height, width]"
            )
        batch, time, channels, height, width = images.shape
        if time != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {time}"
            )
        latent = self.vae_encoder(
            images.reshape(batch * time, channels, height, width)
        ).reshape(batch, time, -1)
        feature = self.lstm(latent)
        output = self.mdn(feature)
        if not all(torch.isfinite(value).all() for value in output):
            raise FloatingPointError("Controller model produced non-finite output")
        return output
