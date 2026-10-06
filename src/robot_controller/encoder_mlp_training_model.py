"""Batched CNN--MLP controller model used by staged training recipes."""

import torch
from torch import nn

from robot_controller.rcco_mlp import RCCO_MLP
from robot_controller.rcco_sp_cnn import RCCO_SP_CNN, _unwrap_state


class CNNMLPTrainingModel(nn.Module):
    """A differentiable realization of an SP_CNN.z -> MLP.z graph path."""

    def __init__(self, controller_spec):
        super().__init__()
        self.controller_spec = controller_spec
        labels = {}
        for component_type in ("SP_CNN", "MLP"):
            matches = [
                label for label, item in controller_spec["components"].items()
                if item["type"] == component_type
            ]
            if len(matches) != 1:
                raise ValueError(
                    "The CNN-MLP recipe requires exactly one "
                    f"{component_type} component, found {matches}"
                )
            labels[component_type] = matches[0]
        required = (
            labels["SP_CNN"], "z", labels["MLP"], "z"
        )
        connections = {
            (
                item["from_component"], item["from_output"],
                item["to_component"], item["to_input"],
            )
            for item in controller_spec["connections"]
        }
        if required not in connections:
            raise ValueError("CNN--MLP training requires SP_CNN.z -> MLP.z")
        self.labels = labels

        cnn_item = controller_spec["components"][labels["SP_CNN"]]
        self.sensor_exp = cnn_item["sensor_exp"]
        self._cnn_component = RCCO_SP_CNN(
            cnn_item["exp"], load_state=False, sensor_exp=self.sensor_exp
        )
        mlp_item = controller_spec["components"][labels["MLP"]]
        self._mlp_component = RCCO_MLP(mlp_item["exp"], load_state=False)
        self.cnn = self._cnn_component.model
        self.mlp = self._mlp_component.model
        self.sequence_length = 1
        self.output_size = self._mlp_component.output_size
        self._modules_by_label = {
            labels["SP_CNN"]: self.cnn,
            labels["MLP"]: self.mlp,
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
        if label == self.labels["SP_CNN"]:
            return self._cnn_component.architecture_signature()
        if label == self.labels["MLP"]:
            return self._mlp_component.architecture_signature()
        raise KeyError(f"Unknown trainable component {label!r}")

    def load_component_state(self, label, payload, *, full_vae=False):
        if full_vae:
            raise ValueError("CNN--MLP training does not accept VAE checkpoints")
        self.component_module(label).load_state_dict(
            _unwrap_state(payload), strict=True
        )

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
            raise TypeError("CNN--MLP training input must be a torch.Tensor")
        if images.ndim != 5 or images.size(1) != 1:
            raise ValueError(
                "CNN--MLP training input must have shape "
                "[batch, 1, channels, height, width]"
            )
        expected = (3, *self.sensor_exp["image_size"])
        if tuple(images.shape[2:]) != expected:
            raise ValueError(
                f"CNN--MLP expected image shape {expected}, got "
                f"{tuple(images.shape[2:])}"
            )
        latent = self.cnn.encode(images[:, 0])
        action = self.mlp(latent)
        if tuple(action.shape) != (images.size(0), self.output_size):
            raise RuntimeError(
                f"CNN--MLP produced shape {tuple(action.shape)}; expected "
                f"[{images.size(0)}, {self.output_size}]"
            )
        if not torch.isfinite(action).all():
            raise FloatingPointError("CNN--MLP produced non-finite actions")
        return action
