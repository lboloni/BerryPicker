"""Batched training model for encoder -> [LSTM] -> head controller graphs.

See robot_controller/DESIGN-BehaviorCloningFlow.md (Phase 3).
"""

import torch
from torch import nn

from robot_controller.rcco_lstm import RCCO_LSTM
from robot_controller.rcco_mdn import RCCO_MDN
from robot_controller.rcco_mlp import RCCO_MLP
from robot_controller.rcco_sp_cnn import RCCO_SP_CNN, _unwrap_state
from robot_controller.rcco_sp_vae import RCCO_SP_VAE


# The component classes building the trainable modules, by role and type
ENCODER_COMPONENTS = {"SP_CNN": RCCO_SP_CNN, "SP_VAE": RCCO_SP_VAE}
HEAD_COMPONENTS = {"MLP": RCCO_MLP, "MDN": RCCO_MDN}
# The input and output port of each component type on the chain
PORTS = {"SP_CNN": (None, "z"), "SP_VAE": (None, "z"), "LSTM": ("z", "h"),
         "MLP": ("z", "a"), "MDN": ("h", "a")}


def _follow_chain(controller_spec):
    """Return the labels of the encoder, the LSTM (or None), and the head of
    the graph path encoder -> [LSTM] -> head."""
    components = controller_spec["components"]
    encoders = [label for label, item in components.items()
                if item["type"] in ENCODER_COMPONENTS]
    if len(encoders) != 1:
        raise ValueError(
            f"A controller chain requires exactly one encoder, found {encoders}")
    successors = {}
    for item in controller_spec["connections"]:
        source = (item["from_component"], item["from_output"])
        successors.setdefault(source, []).append(
            (item["to_component"], item["to_input"]))

    def next_label(label):
        source = (label, PORTS[components[label]["type"]][1])
        targets = [
            (target, port) for target, port in successors.get(source, [])
            if components[target]["type"] in PORTS
            and PORTS[components[target]["type"]][0] == port
        ]
        if len(targets) != 1:
            raise ValueError(
                f"Component {label!r} must feed exactly one LSTM or head, "
                f"found {targets}")
        return targets[0][0]

    following = next_label(encoders[0])
    core = None
    if components[following]["type"] == "LSTM":
        core, following = following, next_label(following)
    if components[following]["type"] not in HEAD_COMPONENTS:
        raise ValueError(
            f"The chain must end in an MLP or MDN head, found {following!r}")
    return encoders[0], core, following


class ChainTrainingModel(nn.Module):
    """A differentiable realization of the graph path
    encoder (SP_CNN | SP_VAE) -> [LSTM] -> head (MLP | MDN).

    The modules are built by their components with load_state=False, so the
    exported states load into the same components at runtime."""

    def __init__(self, controller_spec):
        super().__init__()
        self.controller_spec = controller_spec
        encoder, core, head = _follow_chain(controller_spec)
        self.labels = {"encoder": encoder, "core": core, "head": head}
        components = controller_spec["components"]

        encoder_item = components[encoder]
        self.sensor_exp = encoder_item["sensor_exp"]
        self._components = {encoder: ENCODER_COMPONENTS[encoder_item["type"]](
            encoder_item["exp"], load_state=False, sensor_exp=self.sensor_exp)}
        if core is not None:
            self._components[core] = RCCO_LSTM(
                components[core]["exp"], load_state=False)
        self.head_type = components[head]["type"]
        self._components[head] = HEAD_COMPONENTS[self.head_type](
            components[head]["exp"], load_state=False)

        self.encoder = self._components[encoder].model
        self.core = None if core is None else self._components[core].model
        self.head = self._components[head].model
        # None (no LSTM), "sliding_window", or "stateful"
        self.context_mode = (
            None if core is None else self._components[core].context_mode)
        self.sequence_length = (
            1 if core is None else self._components[core].sequence_length)
        self.stochastic = self.head_type == "MDN"
        self.output_size = (
            self._components[head].output_size)
        self._modules_by_label = {
            label: component.model
            for label, component in self._components.items()}
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
        if label not in self._components:
            raise KeyError(f"Unknown trainable component {label!r}")
        return self._components[label].architecture_signature()

    def load_component_state(self, label, payload, *, full_vae=False):
        module = self.component_module(label)
        if full_vae:
            # gathers only the encoder weights from a full VAE checkpoint
            module.load_vae_state_dict(payload)
        else:
            module.load_state_dict(_unwrap_state(payload), strict=True)

    def component_state_dict(self, label):
        return self.component_module(label).state_dict()

    def encoder_trainable(self):
        return self.labels["encoder"] in self._trainable_labels

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

    def encode(self, images):
        """[batch, time, channels, height, width] -> [batch, time, latent]"""
        if not isinstance(images, torch.Tensor):
            raise TypeError("Controller training input must be a torch.Tensor")
        expected = (3, *self.sensor_exp["image_size"])
        if images.ndim != 5 or tuple(images.shape[2:]) != expected:
            raise ValueError(
                f"Controller training input must have shape [batch, time, "
                f"{expected[0]}, {expected[1]}, {expected[2]}], got "
                f"{tuple(images.shape)}")
        batch, time = images.shape[:2]
        return self.encoder.encode(
            images.reshape(batch * time, *expected)).reshape(batch, time, -1)

    def forward(self, inputs, state=None):
        """Map images [B, T, C, H, W] or cached latents [B, T, Z] to per-step
        head outputs, and return (outputs, state). The outputs are [B, T, O]
        for an MLP head and (mu, sigma, pi), each [B, T, O, G], for an MDN
        head. The state is the recurrent state after the last step (None
        without an LSTM)."""
        latent = inputs if inputs.ndim == 3 else self.encode(inputs)
        batch, time = latent.shape[:2]
        if self.core is None:
            if time != 1:
                raise ValueError("A controller without an LSTM takes one frame")
            feature = latent
        else:
            feature, state = self.core(latent, state)
        output = self.head(feature.reshape(batch * time, -1))
        if self.stochastic:
            output = tuple(
                value.reshape(batch, time, *value.shape[1:]) for value in output)
            values = output
        else:
            output = output.reshape(batch, time, -1)
            values = (output,)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("Controller model produced non-finite output")
        return output, state
