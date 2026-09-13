"""Ordered tensor-model composition; concrete operations register separately."""

from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass
import json

from torch import nn

from training_harness.checkpoints import model_file


@dataclass(frozen=True)
class Operation:
    """Lifecycle hooks for an operation, all receiving its step configuration.

    ``build(step)`` constructs a module without loading source weights.
    Optional ``resolve(step)`` returns JSON-serializable settings with source
    architecture resolved. Optional ``initialize(module, step)`` imports source
    weights for new training only. External adapters own their runtime handles
    and must not register externally owned weights as child modules.
    """

    build: Callable
    resolve: Callable | None = None
    initialize: Callable | None = None


OPERATIONS = {}


def register_operation(name, build, *, resolve=None, initialize=None):
    """Register construction and optional new-training hooks for a step."""
    OPERATIONS[name] = Operation(build, resolve, initialize)


def resolve(values, reference):
    """Resolve a step result or a dotted field in a structured result."""
    value = values
    for field in reference.split("."):
        value = value[field]
    return value


class CompositeModel(nn.Module):
    """Execute named modules in configuration order, retaining autograd."""

    def __init__(self, exp):
        super().__init__()
        self.steps = deepcopy(exp["steps"])
        self.output = exp["output"]
        self.latent_size = exp["latent_size"]
        self.operations = nn.ModuleDict({
            step["name"]: OPERATIONS[step["operation"]].build(step)
            for step in self.steps
        })
        # ``frozen`` is optional for parameter-free operations.
        self.frozen = [step["name"] for step in self.steps if step.get("frozen", False)]
        for name in self.frozen:
            self.operations[name].requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name in self.frozen:
            self.operations[name].eval()
        return self

    def forward_steps(self, sensor_readings):
        """Return named results for auxiliary losses or inspection."""
        values = {"input": sensor_readings}
        for step in self.steps:
            inputs = [resolve(values, name) for name in step["inputs"]]
            values[step["name"]] = self.operations[step["name"]](*inputs)
        return values

    def encode(self, sensor_readings):
        return resolve(self.forward_steps(sensor_readings), self.output)

    def forward(self, sensor_readings):
        return self.encode(sensor_readings)


def configuration_file(exp):
    """Resolved architecture lives beside the run's configured model file."""
    return model_file(exp).with_suffix(".config.json")


def create_composite(exp):
    """Initialize a NEW training model and save its resolved configuration.

    Existing snapshots are not overwritten: use ``restore_composite`` for an
    existing run. The training harness owns model/optimizer checkpoint saving.
    """
    path = configuration_file(exp)
    if path.exists():
        raise FileExistsError(path)
    config = {
        key: deepcopy(exp[key])
        for key in ("class", "image_size", "latent_size", "steps", "output")
    }
    if exp["class"] == "CompositeMultiViewSensorProcessing":
        config.update(num_views=exp["num_views"], cameras=list(exp["cameras"]))
    for index, step in enumerate(config["steps"]):
        operation = OPERATIONS[step["operation"]]
        if operation.resolve is not None:
            config["steps"][index] = operation.resolve(step)
    serialized = json.dumps(config, indent=2)
    model = CompositeModel(config)
    for step in config["steps"]:
        operation = OPERATIONS[step["operation"]]
        if operation.initialize is not None:
            operation.initialize(model.operations[step["name"]], step)
    # Initialization hooks may alter mode; restore the freezing contract.
    model.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized + "\n")
    return model


def read_configuration(exp):
    with configuration_file(exp).open() as stream:
        return json.load(stream)


def restore_composite(exp):
    """Construct from the saved architecture; caller restores checkpoint state.

    No source-resolution or initialization hooks run. This works both for the
    composite alone and as a child of a task model restored by the harness.
    """
    return CompositeModel(read_configuration(exp))
