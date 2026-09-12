"""Validated directed-graph execution for robot-controller components."""

from collections import defaultdict, deque
from pathlib import Path

import torch

from exp_run_config import Config
from robot_controller.abstract_robot_controller import AbstractRobotController
from robot_controller import rcco_factory


Config.PROJECTNAME = "BerryPicker"


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_size(exp):
    size = exp.get("size")
    return None if size is None else _positive_int(size, "size")


def _component_interface(exp, load_experiment):
    rcco_type = rcco_factory.component_type(exp)
    if rcco_type == "Input":
        return {}, {"input": _optional_size(exp)}
    if rcco_type == "Output":
        return {"output": _optional_size(exp)}, {}
    if rcco_type in {"SP_VAE", "SP_CNN"}:
        sp_exp = load_experiment(exp["sp_experiment"], exp["sp_run"])
        latent_size = _positive_int(sp_exp["latent_size"], "latent_size")
        return {"image": None}, {"z": latent_size}
    if rcco_type == "LSTM":
        input_size = _positive_int(exp["input_size"], "input_size")
        hidden_size = _positive_int(exp["hidden_size"], "hidden_size")
        return {"z": input_size}, {"h": hidden_size}
    if rcco_type == "MDN":
        input_size = _positive_int(exp["input_dim"], "input_dim")
        output_size = _positive_int(exp["output_dim"], "output_dim")
        num_gaussians = _positive_int(exp["num_gaussians"], "num_gaussians")
        distribution_size = output_size * num_gaussians
        return {"h": input_size}, {
            "mu": distribution_size,
            "sigma": distribution_size,
            "pi": distribution_size,
            "a": output_size,
        }
    if rcco_type == "Z-combinator":
        input_size = exp.get("input_size")
        output_size = exp.get("output_size")
        if input_size is not None:
            input_size = _positive_int(input_size, "input_size")
        if output_size is not None:
            output_size = _positive_int(output_size, "output_size")
        return {"z1": input_size}, {"z": output_size}
    raise ValueError(f"Unknown rcco type {rcco_type!r}")


def _topological_order(labels, connections):
    indegree = {label: 0 for label in labels}
    outgoing = defaultdict(list)
    for connection in connections:
        source = connection["from_component"]
        destination = connection["to_component"]
        indegree[destination] += 1
        outgoing[source].append(destination)
    ready = deque(label for label in labels if indegree[label] == 0)
    order = []
    while ready:
        label = ready.popleft()
        order.append(label)
        for destination in outgoing[label]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    if len(order) != len(indegree):
        raise ValueError("Robot-controller connections must form an acyclic graph")
    return order


def load_controller_spec(exp_roco, experiment_loader=None):
    """Resolve and validate a graph without constructing models or loading weights."""
    load_experiment = experiment_loader or Config().get_experiment
    component_refs = exp_roco["components"]
    if not isinstance(component_refs, dict) or not component_refs:
        raise ValueError("Controller exp['components'] must be a nonempty mapping")

    components = {}
    for label, reference in component_refs.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Controller component labels must be nonempty strings")
        if not isinstance(reference, dict) or "run" not in reference:
            raise ValueError(f"Component {label!r} must specify a run")
        experiment = reference.get("exp", "robot_controller")
        component_exp = load_experiment(experiment, reference["run"])
        inputs, outputs = _component_interface(component_exp, load_experiment)
        resolved = {
            "experiment": experiment,
            "run": reference["run"],
            "type": rcco_factory.component_type(component_exp),
            "inputs": inputs,
            "outputs": outputs,
            "exp": component_exp,
        }
        if resolved["type"] in {"SP_VAE", "SP_CNN"}:
            resolved["sensor_exp"] = load_experiment(
                component_exp["sp_experiment"], component_exp["sp_run"]
            )
        components[label] = resolved

    connections = exp_roco["connections"]
    if not isinstance(connections, list):
        raise ValueError("Controller exp['connections'] must be a list")
    required_fields = {
        "from_component", "from_output", "to_component", "to_input"
    }
    destinations = set()
    validated_connections = []
    for connection in connections:
        if not isinstance(connection, dict) or set(connection) != required_fields:
            raise ValueError(
                "Every connection must contain exactly from_component, "
                "from_output, to_component, and to_input"
            )
        connection = dict(connection)
        source_label = connection["from_component"]
        destination_label = connection["to_component"]
        if source_label not in components:
            raise KeyError(f"Unknown source component {source_label!r}")
        if destination_label not in components:
            raise KeyError(f"Unknown destination component {destination_label!r}")
        source_port = connection["from_output"]
        destination_port = connection["to_input"]
        if source_port not in components[source_label]["outputs"]:
            raise KeyError(f"Unknown output port {source_label}.{source_port}")
        if destination_port not in components[destination_label]["inputs"]:
            raise KeyError(
                f"Unknown input port {destination_label}.{destination_port}"
            )
        destination = (destination_label, destination_port)
        if destination in destinations:
            raise ValueError(
                f"Input port {destination_label}.{destination_port} has "
                "multiple writers"
            )
        destinations.add(destination)
        source_size = components[source_label]["outputs"][source_port]
        destination_size = components[destination_label]["inputs"][destination_port]
        if source_size is not None and destination_size is not None \
                and source_size != destination_size:
            raise ValueError(
                f"Connection {source_label}.{source_port} ({source_size}) does not "
                f"match {destination_label}.{destination_port} ({destination_size})"
            )
        validated_connections.append(connection)

    order = _topological_order(components, validated_connections)
    return {
        "name": exp_roco.get("name", "RobotController"),
        "components": components,
        "connections": validated_connections,
        "topological_order": order,
    }


class GraphRobotController(AbstractRobotController):
    """Execute a validated acyclic graph whenever external inputs change."""

    def __init__(
        self, exp_roco, *, experiment_loader=None, component_factory=None,
        bundle_path=None,
    ):
        if bundle_path is not None and component_factory is not None:
            raise ValueError("bundle_path cannot be combined with component_factory")
        self.spec = load_controller_spec(exp_roco, experiment_loader)
        resolved = {
            label: component["exp"]
            for label, component in self.spec["components"].items()
        }
        factory = component_factory
        if bundle_path is not None:
            sensor_experiments = {
                id(item["exp"]): item.get("sensor_exp")
                for item in self.spec["components"].values()
            }
            factory = lambda exp: rcco_factory.create_component(
                exp, load_state=False,
                sensor_exp=sensor_experiments.get(id(exp)),
            )
        super().__init__(
            exp_roco,
            experiment_loader=experiment_loader,
            component_factory=factory,
            resolved_components=resolved,
        )
        self._order = self.spec["topological_order"]
        self._outgoing = defaultdict(list)
        for connection in self.connections:
            self._outgoing[connection["from_component"]].append(connection)
        if bundle_path is not None:
            self.load_bundle(bundle_path)

    def load_bundle(self, bundle_path):
        """Load recipe-owned component states instead of configured sources."""
        path = Path(bundle_path)
        if not path.is_file():
            raise FileNotFoundError(f"Controller bundle does not exist: {path}")
        bundle = torch.load(
            path, map_location=Config().runtime["device"], weights_only=True
        )
        if bundle.get("schema_version") != 1:
            raise ValueError("Unsupported controller bundle schema")
        states = bundle.get("component_state_dicts")
        architectures = bundle.get("architectures")
        if not isinstance(states, dict) or not isinstance(architectures, dict):
            raise ValueError("Controller bundle lacks component states or signatures")
        neural_labels = {
            label for label, item in self.spec["components"].items()
            if item["type"] in {"SP_VAE", "LSTM", "MDN"}
        }
        if set(states) != neural_labels or set(architectures) != neural_labels:
            raise ValueError(
                "Controller bundle components do not match controller graph"
            )
        for label in neural_labels:
            component = self.components[label]
            signature = component.architecture_signature()
            if signature != architectures[label]:
                raise ValueError(
                    f"Bundle architecture does not match component {label!r}"
                )
            component.model.load_state_dict(states[label], strict=True)
            component.model.eval()
        self.bundle_path = path
        return path

    def _route_outputs(self, label):
        source = self.components[label]
        for connection in self._outgoing[label]:
            value = source.outputs[connection["from_output"]]
            if value is None:
                raise RuntimeError(
                    f"Component {label!r} did not produce connected output "
                    f"{connection['from_output']!r}"
                )
            destination = self.components[connection["to_component"]]
            destination.set_input(connection["to_input"], value)

    def propagate(self):
        for label in self._order:
            component = self.components[label]
            if isinstance(component, rcco_factory.RCCO_Input):
                if component.dirty:
                    self._route_outputs(label)
                    component.dirty = False
                continue
            if not component.dirty or not component.inputs_ready():
                continue
            produced_output = component.propagate()
            component.dirty = False
            if produced_output is not False:
                self._route_outputs(label)
        return {
            label: component.read_output()
            for label, component in self.components.items()
            if isinstance(component, rcco_factory.RCCO_Output)
        }

    def read_output(self, label):
        if label not in self.components:
            raise KeyError(f"Unknown output component {label!r}")
        component = self.components[label]
        if not isinstance(component, rcco_factory.RCCO_Output):
            raise TypeError(f"Component {label!r} is not an external output")
        return component.read_output()
