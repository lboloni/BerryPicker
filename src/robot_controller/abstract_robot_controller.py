"""
abstract_robot_controller.py

Contains AbstractRobotController, the root class for the robot controllers, an architecture which drives the robot
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from abc import ABC, abstractmethod
from robot_controller.abstract_rcco import AbstractRCComponent
from robot_controller import rcco_factory

class AbstractRobotController(ABC):
    """The root class for robot controllers (roco-s). 
    It follows an asynchronous model. Various external entitites are adding observations, commands etc to it. 

    The main aspects of the model are:
    - A robot controller rc is a **directed graph** of rcco-s
    - A collection of rcco are specified in the exp, they have their internal name (related to exp), also a label internal to the ARC
    - The state of the rc, is the collection of the states of the rcco
    - The edges of the rc are outputs of rccos connected to inputs of other rccos
    - Inputs of the rc, are inputs that are received, such as camera input, remote control etc. These are implemented as rcco-s of a specific kind - RCCO_Input. 
    - Outputs of the rc, are things we want to read out. They are implemented as an RCCO_Output object. A specific example of the output is the command sent to the robot (real or simulated). 
    """
    def __init__(
        self,
        exp_roco,
        *,
        experiment_loader=None,
        component_factory=None,
        resolved_components=None,
    ):
        """Initialize the various components of based on the passed exp"""
        self.exp = exp_roco
        self.components = {}  # dictionary of rccos
        self.component_experiments = {}
        self.connections = []  # list of connections between rccos
        load_experiment = experiment_loader or Config().get_experiment
        create_component = component_factory or rcco_factory.create_component
        for label, val in self.exp["components"].items():
            if resolved_components is None:
                exp_rcco = load_experiment(
                    val.get("exp", "robot_controller"), val["run"]
                )
            else:
                exp_rcco = resolved_components[label]
            rcco = create_component(exp_rcco)
            self.add_component(label, rcco)
            self.component_experiments[label] = exp_rcco
        for val in self.exp["connections"]:
            self.add_connection(**val)

    def receive_input(self, label, value, time=None):
        """Receives an input at a specified time"""
        if label not in self.components:
            raise KeyError(f"Unknown input component {label!r}")
        component = self.components[label]
        receive = getattr(component, "receive_input", None)
        if receive is None:
            raise TypeError(f"Component {label!r} is not an external input")
        receive(value, time=time)

    def add_component(self, label, component: AbstractRCComponent):
        """Adds an rcco, that is currently not connected to anything"""
        if label in self.components:
            raise ValueError(f"Duplicate component label {label!r}")
        if not isinstance(component, AbstractRCComponent):
            raise TypeError(f"Component {label!r} is not an AbstractRCComponent")
        self.components[label] = component

    def add_connection(self, from_component, from_output, to_component, to_input):
        """Connect one component output port to another component input."""
        if from_component not in self.components:
            raise KeyError(f"Unknown source component {from_component!r}")
        if to_component not in self.components:
            raise KeyError(f"Unknown destination component {to_component!r}")
        if from_output not in self.components[from_component].outputs:
            raise KeyError(
                f"Unknown output port {from_component}.{from_output}"
            )
        if to_input not in self.components[to_component].inputs:
            raise KeyError(f"Unknown input port {to_component}.{to_input}")
        self.connections.append({
            "from_component": from_component,
            "from_output": from_output,
            "to_component": to_component,
            "to_input": to_input,
        })

    def save(self):
        """Saves all the components"""
        for rcco in self.components.values():
            rcco.save()

    def load(self):
        """Loads all the components"""
        for rcco in self.components.values():
            rcco.load()

    def reset_context(self):
        """Reset runtime state for a new episode."""
        for rcco in self.components.values():
            rcco.reset_context()

    @abstractmethod
    def propagate(self):
        """Perform all the computations on the graph rcco-s, essentially propagating the data from the inputs to the outputs. This involves propagating on the rcco-s and performing the data transfers."""
        raise NotImplementedError
