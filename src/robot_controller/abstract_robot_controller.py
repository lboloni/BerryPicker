"""
abstract_robot_controller.py

Contains AbstractRobotController, the root class for the robot controllers, an architecture which drives the robot
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from abstract_rcco import AbstractRCComponent


class AbstractRobotController:
    """The root class for robot controllers (roco-s). 
    It follows an asynchronous model. Various external entitites are adding observations, commands etc to it. 

    The main aspects of the model are:
    - A robot controller rc is a **directed graph** of rcco-s
    - A collection of rcco are specified in the exp, they have their internal name (related to exp), also a label internal to the ARC
    - The state of the rc, is the collection of the states of the rcco
    - The edges of the rc are outputs of rccos connected to inputs of other rccos
    - Inputs of the rc, are inputs that are received, such as camera input, remote control etc. These are implemented as rcco-s of a specific kind - RCCO_Input. 
    - Outputs of the rc, are things we want to read out. They are implemented as an RCCO_Output object. A specific example of the input is 
    - The final output is a command sent to the robot (real or simulated)
    """
    def __init__(self, exp_roco):
        """Initialize the various components of based on the passed exp"""
        self.exp = exp_roco
        self.rccos = {} # dictionary of rccos
        self.connections = [] # list of connections between rccos

    def receive_input(self, label, value, time=None):
        """Receives an input at a specified time"""
        self.inputs[label] = {"value": value, "time": time}
        # FIXME: follow the connections, and make things dirty as needed

    def add_rcco(self, label, rcco: AbstractRCComponent):
        """Adds an rcco, that is currently not connected to anything"""
        self.rccos[label] = rcco

    def add_connection(self, from_rcco, from_output, to_rcco, to_input):
        """Creates a connection between the output of an rcco to an input of another. The parameters are all labels"""
        self.connections.append({"from_rcco": from_rcco, "from_output": from_output, "to_rcco": to_rcco, "to_input": to_input})

    def save(self):
        """Saves all the components"""
        for rcco in self.components:
            rcco.save()

    def load(self):
        """Loads all the components"""
        for rcco in self.components:
            rcco.load()

    def propagate(self):
        """Perform all the computations on the graph rcco-s, essentially propagating the data from the inputs to the outputs. This involves propagating on the rcco-s and performing the data transfers."""
        

