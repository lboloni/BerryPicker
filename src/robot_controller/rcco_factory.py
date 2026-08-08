"""
rcco_factory.py

Creating different rcco-s based on the specification in the exp/run
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from abstract_rcco import AbstractRCComponent

class RCCO_Input(AbstractRCComponent):
    """An RCCO that implements an external input to the robot controller. Examples include camera inputs, sensors, remote control, proprioception etc.
    The input values are stored in an *output* called input (due to the way this fits in the graph)"""
    
    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.outputs["input"] = None
        self.time = None

    def receive_input(self, value, time=None):        
        """Receives an input and stores it into the output port"""
        self.outputs["input"] = value
        self.time = time

class RCCO_Output(AbstractRCComponent):
    """An RCCO that implements the external output of the robot controller. Examples include the control sent to the robot itself, visualization outputs etc"""

    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["output"] = None

    def read_output(self):
        return self.inputs["output"]


def create_component(exp):
    if exp["rcco-type"] == "Input":
        return RCCO_Input(exp)
    if exp["rcco-type"] == "Output":
        return RCCO_Output(exp)
    if exp["rcco-type"] == "SP_VAE":
        import robot_controller.rcco_sp_vae as rcco_sp_vae; return rcco_sp_vae.RCCO_SP_VAE(exp)
    if exp["rcco-type"] == "SP_CNN":
        import robot_controller.rcco_sp_cnn as rcco_sp_cnn; return rcco_sp_cnn.RCCO_SP_CNN(exp)
    if exp["rcco-rype"] == "LSTM":
        import robot_controller.rcco_robotdrive_lstm as rcco_robotdrive_lstm; return rcco_robotdrive_lstm.RCCO_LSTM(exp)
    if exp["rcco-rype"] == "Z-combinator":
        import rcco_z_combinator; return rcco_robotdrive_lstm.RCCO_Z_Combinator(exp)
    raise Exception(f"Unknown rcco type {exp['rcco-type']}")
