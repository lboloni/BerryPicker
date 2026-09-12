"""Create robot-controller components from experiment/run configurations."""

from exp_run_config import Config

from robot_controller.abstract_rcco import AbstractRCComponent


Config.PROJECTNAME = "BerryPicker"


class RCCO_Input(AbstractRCComponent):
    """Expose a value received outside the graph through the ``input`` port."""

    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.outputs["input"] = None
        self.output_sizes["input"] = exp_rcco.get("size")
        self.time = None

    def receive_input(self, value, time=None):
        if value is None:
            raise ValueError("External controller input cannot be None")
        self.outputs["input"] = value
        self.time = time
        self.dirty = True

    def reset_context(self):
        self.time = None
        super().reset_context()


class RCCO_Output(AbstractRCComponent):
    """Expose a graph value to an external consumer."""

    def __init__(self, exp_rcco):
        super().__init__(exp_rcco)
        self.inputs["output"] = None
        self.input_sizes["output"] = exp_rcco.get("size")

    def read_output(self):
        return self.inputs["output"]


def component_type(exp):
    """Return the canonical component type or fail on a malformed exp/run."""
    try:
        value = exp["rcco-type"]
    except KeyError as error:
        raise KeyError("Robot component exp/run must define 'rcco-type'") from error
    if not isinstance(value, str) or not value:
        raise ValueError("'rcco-type' must be a nonempty string")
    return value


def create_component(exp, *, load_state=True, sensor_exp=None):
    """Construct the component selected by ``exp['rcco-type']``."""
    rcco_type = component_type(exp)
    if rcco_type == "Input":
        return RCCO_Input(exp)
    if rcco_type == "Output":
        return RCCO_Output(exp)
    if rcco_type == "SP_VAE":
        from robot_controller.rcco_sp_vae import RCCO_SP_VAE

        return RCCO_SP_VAE(
            exp, load_state=load_state, sensor_exp=sensor_exp
        )
    if rcco_type == "SP_CNN":
        from robot_controller.rcco_sp_cnn import RCCO_SP_CNN

        return RCCO_SP_CNN(exp)
    if rcco_type == "LSTM":
        from robot_controller.rcco_lstm import RCCO_LSTM

        return RCCO_LSTM(exp, load_state=load_state)
    if rcco_type == "MDN":
        from robot_controller.rcco_mdn import RCCO_MDN

        return RCCO_MDN(exp, load_state=load_state)
    if rcco_type == "Z-combinator":
        from robot_controller.rcco_z_combinator import RCCO_Z_Combinator

        return RCCO_Z_Combinator(exp)
    raise ValueError(f"Unknown rcco type {rcco_type!r}")
