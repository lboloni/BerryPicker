"""Choosing the position class of a robot from its exp/run.

The robot exp/run names the robot (robot_name) and, optionally, the key under
which the demonstrations record its positions (action_key, by default
rc-position-target, the key of the AL5D). The WidowX and the WidowX AI are
not here yet: their demonstrations record commands, not positions.
"""

from robot.al5d import RobotPosition
from robot.franka import FrankaPosition
from robot.g1 import G1ArmPosition
from robot.inspire_hand import InspireHandPosition

POSITION_CLASSES = {
    "al5d": RobotPosition,
    "franka": FrankaPosition,
    "g1_arm": G1ArmPosition,
    "inspire_hand": InspireHandPosition,
}


def create_position(exp_robot, values=None):
    """The position of the robot of exp_robot, with the given values (or the default)"""
    return POSITION_CLASSES[exp_robot["robot_name"]](exp_robot, values)


def position_fields(exp_robot):
    """The fields of the normalized position vector of the robot, in order"""
    if exp_robot["robot_name"] == "al5d":
        return list(RobotPosition.FIELDS)
    return list(exp_robot["JOINTS"])


def robot_output_size(exp_robot):
    """The size of the normalized position vector, the output_size of the models predicting it"""
    return len(position_fields(exp_robot))


def action_key(exp_robot):
    """The key under which the demonstrations record the positions of the robot"""
    return exp_robot.get("action_key", "rc-position-target")
