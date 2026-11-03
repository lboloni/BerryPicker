"""Joint-space positions of the Franka Emika Panda."""

from robot.joint_position import JointPosition


class FrankaPosition(JointPosition):
    """A position of the Franka Panda, as the joints of a robot_franka exp/run."""

    ROBOT_NAME = "franka"
