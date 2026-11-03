"""Joint-space positions of an arm of the Unitree G1 humanoid."""

from robot.joint_position import JointPosition


class G1ArmPosition(JointPosition):
    """A position of the arm of the Unitree G1, as the joints of a robot_g1 exp/run."""

    ROBOT_NAME = "g1_arm"
