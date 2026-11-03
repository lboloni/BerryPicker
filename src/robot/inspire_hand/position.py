"""Joint-space positions of the Inspire dexterous hand."""

from robot.joint_position import JointPosition


class InspireHandPosition(JointPosition):
    """A position of the Inspire hand, as the joints of a robot_inspire_hand exp/run."""

    ROBOT_NAME = "inspire_hand"
