"""High-level WidowX AI pose and command representations.

The WidowX AI pose has the same fields as the WidowX pose (meters, and roll,
pitch, yaw in radians), so that the Xbox and AutoMove controllers and the
recorded data have the same shape for both arms. The conversion to the
angle-axis representation of the trossen_arm driver is in pose_to_driver()
and pose_from_driver().
"""

from math import isfinite

import numpy as np
from scipy.spatial.transform import Rotation

from robot.widowx.position import WidowXCommand, WidowXPose


class WXAIPose(WidowXPose):
    """An end-effector pose of the WidowX AI in meters and radians."""

    ROBOT_NAME = "wxai"

    def __str__(self):
        return "WidowX AI pose:\n" + "".join(
            f" {field}: {self.values[field]:.4f}\n" for field in self.FIELDS
        )


class WXAICommand(WidowXCommand):
    """A WidowX AI pose target and a gripper action.

    The gripper actions are the ones of the WidowX: grasp closes and release
    opens the gripper with an effort of gripper_pressure times the
    gripper_max_effort of the robot exprun. If joints is given (six arm
    joints in rad and the gripper in m), the robot moves in joint space and
    the pose is ignored; this is the command of the leader arm.
    """

    def __init__(self, pose, gripper_action="hold", gripper_pressure=None, joints=None):
        super().__init__(pose, gripper_action, gripper_pressure)
        if not isinstance(pose, WXAIPose):
            raise TypeError("WXAICommand pose must be a WXAIPose")
        if joints is not None:
            joints = [float(value) for value in joints]
            if len(joints) != 7 or not all(isfinite(value) for value in joints):
                raise ValueError("WXAICommand joints must be 7 finite values")
        self.joints = joints

    def __copy__(self):
        return WXAICommand(self.pose, self.gripper_action, self.gripper_pressure, self.joints)

    def as_dict(self):
        values = super().as_dict()
        values["joints"] = self.joints
        return values


def pose_to_driver(pose):
    """The trossen_arm Cartesian vector of a pose: translation and angle-axis."""
    rotation = Rotation.from_euler("xyz", [pose["roll"], pose["pitch"], pose["yaw"]])
    return np.concatenate([[pose["x"], pose["y"], pose["z"]], rotation.as_rotvec()])


def pose_from_driver(exp, values):
    """The pose of a trossen_arm Cartesian vector."""
    values = np.asarray(values, dtype=float)
    roll, pitch, yaw = Rotation.from_rotvec(values[3:6]).as_euler("xyz")
    return WXAIPose(exp, {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    })
