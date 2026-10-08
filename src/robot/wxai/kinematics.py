"""Forward and inverse kinematics of the WidowX AI on its MuJoCo model.

The model is the wxai_follower.xml of the trossen_arm_mujoco package. Its
ee_site is 0.156 m in front of the flange, the tool frame of the
wxai_v0_follower end effector of the trossen_arm driver (t_flange_tool =
0.156062 m), so a pose here is meant to be the Cartesian position of the
driver. The base frame is the base_link of the model.
"""

import pathlib

import mujoco
import numpy as np
import trossen_arm_mujoco
from scipy.spatial.transform import Rotation


ASSETS = pathlib.Path(trossen_arm_mujoco.__file__).resolve().parent / "assets"
ARM_JOINTS = 6


def model_path(exp):
    """The MuJoCo model of the robot exprun, in the trossen_arm_mujoco assets."""
    return ASSETS / exp["mujoco_model"]


class WXAIKinematics:
    """Kinematics of the six arm joints, on a private MjData of the robot model."""

    def __init__(self, exp, model=None):
        self.exp = exp
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(model_path(exp)))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("ee_site").id
        self.lower = self.model.jnt_range[:ARM_JOINTS, 0].copy()
        self.upper = self.model.jnt_range[:ARM_JOINTS, 1].copy()

    def forward(self, joints):
        """The position and rotation matrix of the end effector for six arm joints."""
        self.data.qpos[:ARM_JOINTS] = joints[:ARM_JOINTS]
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.site].copy(),
                self.data.site_xmat[self.site].reshape(3, 3).copy())

    def inverse(self, position, rotation, initial_joints):
        """Damped least squares IK from the initial joints. Returns the six arm
        joints and whether the target is reached within the tolerances."""
        joints = np.clip(np.asarray(initial_joints[:ARM_JOINTS], dtype=float), self.lower, self.upper)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = self.exp["ik_damping"] ** 2 * np.eye(6)
        for _ in range(self.exp["ik_max_iterations"]):
            current_position, current_rotation = self.forward(joints)
            error = np.concatenate([
                position - current_position,
                Rotation.from_matrix(rotation @ current_rotation.T).as_rotvec(),
            ])
            if (np.linalg.norm(error[:3]) <= self.exp["ik_position_tolerance"]
                    and np.linalg.norm(error[3:]) <= self.exp["ik_rotation_tolerance"]):
                return joints, True
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jacobian = np.vstack([jacp, jacr])[:, :ARM_JOINTS]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
            joints = np.clip(joints + self.exp["ik_step"] * step, self.lower, self.upper)
        return joints, False

    def inverse_pose(self, pose, initial_joints):
        """IK of a WXAIPose."""
        position = np.array([pose["x"], pose["y"], pose["z"]])
        rotation = Rotation.from_euler("xyz", [pose["roll"], pose["pitch"], pose["yaw"]]).as_matrix()
        return self.inverse(position, rotation, initial_joints)
