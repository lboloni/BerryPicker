"""A kinematic stand-in for trossen_arm.TrossenArmDriver, for machines without the arm.

It implements the subset of the driver API used by PositionController, with
the method names and arguments of trossen_arm 1.11. Motion is instantaneous;
the Cartesian moves are solved with the IK of the MuJoCo model. How the real
driver reports an unreachable Cartesian target is not known yet (see
PLAN-WXAI.md, problems); the fake raises RuntimeError.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import ARM_JOINTS, WXAIKinematics


class FakeTrossenArmDriver:
    GRIPPER_OPEN = 0.04

    def __init__(self, exp):
        self.kinematics = WXAIKinematics(exp)
        self.configured = False
        self.joints = np.asarray(exp["SLEEP_JOINTS"], dtype=float)
        self.gripper = 0.0
        self.modes = {"arm": None, "gripper": None}
        self.calls = []

    def configure(self, model, end_effector, serv_ip, clear_error, timeout=20.0):
        self.calls.append(("configure", serv_ip, clear_error))
        self.configured = True

    def get_is_configured(self):
        return self.configured

    def get_num_joints(self):
        return ARM_JOINTS + 1

    def set_arm_modes(self, mode):
        self.modes["arm"] = mode

    def set_gripper_mode(self, mode):
        self.modes["gripper"] = mode

    def set_all_modes(self, mode):
        self.modes = {"arm": mode, "gripper": mode}

    def get_arm_positions(self):
        return list(self.joints)

    def get_gripper_position(self):
        return self.gripper

    def get_all_positions(self):
        return list(self.joints) + [self.gripper]

    def get_all_efforts(self):
        return [0.0] * (ARM_JOINTS + 1)

    def get_all_external_efforts(self):
        return [0.0] * (ARM_JOINTS + 1)

    def get_cartesian_positions(self):
        position, rotation = self.kinematics.forward(self.joints)
        return list(position) + list(Rotation.from_matrix(rotation).as_rotvec())

    def set_cartesian_positions(self, goal_positions, interpolation_space, goal_time=2.0,
                                blocking=True, goal_feedforward_velocities=None,
                                goal_feedforward_accelerations=None, num_trajectory_check_samples=0):
        goal = np.asarray(goal_positions, dtype=float)
        joints, reached = self.kinematics.inverse(
            goal[:3], Rotation.from_rotvec(goal[3:]).as_matrix(), self.joints)
        if not reached:
            raise RuntimeError(f"Fake WidowX AI driver: Cartesian goal {goal} is unreachable")
        self.calls.append(("set_cartesian_positions", goal_time, blocking))
        self.joints = joints

    def set_arm_positions(self, goal_positions, goal_time=2.0, blocking=True,
                          goal_feedforward_velocities=None, goal_feedforward_accelerations=None):
        self.calls.append(("set_arm_positions", goal_time, blocking))
        self.joints = np.clip(np.asarray(goal_positions, dtype=float),
                              self.kinematics.lower, self.kinematics.upper)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True,
                          goal_feedforward_velocities=None, goal_feedforward_accelerations=None):
        self.set_arm_positions(goal_positions[:ARM_JOINTS], goal_time, blocking)
        self.gripper = min(self.GRIPPER_OPEN, max(0.0, float(goal_positions[ARM_JOINTS])))

    def set_all_external_efforts(self, goal_external_efforts, goal_time=2.0, blocking=True):
        self.calls.append(("set_all_external_efforts", list(goal_external_efforts)))

    def set_gripper_external_effort(self, goal_external_effort, goal_time=2.0, blocking=True):
        self.calls.append(("set_gripper_external_effort", goal_external_effort))
        self.gripper = self.GRIPPER_OPEN if goal_external_effort > 0 else 0.0

    def cleanup(self, reboot_controller=False):
        self.calls.append(("cleanup",))
        self.configured = False
