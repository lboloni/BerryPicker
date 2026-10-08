"""A MuJoCo stand-in for trossen_arm.TrossenArmDriver.

It implements the subset of the driver API used by PositionController, with
the method names and arguments of trossen_arm 1.11, on the position
actuators of the trossen_arm_mujoco model. A command becomes a linear joint
trajectory over its goal time; advance(dt) steps the physics along it. A
blocking command advances the physics until the goal time.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from robot.wxai.kinematics import ARM_JOINTS, WXAIKinematics

from .mujoco_runtime import create_runtime, release_runtime


class MujocoTrossenArmDriver:

    def __init__(self, exp):
        self.exp = exp
        self.runtime = create_runtime(exp)
        self.kinematics = WXAIKinematics(exp)
        model = self.runtime.model
        joints = [f"joint_{index}" for index in range(ARM_JOINTS)] + ["left_carriage_joint"]
        self.qpos = np.array([model.joint(name).qposadr[0] for name in joints])
        self.actuators = np.array([model.actuator(f"joint_{index}").id for index in range(ARM_JOINTS)]
                                  + [model.actuator("left_gripper").id])
        self.gripper_open = model.actuator_ctrlrange[self.actuators[ARM_JOINTS], 1]
        self.configured = False
        self.modes = {"arm": None, "gripper": None}
        # one linear trajectory per actuator: start and goal value, start time and duration
        self.start = self._positions()
        self.goal = self.start.copy()
        self.start_time = np.zeros(ARM_JOINTS + 1)
        self.duration = np.zeros(ARM_JOINTS + 1)
        self.runtime.data.ctrl[self.actuators] = self.start

    def _positions(self):
        return self.runtime.data.qpos[self.qpos].copy()

    def _go(self, mask, goal, goal_time, blocking):
        now = self.runtime.data.time
        self.start[mask] = self.runtime.data.ctrl[self.actuators][mask]
        self.goal[mask] = goal
        self.start_time[mask] = now
        self.duration[mask] = goal_time
        if blocking:
            self.advance(goal_time)

    def advance(self, dt):
        """Step the physics by dt seconds along the joint trajectories."""
        steps = max(1, round(dt / self.runtime.model.opt.timestep))
        for _ in range(steps):
            elapsed = self.runtime.data.time - self.start_time
            fraction = np.where(self.duration > 0, np.clip(elapsed / np.maximum(self.duration, 1e-9), 0, 1), 1)
            self.runtime.data.ctrl[self.actuators] = self.start + fraction * (self.goal - self.start)
            self.runtime.step(self.runtime.model.opt.timestep)

    def configure(self, model, end_effector, serv_ip, clear_error, timeout=20.0):
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
        return list(self._positions()[:ARM_JOINTS])

    def get_gripper_position(self):
        return float(self._positions()[ARM_JOINTS])

    def get_all_positions(self):
        return list(self._positions())

    def get_all_efforts(self):
        return list(self.runtime.data.actuator_force[self.actuators])

    def get_all_external_efforts(self):
        return [0.0] * (ARM_JOINTS + 1)

    def get_cartesian_positions(self):
        position, rotation = self.kinematics.forward(self._positions()[:ARM_JOINTS])
        return list(position) + list(Rotation.from_matrix(rotation).as_rotvec())

    def set_cartesian_positions(self, goal_positions, interpolation_space, goal_time=2.0,
                                blocking=True, goal_feedforward_velocities=None,
                                goal_feedforward_accelerations=None, num_trajectory_check_samples=0):
        goal = np.asarray(goal_positions, dtype=float)
        joints, reached = self.kinematics.inverse(
            goal[:3], Rotation.from_rotvec(goal[3:]).as_matrix(), self._positions()[:ARM_JOINTS])
        if not reached:
            raise RuntimeError(f"MuJoCo WidowX AI driver: Cartesian goal {goal} is unreachable")
        self.set_arm_positions(joints, goal_time, blocking)

    def set_arm_positions(self, goal_positions, goal_time=2.0, blocking=True,
                          goal_feedforward_velocities=None, goal_feedforward_accelerations=None):
        mask = np.arange(ARM_JOINTS + 1) < ARM_JOINTS
        self._go(mask, np.asarray(goal_positions, dtype=float), goal_time, blocking)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True,
                          goal_feedforward_velocities=None, goal_feedforward_accelerations=None):
        goal = np.asarray(goal_positions, dtype=float).copy()
        goal[ARM_JOINTS] = min(self.gripper_open, max(0.0, goal[ARM_JOINTS]))
        self._go(np.ones(ARM_JOINTS + 1, dtype=bool), goal, goal_time, blocking)

    def set_gripper_external_effort(self, goal_external_effort, goal_time=2.0, blocking=True):
        """The simulated gripper is position controlled: a positive effort opens
        it fully, a negative one closes it until contact."""
        mask = np.arange(ARM_JOINTS + 1) == ARM_JOINTS
        self._go(mask, self.gripper_open if goal_external_effort > 0 else 0.0, goal_time, blocking)

    def cleanup(self, reboot_controller=False):
        self.configured = False
        release_runtime(self.exp["scene"])
