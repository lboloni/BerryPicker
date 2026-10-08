"""High-level controller delegating WidowX AI operations to the trossen_arm driver."""

from copy import copy
from math import isfinite
import logging
from time import monotonic

import trossen_arm

from .kinematics import WXAIKinematics
from .position import WXAICommand, WXAIPose, pose_from_driver, pose_to_driver


logger = logging.getLogger(__name__)


def create_driver(exp):
    """The trossen_arm driver, its kinematic fake, or its MuJoCo simulation."""
    if exp["driver"] == "fake":
        from .fake_driver import FakeTrossenArmDriver
        return FakeTrossenArmDriver(exp)
    if exp["driver"] == "mujoco":
        from .simulation.mujoco_driver import MujocoTrossenArmDriver
        return MujocoTrossenArmDriver(exp)
    if exp["driver"] != "trossen_arm":
        raise ValueError(f"Unsupported WidowX AI driver: {exp['driver']}")
    return trossen_arm.TrossenArmDriver()


class PositionController:
    """Control and observe a WidowX AI through ``trossen_arm.TrossenArmDriver``.

    The interface is that of robot.widowx.PositionController.
    """

    POSE = WXAIPose
    COMMAND = WXAICommand

    def __init__(self, exp, driver=None):
        self.exp = exp
        self.driver = driver if driver is not None else create_driver(exp)
        self.kinematics = WXAIKinematics(exp)
        self.started = False
        self.target = WXAIPose(exp)
        self.gripper_action = "hold"
        self.gripper_mode = None

    def start_robot(self):
        self.gripper_mode = None
        self.driver.configure(
            getattr(trossen_arm.Model, self.exp["model"]),
            getattr(trossen_arm.StandardEndEffector, self.exp["end_effector"]),
            self.exp["ip_address"],
            self.exp["clear_error"],
        )
        self.driver.set_arm_modes(trossen_arm.Mode.position)
        self._set_gripper_mode(trossen_arm.Mode.external_effort)
        self.started = True
        try:
            startup_pose = self.exp["startup_pose"]
            if startup_pose == "home":
                self.go_home()
            elif startup_pose == "sleep":
                self.go_sleep()
            elif startup_pose == "default":
                self.move(WXAICommand(WXAIPose(self.exp)), moving_time=self.exp["home_time"], blocking=True)
            elif startup_pose != "hold":
                raise ValueError(f"Unsupported WidowX AI startup_pose: {startup_pose}")
            self.target = self.get_position()
        except Exception:
            self.started = False
            self.driver.cleanup()
            raise

    def _require_started(self):
        if not self.started:
            raise RuntimeError("WidowX AI robot is not started")

    def get_position(self):
        self._require_started()
        return pose_from_driver(self.exp, self.driver.get_cartesian_positions())

    def get_target(self):
        return copy(self.target)

    def get_state(self):
        pose = self.get_position()
        return {
            "timestamp": monotonic(),
            "pose": pose.as_dict(),
            "joint_positions": [float(value) for value in self.driver.get_all_positions()],
            "joint_efforts": [float(value) for value in self.driver.get_all_efforts()],
            "joint_external_efforts": [float(value) for value in self.driver.get_all_external_efforts()],
            "gripper_action": self.gripper_action,
            "gripper_position": float(self.driver.get_gripper_position()),
        }

    def can_reach(self, pose):
        """IK check on the MuJoCo model, from the current joints, or from the
        sleep joints before the start (the driver has no joints before it is
        configured). The driver has no motion-free reachability check."""
        if not isinstance(pose, WXAIPose):
            raise TypeError("WidowX AI reachability requires a WXAIPose")
        pose.validate(self.exp)
        joints = self.driver.get_arm_positions() if self.started else self.exp["SLEEP_JOINTS"]
        _, reachable = self.kinematics.inverse_pose(pose, joints)
        return bool(reachable)

    def move(self, command, moving_time=None, blocking=True):
        self._require_started()
        if isinstance(command, WXAIPose):
            command = WXAICommand(command)
        if not isinstance(command, WXAICommand):
            raise TypeError("WidowX AI move requires a WXAICommand or WXAIPose")
        moving_time = self.exp["moving_time"] if moving_time is None else moving_time
        if command.joints is not None:
            self.move_joint_positions(command.joints, moving_time, blocking)
            return
        command.pose.validate(self.exp)
        if not self.can_reach(command.pose):
            raise ValueError(f"WidowX AI target is not reachable:\n{command.pose}")
        self.driver.set_cartesian_positions(
            pose_to_driver(command.pose),
            getattr(trossen_arm.InterpolationSpace, self.exp["interpolation_space"]),
            moving_time,
            blocking,
        )
        self._apply_gripper(command)
        self.target = copy(command.pose)

    def _set_gripper_mode(self, mode):
        """Grasp and release use external effort, joint commands position mode."""
        if self.gripper_mode != mode:
            self.driver.set_gripper_mode(mode)
            self.gripper_mode = mode

    def _apply_gripper(self, command):
        if command.gripper_action == "hold":
            return
        self._set_gripper_mode(trossen_arm.Mode.external_effort)
        pressure = self.exp["gripper_pressure"] if command.gripper_pressure is None \
            else command.gripper_pressure
        effort = pressure * self.exp["gripper_max_effort"]
        if command.gripper_action == "grasp":
            effort = -effort
        self.driver.set_gripper_external_effort(effort, self.exp["gripper_time"], False)
        self.gripper_action = command.gripper_action

    def command_gripper(self, effort, duration):
        """Apply a gripper external effort in N (positive opens) for duration s."""
        self._require_started()
        effort, duration = float(effort), float(duration)
        if not isfinite(effort) or not isfinite(duration) or duration < 0:
            raise ValueError("WidowX AI gripper effort must be finite and duration nonnegative")
        self.driver.set_gripper_external_effort(effort, duration, True)

    def move_joint_positions(self, joints, moving_time=None, blocking=True):
        """Move to six arm joints (rad), or six arm joints and the gripper (m)."""
        self._require_started()
        moving_time = self.exp["moving_time"] if moving_time is None else moving_time
        if len(joints) == 7:
            self._set_gripper_mode(trossen_arm.Mode.position)
            self.driver.set_all_positions(list(joints), moving_time, blocking)
        else:
            self.driver.set_arm_positions(list(joints), moving_time, blocking)
        self.target = self.get_position()

    def go_home(self, moving_time=None):
        self._require_started()
        self.move_joint_positions(
            self.exp["HOME_JOINTS"], self.exp["home_time"] if moving_time is None else moving_time)

    def go_sleep(self, moving_time=None):
        self._require_started()
        self.move_joint_positions(
            self.exp["SLEEP_JOINTS"], self.exp["home_time"] if moving_time is None else moving_time)

    def update(self, dt):
        """Called once per collection tick. The real arm runs by itself; the
        MuJoCo simulation is advanced by dt."""
        if self.exp["driver"] == "mujoco":
            self.driver.advance(dt)

    def stop_robot(self):
        self._require_started()
        try:
            shutdown_pose = self.exp["shutdown_pose"]
            if shutdown_pose == "home":
                self.go_home()
            elif shutdown_pose == "sleep":
                self.go_sleep()
            elif shutdown_pose != "hold":
                raise ValueError(f"Unsupported WidowX AI shutdown_pose: {shutdown_pose}")
        finally:
            self.started = False
            self.driver.set_all_modes(trossen_arm.Mode.idle)
            self.driver.cleanup()

