"""Strict headless simulation of the WidowX position-controller interface."""

from copy import copy
from math import isfinite
from time import monotonic

from .move import move_pose_by
from .position import WidowXCommand, WidowXPose


class SimulatedPositionController:
    def __init__(self, exp):
        self.exp = exp
        self.started = False
        self.pos = WidowXPose(exp)
        self.target = copy(self.pos)
        self.gripper_action = "hold"

    def _require_started(self):
        if not self.started:
            raise RuntimeError("Simulated WidowX robot is not started")

    def start_robot(self):
        if self.started:
            raise RuntimeError("Simulated WidowX robot is already started")
        self.started = True

    def stop_robot(self):
        self._require_started()
        self.started = False

    def get_position(self):
        self._require_started()
        return copy(self.pos)

    def get_target(self):
        return copy(self.target)

    def get_state(self):
        self._require_started()
        return {
            "timestamp": monotonic(),
            "pose": self.pos.as_dict(),
            "joint_positions": [],
            "gripper_action": self.gripper_action,
            "gripper_position": None,
        }

    def can_reach(self, pose):
        if not isinstance(pose, WidowXPose):
            raise TypeError("WidowX reachability requires a WidowXPose")
        pose.validate(self.exp)
        return True

    def move(self, command, moving_time=None, blocking=True):
        del moving_time, blocking
        self._require_started()
        if isinstance(command, WidowXPose):
            command = WidowXCommand(command)
        if not isinstance(command, WidowXCommand):
            raise TypeError("WidowX move requires a WidowXCommand or WidowXPose")
        command.pose.validate(self.exp)
        self.pos = copy(command.pose)
        self.target = copy(command.pose)
        if command.gripper_action != "hold":
            self.gripper_action = command.gripper_action

    def command_gripper(self, effort, duration):
        self._require_started()
        effort, duration = float(effort), float(duration)
        if not isfinite(effort) or not isfinite(duration) or duration < 0:
            raise ValueError("WidowX gripper effort must be finite and duration nonnegative")

    def move_cartesian(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        moving_time=None,
        wp_moving_time=0.2,
        wp_accel_time=0.1,
        wp_period=0.05,
    ):
        self._require_started()
        for name, value in (
            ("moving_time", moving_time),
            ("wp_moving_time", wp_moving_time),
            ("wp_accel_time", wp_accel_time),
            ("wp_period", wp_period),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or not isfinite(value) or value <= 0
            ):
                raise ValueError(f"Simulated WidowX {name} must be positive and finite")
        self.pos = move_pose_by(self.exp, self.pos, {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        })
        self.target = copy(self.pos)

    def go_home(self, moving_time=None):
        del moving_time
        self._require_started()
        self.pos = WidowXPose(self.exp)
        self.target = copy(self.pos)

    def go_sleep(self, moving_time=None):
        del moving_time
        self.go_home()
