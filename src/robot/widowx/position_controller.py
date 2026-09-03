"""High-level controller delegating WidowX operations to Interbotix."""

from copy import copy
from math import isfinite
from time import monotonic

from .position import WidowXCommand, WidowXPose
from .runtime import get_default_runtime


class PositionController:
    """Control and observe a WidowX through ``InterbotixManipulatorXS``."""

    def __init__(self, exp, runtime=None, bot=None):
        self.exp = exp
        self.runtime = runtime or get_default_runtime()
        self.started = False
        self.target = WidowXPose(exp)
        self.gripper_action = "hold"
        if bot is None:
            self.bot = self.runtime.create_manipulator(
                self,
                robot_model=exp.get("robot_model", "wx250s"),
                robot_name=exp.get("interbotix_robot_name"),
                group_name=exp.get("group_name", "arm"),
                gripper_name=exp.get("gripper_name", "gripper"),
                moving_time=exp.get("moving_time", 2.0),
                accel_time=exp.get("accel_time", 0.3),
                gripper_pressure=exp.get("gripper_pressure", 0.5),
            )
        else:
            self.bot = bot
            self.runtime.register_manipulator(
                self, bot, exp.get("interbotix_robot_name", exp.get("robot_model", "wx250s"))
            )

    def start_robot(self):
        self.runtime.acquire(self)
        self.started = True
        try:
            startup_pose = self.exp.get("startup_pose", "hold")
            if startup_pose == "hold":
                self.bot.arm.capture_joint_positions()
            elif startup_pose == "home":
                self.go_home()
            elif startup_pose == "sleep":
                self.go_sleep()
            elif startup_pose == "default":
                self.move(WidowXCommand(WidowXPose(self.exp)))
            else:
                raise ValueError(f"Unsupported WidowX startup_pose: {startup_pose}")
            self.target = self.get_position()
        except Exception:
            self.started = False
            self.runtime.release(self)
            raise

    def _require_started(self):
        if not self.started:
            raise RuntimeError("WidowX robot is not started")

    def get_position(self):
        self._require_started()
        return self._pose_from_matrix(self.bot.arm.get_ee_pose())

    def _pose_from_matrix(self, matrix):
        roll, pitch, yaw = self.runtime.rotation_matrix_to_euler_angles(matrix[:3, :3])
        return WidowXPose(self.exp, {
            "x": float(matrix[0, 3]),
            "y": float(matrix[1, 3]),
            "z": float(matrix[2, 3]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        })

    def get_target(self):
        return copy(self.target)

    def get_state(self):
        pose = self.get_position()
        state = {
            "timestamp": monotonic(),
            "pose": pose.as_dict(),
            "joint_positions": [float(value) for value in self.bot.arm.get_joint_positions()],
            "gripper_action": self.gripper_action,
        }
        if hasattr(self.bot, "gripper"):
            state["gripper_position"] = float(self.bot.gripper.get_finger_position())
        return state

    @staticmethod
    def _pose_kwargs(pose):
        return {field: pose[field] for field in WidowXPose.FIELDS}

    def can_reach(self, pose):
        if not isinstance(pose, WidowXPose):
            raise TypeError("WidowX reachability requires a WidowXPose")
        pose.validate(self.exp)
        _, reachable = self.bot.arm.set_ee_pose_components(
            **self._pose_kwargs(pose), execute=False
        )
        return bool(reachable)

    def move(self, command, moving_time=None, blocking=True):
        self._require_started()
        if isinstance(command, WidowXPose):
            command = WidowXCommand(command)
        if not isinstance(command, WidowXCommand):
            raise TypeError("WidowX move requires a WidowXCommand or WidowXPose")
        command.pose.validate(self.exp)
        if command.gripper_action != "hold" and not hasattr(self.bot, "gripper"):
            raise RuntimeError("WidowX controller has no configured gripper")
        _, reachable = self.bot.arm.set_ee_pose_components(
            **self._pose_kwargs(command.pose),
            moving_time=moving_time,
            blocking=blocking,
        )
        if not reachable:
            raise ValueError(f"Interbotix could not reach WidowX target:\n{command.pose}")
        self._apply_gripper(command)
        self.target = copy(command.pose)

    def _apply_gripper(self, command):
        if command.gripper_action == "hold":
            return
        if not hasattr(self.bot, "gripper"):
            raise RuntimeError("WidowX controller has no configured gripper")
        if command.gripper_pressure is not None:
            self.bot.gripper.set_pressure(command.gripper_pressure)
        delay = self.exp.get("gripper_delay", 1.0)
        if command.gripper_action == "grasp":
            self.bot.gripper.grasp(delay=delay)
        else:
            self.bot.gripper.release(delay=delay)
        self.gripper_action = command.gripper_action

    def command_gripper(self, effort, duration):
        self._require_started()
        if not hasattr(self.bot, "gripper"):
            raise RuntimeError("WidowX controller has no configured gripper")
        effort, duration = float(effort), float(duration)
        if not isfinite(effort) or not isfinite(duration) or duration < 0:
            raise ValueError("WidowX gripper effort must be finite and duration nonnegative")
        self.bot.gripper.gripper_controller(effort, duration)

    def move_joint_positions(self, joints, moving_time=None, blocking=True):
        self._require_started()
        if not self.bot.arm.set_joint_positions(
            joints, moving_time=moving_time, blocking=blocking
        ):
            raise ValueError("Interbotix rejected WidowX joint positions")
        self.target = self._pose_from_matrix(self.bot.arm.get_ee_pose_command())

    def move_cartesian(self, **kwargs):
        self._require_started()
        if not self.bot.arm.set_ee_cartesian_trajectory(**kwargs):
            raise ValueError("Interbotix could not execute WidowX Cartesian trajectory")
        self.target = self._pose_from_matrix(self.bot.arm.get_ee_pose_command())

    def go_home(self, moving_time=None):
        self._require_started()
        if self.bot.arm.go_to_home_pose(moving_time=moving_time) is False:
            raise ValueError("Interbotix rejected the WidowX home pose")
        self.target = self._pose_from_matrix(self.bot.arm.get_ee_pose_command())

    def go_sleep(self, moving_time=None):
        self._require_started()
        if self.bot.arm.go_to_sleep_pose(moving_time=moving_time) is False:
            raise ValueError("Interbotix rejected the WidowX sleep pose")
        self.target = self._pose_from_matrix(self.bot.arm.get_ee_pose_command())

    def stop_robot(self):
        self._require_started()
        try:
            shutdown_pose = self.exp.get("shutdown_pose", "sleep")
            if shutdown_pose == "home":
                self.go_home()
            elif shutdown_pose == "sleep":
                self.go_sleep()
            elif shutdown_pose != "hold":
                raise ValueError(f"Unsupported WidowX shutdown_pose: {shutdown_pose}")
        finally:
            self.started = False
            self.runtime.release(self)
