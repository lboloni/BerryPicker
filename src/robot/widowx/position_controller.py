"""High-level controller delegating WidowX operations to Interbotix."""

from copy import copy
from math import isfinite
import logging
from time import monotonic, perf_counter, sleep

from .position import WidowXCommand, WidowXPose
from .runtime import get_default_runtime


logger = logging.getLogger(__name__)


class PositionController:
    """Control and observe a WidowX through ``InterbotixManipulatorXS``."""

    POSE = WidowXPose
    COMMAND = WidowXCommand

    def __init__(self, exp, runtime=None, bot=None):
        self.exp = exp
        self.runtime = runtime or get_default_runtime()
        self.started = False
        self.target = WidowXPose(exp)
        self.gripper_action = "hold"
        # the joints where the arm rested at a ready start; the sleep pose goes there
        self.rest_joints = None
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
            elif startup_pose == "ready":
                # with the torque off, the arm rests on its cradle below the sleep joints
                if not self._at_joints(self.bot.arm.group_info.joint_sleep_positions,
                                       self.exp["sleep_tolerance"]):
                    raise RuntimeError("WidowX must start from the sleep pose; park it first")
                # the cradle is beyond the joint limits of the shoulder and the elbow
                info = self.bot.arm.group_info
                self.rest_joints = [
                    min(max(joint, low), high) for joint, low, high in zip(
                        self.bot.arm.get_joint_positions(),
                        info.joint_lower_limits, info.joint_upper_limits)
                ]
                # at torque on the motors go back to their last goal: slowly, not in 0.1 s
                self.bot.arm.set_trajectory_time(
                    moving_time=self.exp["transition_time"],
                    accel_time=self.exp["transition_time"] / 4,
                )
                self._set_pwm(self.exp["pwm_max"])
                self.bot.core.robot_torque_enable("group", "all", True)
                self.bot.arm.capture_joint_positions()
                self._transition(self.exp["READY_JOINTS"])
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
        """IK check against the joint position limits only, from the current joints,
        or from READY_JOINTS before the start (the arm will move from there).
        Interbotix would also reject solutions that cannot be reached within one
        moving_time; that check is left to move, so it is switched off here."""
        if not isinstance(pose, WidowXPose):
            raise TypeError("WidowX reachability requires a WidowXPose")
        pose.validate(self.exp)
        joints = self.bot.arm.get_joint_positions() if self.started else self.exp["READY_JOINTS"]
        moving_time = self.bot.arm.moving_time
        self.bot.arm.moving_time = float("inf")
        started = perf_counter()
        try:
            solution, reachable = self.bot.arm.set_ee_pose_components(
                **self._pose_kwargs(pose), custom_guess=joints, execute=False,
            )
        finally:
            self.bot.arm.moving_time = moving_time
        logger.info(
            "IK check: duration=%.4fs reachable=%s target=%s measured_joints=%s solution=%s",
            perf_counter() - started, reachable, pose.as_dict(), joints, solution,
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
            custom_guess=self.bot.arm.get_joint_positions(),
            moving_time=moving_time,
            blocking=blocking,
        )
        if not reachable:
            raise ValueError(
                f"Interbotix could not reach WidowX target:\n{command.pose}"
            )
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

    def _transition(self, joints):
        """Joint-space move with transition_time, then wait until the joints settle.
        The trajectory time is set before the move, since Interbotix checks the joint
        speeds with it; afterwards it goes back to the streaming moving_time."""
        self.bot.arm.set_trajectory_time(
            moving_time=self.exp["transition_time"],
            accel_time=self.exp["transition_time"] / 4,
        )
        if not self.bot.arm.set_joint_positions(joints, blocking=True):
            raise ValueError(f"Interbotix rejected the WidowX transition to {joints}")
        deadline = monotonic() + self.exp["settle_timeout"]
        while not self._at_joints(joints):
            if monotonic() > deadline:
                raise RuntimeError(
                    f"WidowX did not settle at {joints}: {self.bot.arm.get_joint_positions()}"
                )
            sleep(0.05)
        self.bot.arm.set_trajectory_time(
            moving_time=self.exp["moving_time"], accel_time=self.exp["accel_time"]
        )
        self.target = self._pose_from_matrix(self.bot.arm.get_ee_pose_command())

    def _at_joints(self, joints, tolerance=None):
        tolerance = self.exp["settle_tolerance"] if tolerance is None else tolerance
        return max(
            abs(measured - goal)
            for measured, goal in zip(self.bot.arm.get_joint_positions(), joints)
        ) <= tolerance

    def go_home(self):
        self._require_started()
        self._transition([0.0] * len(self.exp["READY_JOINTS"]))

    def go_sleep(self):
        """Through the ready pose to the sleep pose: where the arm rested at the start,
        or the nominal sleep joints."""
        self._require_started()
        self._transition(self.exp["READY_JOINTS"])
        self._transition(self.rest_joints or self.bot.arm.group_info.joint_sleep_positions)

    def _set_pwm(self, value):
        """Goal_PWM limits the motor output also in position mode. Written motor by
        motor, since a group write does not reach the shadow motors."""
        for name in self.exp["pwm_ramp_motors"]:
            self.bot.core.robot_set_motor_registers("single", name, "Goal_PWM", int(value))

    def _torque_off_softly(self):
        """Lower Goal_PWM from pwm_ramp_start to 0 over torque_off_ramp_time, so that
        the arm settles under gravity, then torque off, then Goal_PWM back to the
        maximum. Holding the arm needs only a few percent of pwm_max."""
        ramp = self.exp["torque_off_ramp_time"] > 0
        if ramp:
            steps = 20
            for step in range(steps + 1):
                self._set_pwm(self.exp["pwm_ramp_start"] * (1 - step / steps))
                sleep(self.exp["torque_off_ramp_time"] / steps)
        self.bot.core.robot_torque_enable("group", "all", False)
        if ramp:
            self._set_pwm(self.exp["pwm_max"])

    def stop_robot(self):
        self._require_started()
        try:
            shutdown_pose = self.exp.get("shutdown_pose", "sleep")
            if shutdown_pose == "home":
                self.go_home()
            elif shutdown_pose == "sleep":
                self.go_sleep()
                self._torque_off_softly()
            elif shutdown_pose != "hold":
                raise ValueError(f"Unsupported WidowX shutdown_pose: {shutdown_pose}")
        finally:
            self.started = False
            self.runtime.release(self)
