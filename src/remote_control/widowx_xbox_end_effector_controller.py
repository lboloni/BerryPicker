"""XBox input mapped to Cartesian WidowX end-effector commands."""

from copy import copy
from math import isfinite
import logging

from robot.widowx import WidowXCommand, WidowXPose
from robot.widowx.move import move_pose_by_clamped

logger = logging.getLogger(__name__)


class WidowXXboxEndEffectorController:
    """Convert XBox input into bounded, IK-checked end-effector commands.

    The sticks control x/y/z/yaw, the trigger difference controls roll, and
    the vertical D-pad axis controls pitch. All rotations are nonmodal.
    """

    BUTTON_FIELDS = (
        "button_exit",
        "button_home",
        "button_release",
        "button_grasp",
    )

    def __init__(self, exp, robot_controller):
        if robot_controller is None:
            raise ValueError(
                "WidowXXboxEndEffectorController requires a robot controller"
            )
        self.exp = exp
        self.robot_controller = robot_controller
        self.velocity = self._read_velocity(exp["velocity"])
        self.max_input_dt = self._positive_finite(
            exp.get("max_input_dt", 0.25), "max_input_dt"
        )
        self.gripper_pressure = exp.get("gripper_pressure")
        if self.gripper_pressure is not None:
            WidowXCommand(
                WidowXPose(robot_controller.exp),
                gripper_pressure=self.gripper_pressure,
            )
        self.buttons = self._read_buttons(exp)

        self.pos_target = copy(robot_controller.get_target())
        self.pos_home = None
        self.last_command = WidowXCommand(self.pos_target)
        self.exit_control = False
        self.synchronized = False
        self.last_target_rejected = False
        self.rejected_target_count = 0

    @staticmethod
    def _positive_finite(value, name):
        if (
            not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"WidowX XBox end-effector {name} must be positive and finite"
            )
        return float(value)

    @classmethod
    def _read_velocity(cls, velocity):
        if not isinstance(velocity, dict) or set(velocity) != set(WidowXPose.FIELDS):
            raise ValueError(
                "WidowX XBox end-effector velocity must specify every pose field"
            )
        return {
            field: cls._positive_finite(value, f"velocity.{field}")
            for field, value in velocity.items()
        }

    @classmethod
    def _read_buttons(cls, exp):
        buttons = {}
        for field in cls.BUTTON_FIELDS:
            value = exp[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"WidowX XBox end-effector {field} must be a nonempty string"
                )
            buttons[field] = value
        if len(set(buttons.values())) != len(buttons):
            raise ValueError("WidowX XBox end-effector buttons must be distinct")
        return buttons

    @staticmethod
    def _axis(joystick, name):
        value = getattr(joystick, name)
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"WidowX XBox end-effector axis {name} must be finite")
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"WidowX XBox end-effector axis {name} must be in [-1, 1]"
            )
        return float(value)

    def synchronize(self, actual_pose):
        """Capture the actual startup pose as the initial target and home pose."""
        if self.synchronized:
            raise RuntimeError(
                "WidowX XBox end-effector controller is already synchronized"
            )
        if not isinstance(actual_pose, WidowXPose):
            raise TypeError(
                "WidowX XBox end-effector synchronization requires a WidowXPose"
            )
        actual_pose.validate(self.robot_controller.exp)
        self.pos_target = copy(actual_pose)
        self.pos_home = copy(actual_pose)
        self.last_command = WidowXCommand(actual_pose)
        self.synchronized = True

    def poll_controller(self, joystick, dt):
        """Read one gamepad sample and return its native WidowX command."""
        if not self.synchronized:
            raise RuntimeError(
                "WidowX XBox end-effector controller is not synchronized"
            )
        dt = self._positive_finite(dt, "timestep")
        elapsed_dt = dt
        moving_time = self._positive_finite(
            self.robot_controller.exp["moving_time"], "moving_time"
        )
        dt = min(dt, self.max_input_dt, moving_time)
        presses = joystick.check_presses()
        pressed = set(presses.names)

        if self.buttons["button_exit"] in pressed:
            self.exit_control = True
            return None

        release = self.buttons["button_release"] in pressed
        grasp = self.buttons["button_grasp"] in pressed
        release_held = joystick[self.buttons["button_release"]] is not None
        grasp_held = joystick[self.buttons["button_grasp"]] is not None
        logger.info(
            "Gamepad poll: elapsed_dt=%.4f input_dt=%.4f moving_time=%.4f "
            "presses=%s release_held=%s grasp_held=%s",
            elapsed_dt, dt, moving_time, sorted(pressed), release_held, grasp_held,
        )
        if release_held and grasp_held:
            raise ValueError(
                "WidowX XBox end-effector controller cannot grasp and "
                "release simultaneously"
            )
        if release and grasp:
            # Press history has no ordering. Prefer the button still held; if
            # both have been released, retain the gripper state until a fresh press.
            release, grasp = release_held, grasp_held
            logger.warning(
                "Both gripper presses accumulated between polls; "
                "using held state: release=%s grasp=%s", release, grasp,
            )
        gripper_action = "release" if release else "grasp" if grasp else "hold"

        if self.buttons["button_home"] in pressed:
            candidate = copy(self.pos_home)
        else:
            axes = {
                name: self._axis(joystick, name)
                for name in ("lx", "ly", "rx", "ry", "lt", "rt", "dy")
            }
            logger.info("XBox end-effector axes=%s", axes)
            trigger_axis = max(
                -1.0,
                min(
                    1.0,
                    axes["lt"] - axes["rt"],
                ),
            )
            deltas = {
                "x": axes["ly"] * self.velocity["x"] * dt,
                "y": axes["lx"] * self.velocity["y"] * dt,
                "z": axes["ry"] * self.velocity["z"] * dt,
                "roll": trigger_axis * self.velocity["roll"] * dt,
                "pitch": -axes["dy"] * self.velocity["pitch"] * dt,
                "yaw": axes["rx"] * self.velocity["yaw"] * dt,
            }
            candidate = move_pose_by_clamped(
                self.robot_controller.exp, self.pos_target, deltas
            )

        self.last_target_rejected = False
        if candidate.as_dict() != self.pos_target.as_dict():
            logger.info(
                "Intended WidowX movement before IK check (meters/radians): from=%s to=%s",
                self.pos_target.as_dict(),
                candidate.as_dict(),
            )
            reachable = self.robot_controller.can_reach(candidate)
            logger.info(
                "Gamepad IK accepted=%s gripper_action=%s",
                reachable,
                gripper_action,
            )
            if reachable:
                self.pos_target = candidate
            else:
                self.last_target_rejected = True
                self.rejected_target_count += 1

        pressure = self.gripper_pressure if gripper_action != "hold" else None
        self.last_command = WidowXCommand(
            self.pos_target, gripper_action, pressure
        )
        return copy(self.last_command)

    def get_state(self):
        return {
            "target": self.pos_target.as_dict(),
            "last_target_rejected": self.last_target_rejected,
            "rejected_target_count": self.rejected_target_count,
        }
