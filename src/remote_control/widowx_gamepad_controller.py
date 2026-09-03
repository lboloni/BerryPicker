"""Xbox-style gamepad mapping for native WidowX pose commands."""

from copy import copy
from math import isfinite

from robot.widowx import WidowXCommand, WidowXPose
from robot.widowx.move import move_pose_by_clamped


class WidowXGamepadController:
    """Convert gamepad input into bounded, IK-checked WidowX commands.

    The sticks control x/y/z/yaw. The trigger difference controls either roll
    or pitch, selected by the orientation-mode button.
    """

    ORIENTATION_MODES = ("roll", "pitch")
    BUTTON_FIELDS = (
        "button_exit",
        "button_home",
        "button_orientation_mode",
        "button_release",
        "button_grasp",
    )

    def __init__(self, exp, robot_controller):
        if robot_controller is None:
            raise ValueError("WidowXGamepadController requires a robot controller")
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
        self.orientation_mode = exp.get("initial_orientation_mode", "roll")
        if self.orientation_mode not in self.ORIENTATION_MODES:
            raise ValueError(
                "WidowX gamepad initial_orientation_mode must be roll or pitch"
            )

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
            raise ValueError(f"WidowX gamepad {name} must be positive and finite")
        return float(value)

    @classmethod
    def _read_velocity(cls, velocity):
        if not isinstance(velocity, dict) or set(velocity) != set(WidowXPose.FIELDS):
            raise ValueError("WidowX gamepad velocity must specify every pose field")
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
                raise ValueError(f"WidowX gamepad {field} must be a nonempty string")
            buttons[field] = value
        if len(set(buttons.values())) != len(buttons):
            raise ValueError("WidowX gamepad buttons must be distinct")
        return buttons

    @staticmethod
    def _axis(joystick, name):
        value = getattr(joystick, name)
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"WidowX gamepad axis {name} must be finite")
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"WidowX gamepad axis {name} must be in [-1, 1]")
        return float(value)

    def synchronize(self, actual_pose):
        """Capture the actual startup pose as the initial target and home pose."""
        if self.synchronized:
            raise RuntimeError("WidowX gamepad controller is already synchronized")
        if not isinstance(actual_pose, WidowXPose):
            raise TypeError("WidowX gamepad synchronization requires a WidowXPose")
        actual_pose.validate(self.robot_controller.exp)
        self.pos_target = copy(actual_pose)
        self.pos_home = copy(actual_pose)
        self.last_command = WidowXCommand(actual_pose)
        self.synchronized = True

    def poll_controller(self, joystick, dt):
        """Read one gamepad sample and return its native WidowX command."""
        if not self.synchronized:
            raise RuntimeError("WidowX gamepad controller is not synchronized")
        dt = self._positive_finite(dt, "timestep")
        dt = min(dt, self.max_input_dt)
        presses = joystick.check_presses()
        pressed = set(presses.names)

        if self.buttons["button_exit"] in pressed:
            self.exit_control = True
            return None
        if self.buttons["button_orientation_mode"] in pressed:
            index = self.ORIENTATION_MODES.index(self.orientation_mode)
            self.orientation_mode = self.ORIENTATION_MODES[1 - index]

        release = self.buttons["button_release"] in pressed
        grasp = self.buttons["button_grasp"] in pressed
        if release and grasp:
            raise ValueError("WidowX gamepad cannot grasp and release simultaneously")
        gripper_action = "release" if release else "grasp" if grasp else "hold"

        if self.buttons["button_home"] in pressed:
            candidate = copy(self.pos_home)
        else:
            trigger_axis = max(
                -1.0,
                min(
                    1.0,
                    self._axis(joystick, "lt") - self._axis(joystick, "rt"),
                ),
            )
            deltas = {
                "x": self._axis(joystick, "ly") * self.velocity["x"] * dt,
                "y": self._axis(joystick, "lx") * self.velocity["y"] * dt,
                "z": self._axis(joystick, "ry") * self.velocity["z"] * dt,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": self._axis(joystick, "rx") * self.velocity["yaw"] * dt,
            }
            deltas[self.orientation_mode] = (
                trigger_axis * self.velocity[self.orientation_mode] * dt
            )
            candidate = move_pose_by_clamped(
                self.robot_controller.exp, self.pos_target, deltas
            )

        self.last_target_rejected = False
        if candidate.as_dict() != self.pos_target.as_dict():
            if self.robot_controller.can_reach(candidate):
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
            "orientation_mode": self.orientation_mode,
            "last_target_rejected": self.last_target_rejected,
            "rejected_target_count": self.rejected_target_count,
        }
