"""Automated native-pose waypoint generation for WidowX demonstrations."""

from copy import copy
import math
import random

from exp_run_config import Config
from robot.widowx import WidowXCommand, WidowXPose
from robot.widowx.move import move_pose_towards


class WidowXAutoMoveController:
    """Generate reachable WidowX poses and approach them at configured velocities."""

    def __init__(self, exp, robot_controller):
        if robot_controller is None:
            raise ValueError("WidowXAutoMoveController requires a robot controller")
        if exp["automove_type"] != "random_widowx_pose":
            raise ValueError(f"Unsupported WidowX automove_type: {exp['automove_type']}")
        self.exp = exp
        self.robot_controller = robot_controller
        self.automove_type = exp["automove_type"]
        runtime = getattr(Config._instance, "runtime", {})
        self.random_seed = runtime.get("automove_random_seed", exp["random_seed"])
        if type(self.random_seed) is not int:
            raise ValueError("WidowX AutoMove random_seed must be an integer")
        self.rng = random.Random(self.random_seed)
        self.max_timesteps = exp["max_timesteps"]
        self.waypoint_count = exp["waypoint_count"]
        self.max_sampling_attempts = exp["max_sampling_attempts"]
        self.waypoint_reached_distance = exp["waypoint_reached_distance"]
        self.interactive_confirm = exp.get("interactive_confirm", False)
        self.controller_interval = exp["controller_interval"]
        self.robot_interval = exp["robot_interval"]
        self._validate_common_configuration()
        self.pose_fields = self._read_pose_fields(exp["pose"])
        self.velocity = self._read_velocity(exp["motion"])
        self.gripper_actions = self._read_gripper_actions(
            exp.get("gripper_actions", ["hold"])
        )
        self.gripper_pressure = exp.get("gripper_pressure")
        if self.gripper_pressure is not None:
            WidowXCommand(WidowXPose(robot_controller.exp), gripper_pressure=self.gripper_pressure)
        self.waypoints = []
        self.pos_target = copy(robot_controller.get_target())
        self._gripper_sent = False

    def _validate_common_configuration(self):
        for name in ("max_timesteps", "waypoint_count", "max_sampling_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"WidowX AutoMove {name} must be a positive integer")
        for name in (
            "waypoint_reached_distance",
            "controller_interval",
            "robot_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"WidowX AutoMove {name} must be positive and finite")
        if type(self.interactive_confirm) is not bool:
            raise ValueError("WidowX AutoMove interactive_confirm must be boolean")

    @staticmethod
    def _range(value, name):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a two-value list")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain finite numbers")
        low, high = map(float, value)
        if low > high:
            raise ValueError(f"{name} minimum must not exceed its maximum")
        return low, high

    @staticmethod
    def _choices(value, name):
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a nonempty list")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain finite numbers")
        return tuple(map(float, value))

    def _read_pose_fields(self, fields):
        if set(fields) != set(WidowXPose.FIELDS):
            raise ValueError("WidowX AutoMove pose must configure every WidowX pose field")
        result = {}
        minimum = {}
        maximum = {}
        for field in WidowXPose.FIELDS:
            definition = fields[field]
            if not isinstance(definition, dict) or set(definition) not in (
                {"fixed"}, {"random"}, {"choices"}
            ):
                raise ValueError(
                    f"pose.{field} must contain exactly one of fixed, random, or choices"
                )
            if "fixed" in definition:
                value = definition["fixed"]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"pose.{field}.fixed must be finite")
                value = float(value)
                result[field] = ("fixed", value)
                minimum[field] = maximum[field] = value
            elif "random" in definition:
                limits = self._range(definition["random"], f"pose.{field}.random")
                result[field] = ("random", limits)
                minimum[field], maximum[field] = limits
            else:
                choices = self._choices(definition["choices"], f"pose.{field}.choices")
                result[field] = ("choices", choices)
                minimum[field], maximum[field] = min(choices), max(choices)
        WidowXPose(self.robot_controller.exp, minimum)
        WidowXPose(self.robot_controller.exp, maximum)
        return result

    @staticmethod
    def _read_velocity(motion):
        if motion.get("type") != "widowx_pose_velocity":
            raise ValueError("WidowX AutoMove requires motion.type widowx_pose_velocity")
        velocity = motion.get("velocity")
        if not isinstance(velocity, dict) or set(velocity) != set(WidowXPose.FIELDS):
            raise ValueError("WidowX pose velocity must specify every pose field")
        result = {}
        for field, value in velocity.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"motion.velocity.{field} must be positive and finite")
            result[field] = float(value)
        return result

    @staticmethod
    def _read_gripper_actions(actions):
        if not isinstance(actions, list) or not actions:
            raise ValueError("WidowX AutoMove gripper_actions must be a nonempty list")
        if any(action not in WidowXCommand.GRIPPER_ACTIONS for action in actions):
            raise ValueError("WidowX AutoMove contains an unsupported gripper action")
        return tuple(actions)

    def _sample_pose(self):
        values = {}
        for field, (kind, definition) in self.pose_fields.items():
            if kind == "fixed":
                values[field] = definition
            elif kind == "random":
                values[field] = self.rng.uniform(*definition)
            else:
                values[field] = self.rng.choice(definition)
        return WidowXPose(self.robot_controller.exp, values)

    def generate_waypoints(self):
        """Populate the route with IK-checked waypoint commands."""
        self.waypoints = []
        attempts = 0
        while len(self.waypoints) < self.waypoint_count:
            if attempts >= self.max_sampling_attempts:
                raise RuntimeError(
                    f"Could not sample {self.waypoint_count} reachable WidowX waypoints "
                    f"in {self.max_sampling_attempts} attempts"
                )
            attempts += 1
            pose = self._sample_pose()
            if self.robot_controller.can_reach(pose):
                self.waypoints.append(WidowXCommand(
                    pose,
                    gripper_action=self.rng.choice(self.gripper_actions),
                    gripper_pressure=self.gripper_pressure,
                ))
        self._gripper_sent = False

    def next_command(self, dt):
        """Return the next velocity-limited command, or ``None`` when complete."""
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("WidowX AutoMove timestep must be positive and finite")
        current = self.robot_controller.get_position()
        while self.waypoints:
            waypoint = self.waypoints[0]
            if current.empirical_distance(self.robot_controller.exp, waypoint.pose) \
                    <= self.waypoint_reached_distance:
                if waypoint.gripper_action != "hold" and not self._gripper_sent:
                    self._gripper_sent = True
                    self.pos_target = copy(current)
                    return WidowXCommand(
                        current, waypoint.gripper_action, waypoint.gripper_pressure
                    )
                self.waypoints.pop(0)
                self._gripper_sent = False
                continue
            max_steps = {
                field: self.velocity[field] * dt for field in WidowXPose.FIELDS
            }
            target = move_pose_towards(
                self.robot_controller.exp, current, waypoint.pose, max_steps
            )
            if not self.robot_controller.can_reach(target):
                raise RuntimeError(f"WidowX AutoMove intermediate pose is unreachable:\n{target}")
            self.pos_target = copy(target)
            return WidowXCommand(target)
        return None
