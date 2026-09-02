"""Configured automated waypoint generation for AL5D demonstrations."""

import math
import random
import time

from exp_run_config import Config, Experiment
from robot.al5d import PositionController, RobotPosition
from robot.al5d.move import move_position_towards

from .abstract_controller import AbstractController


class AutoMoveController(AbstractController):
    """Generate and follow configured random RobotPosition or end-effector paths."""

    CARTESIAN_FIELDS = ("x", "y", "z")
    DERIVED_POSITION_FIELDS = {"height", "distance", "heading"}

    def __init__(self, exp: Experiment, robot_controller: PositionController = None,
                 camera_controller=None, demonstration_recorder=None):
        super().__init__(robot_controller, camera_controller, demonstration_recorder)
        if robot_controller is None:
            raise ValueError("AutoMoveController requires a robot controller")
        self.exp = exp
        self.robot_controller = robot_controller
        self.automove_type = exp["automove_type"]
        self.random_seed = Config().runtime.get(
            "automove_random_seed", exp["random_seed"])
        if type(self.random_seed) is not int:
            raise ValueError("AutoMove random_seed must be an integer")
        self.rng = random.Random(self.random_seed)
        self.max_timesteps = exp["max_timesteps"]
        self.interactive_confirm = exp["interactive_confirm"]
        self.controller_interval = exp["controller_interval"]
        self.robot_interval = exp["robot_interval"]
        self.waypoint_count = exp["waypoint_count"]
        self.waypoint_reached_distance = exp["waypoint_reached_distance"]
        self.max_sampling_attempts = exp["max_sampling_attempts"]
        self._validate_common_configuration()

        if self.automove_type == "random_robot_position":
            self._initialize_robot_position_configuration()
        elif self.automove_type == "random_ee_box":
            self._initialize_end_effector_configuration("end_effector_box")
        elif self.automove_type == "random_ee_plane":
            self._initialize_end_effector_configuration("end_effector_plane")
        else:
            raise ValueError(f"Unsupported automove_type: {self.automove_type}")

    def _validate_common_configuration(self):
        if self.max_timesteps <= 0:
            raise ValueError("AutoMove max_timesteps must be positive")
        if self.waypoint_count <= 0:
            raise ValueError("AutoMove waypoint_count must be positive")
        if self.max_sampling_attempts <= 0:
            raise ValueError("AutoMove max_sampling_attempts must be positive")
        for name, value in (
            ("controller_interval", self.controller_interval),
            ("robot_interval", self.robot_interval),
            ("waypoint_reached_distance", self.waypoint_reached_distance),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"AutoMove {name} must be a positive finite number")

    @staticmethod
    def _range(value, name):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a two-value list")
        low, high = value
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain finite numbers")
        if low > high:
            raise ValueError(f"{name} minimum must not exceed its maximum")
        return float(low), float(high)

    @staticmethod
    def _choices(value, name):
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a nonempty list")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain finite numbers")
        return tuple(float(item) for item in value)

    def _initialize_robot_position_configuration(self):
        fields = self.exp["robot_position"]
        if set(fields) != set(RobotPosition.FIELDS):
            raise ValueError("random_robot_position must configure every RobotPosition field")
        self.position_fields = {}
        for field in RobotPosition.FIELDS:
            definition = fields[field]
            if not isinstance(definition, dict) or set(definition) not in (
                    {"fixed"}, {"random"}, {"choices"}):
                raise ValueError(
                    f"robot_position.{field} must contain exactly one of "
                    "fixed, random, or choices"
                )
            if "fixed" in definition:
                value = definition["fixed"]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"robot_position.{field}.fixed must be finite")
                self.position_fields[field] = ("fixed", float(value))
            elif "random" in definition:
                self.position_fields[field] = (
                    "random", self._range(definition["random"], f"robot_position.{field}.random")
                )
            else:
                self.position_fields[field] = (
                    "choices", self._choices(
                        definition["choices"], f"robot_position.{field}.choices")
                )
        minimum = RobotPosition(self.robot_controller.exp)
        maximum = RobotPosition(self.robot_controller.exp)
        for field, (kind, value) in self.position_fields.items():
            if kind == "fixed":
                minimum[field] = value
                maximum[field] = value
            elif kind == "random":
                minimum[field], maximum[field] = value
            else:
                minimum[field], maximum[field] = min(value), max(value)
        self._validate_position(minimum)
        self._validate_position(maximum)
        motion = self.exp["motion"]
        if motion["type"] != "robot_position_velocity":
            raise ValueError("random_robot_position requires motion.type robot_position_velocity")
        velocity = motion["velocity"]
        if set(velocity) != set(RobotPosition.FIELDS):
            raise ValueError("robot_position_velocity must specify every RobotPosition velocity")
        self.velocity = {}
        for field in RobotPosition.FIELDS:
            value = velocity[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"motion.velocity.{field} must be a positive finite number")
            self.velocity[field] = float(value)

    def _initialize_end_effector_configuration(self, workspace_key):
        fixed_fields = self.exp["end_effector_fixed_robot_position"]
        expected_fields = set(RobotPosition.FIELDS) - self.DERIVED_POSITION_FIELDS
        if set(fixed_fields) != expected_fields:
            raise ValueError(
                "end_effector_fixed_robot_position must specify wrist_angle, "
                "wrist_rotation, and gripper"
            )
        self.fixed_fields = {}
        for field, value in fixed_fields.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"end_effector_fixed_robot_position.{field} must be finite")
            self.fixed_fields[field] = float(value)
        motion = self.exp["motion"]
        if motion["type"] != "end_effector_linear_velocity":
            raise ValueError(
                f"{self.automove_type} requires motion.type end_effector_linear_velocity"
            )
        self.linear_velocity = motion["linear_velocity"]
        if (
            not isinstance(self.linear_velocity, (int, float))
            or not math.isfinite(self.linear_velocity)
            or self.linear_velocity <= 0
        ):
            raise ValueError("motion.linear_velocity must be a positive finite number")
        if self.automove_type == "random_ee_box":
            box = self.exp[workspace_key]
            if set(box) != set(self.CARTESIAN_FIELDS):
                raise ValueError("end_effector_box must specify x, y, and z ranges")
            self.box_ranges = {
                axis: self._range(box[axis], f"end_effector_box.{axis}")
                for axis in self.CARTESIAN_FIELDS
            }
        else:
            self._initialize_plane(self.exp[workspace_key])
        self._validate_position(self.from_end_effector_xyz(self.to_end_effector_xyz(self.pos_current)))

    def _initialize_plane(self, plane):
        axis_aligned_keys = {"fixed", "ranges"}
        parametric_keys = {"origin", "u_direction", "v_direction", "u_range", "v_range"}
        if set(plane) == axis_aligned_keys:
            fixed = plane["fixed"]
            ranges = plane["ranges"]
            if not isinstance(fixed, dict) or len(fixed) != 1:
                raise ValueError("end_effector_plane.fixed must contain exactly one axis")
            fixed_axis, fixed_value = next(iter(fixed.items()))
            if fixed_axis not in self.CARTESIAN_FIELDS:
                raise ValueError("end_effector_plane.fixed axis must be x, y, or z")
            if not isinstance(fixed_value, (int, float)) or not math.isfinite(fixed_value):
                raise ValueError("end_effector_plane.fixed value must be finite")
            free_axes = set(self.CARTESIAN_FIELDS) - {fixed_axis}
            if set(ranges) != free_axes:
                raise ValueError("end_effector_plane.ranges must specify the two non-fixed axes")
            self.plane_type = "axis_aligned"
            self.plane_fixed_axis = fixed_axis
            self.plane_fixed_value = float(fixed_value)
            self.plane_ranges = {
                axis: self._range(ranges[axis], f"end_effector_plane.ranges.{axis}")
                for axis in free_axes
            }
            return
        if set(plane) != parametric_keys:
            raise ValueError("end_effector_plane must use axis-aligned or parametric form")
        self.plane_type = "parametric"
        self.plane_origin = self._vector(plane["origin"], "end_effector_plane.origin")
        self.plane_u = self._vector(plane["u_direction"], "end_effector_plane.u_direction")
        self.plane_v = self._vector(plane["v_direction"], "end_effector_plane.v_direction")
        if self._norm(self.plane_u) == 0 or self._norm(self.plane_v) == 0:
            raise ValueError("end_effector_plane directions must be nonzero")
        if self._norm(self._cross(self.plane_u, self.plane_v)) == 0:
            raise ValueError("end_effector_plane directions must not be collinear")
        self.plane_u_range = self._range(plane["u_range"], "end_effector_plane.u_range")
        self.plane_v_range = self._range(plane["v_range"], "end_effector_plane.v_range")

    def _vector(self, value, name):
        if not isinstance(value, dict) or set(value) != set(self.CARTESIAN_FIELDS):
            raise ValueError(f"{name} must specify x, y, and z")
        vector = tuple(float(value[axis]) for axis in self.CARTESIAN_FIELDS)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(f"{name} must contain finite values")
        return vector

    @staticmethod
    def _norm(vector):
        return math.sqrt(sum(item * item for item in vector))

    @staticmethod
    def _cross(left, right):
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def to_end_effector_xyz(self, position):
        """Map AL5D distance/heading/height to x/y/z in inches."""
        heading = math.radians(position["heading"])
        return {
            "x": position["distance"] * math.cos(heading),
            "y": position["distance"] * math.sin(heading),
            "z": position["height"],
        }

    def from_end_effector_xyz(self, point):
        """Map x/y/z in inches to an AL5D RobotPosition with fixed wrist fields."""
        if set(point) != set(self.CARTESIAN_FIELDS):
            raise ValueError("End-effector point must specify x, y, and z")
        x, y, z = (float(point[axis]) for axis in self.CARTESIAN_FIELDS)
        if not all(math.isfinite(item) for item in (x, y, z)):
            raise ValueError("End-effector point must contain finite values")
        position = RobotPosition(self.robot_controller.exp)
        position["height"] = z
        position["distance"] = math.hypot(x, y)
        position["heading"] = math.degrees(math.atan2(y, x))
        for field, value in self.fixed_fields.items():
            position[field] = value
        self._validate_position(position)
        return position

    def _validate_position(self, position):
        if not RobotPosition.limit(self.robot_controller.exp, position):
            raise ValueError(f"AutoMove target is outside AL5D limits:\n{position}")

    def _sample_robot_position(self):
        position = RobotPosition(self.robot_controller.exp)
        for field, (kind, value) in self.position_fields.items():
            if kind == "fixed":
                position[field] = value
            elif kind == "random":
                position[field] = self.rng.uniform(*value)
            else:
                position[field] = self.rng.choice(value)
        return position

    def _sample_end_effector_point(self):
        if self.automove_type == "random_ee_box":
            return {axis: self.rng.uniform(*self.box_ranges[axis]) for axis in self.CARTESIAN_FIELDS}
        if self.plane_type == "axis_aligned":
            point = {self.plane_fixed_axis: self.plane_fixed_value}
            point.update({
                axis: self.rng.uniform(*self.plane_ranges[axis])
                for axis in self.plane_ranges
            })
            return point
        u = self.rng.uniform(*self.plane_u_range)
        v = self.rng.uniform(*self.plane_v_range)
        return {
            axis: self.plane_origin[index] + u * self.plane_u[index] + v * self.plane_v[index]
            for index, axis in enumerate(self.CARTESIAN_FIELDS)
        }

    def _validate_current_end_effector_point(self):
        point = self.to_end_effector_xyz(self.pos_current)
        tolerance = self.waypoint_reached_distance
        if self.automove_type == "random_ee_box":
            for axis, (low, high) in self.box_ranges.items():
                if not low - tolerance <= point[axis] <= high + tolerance:
                    raise ValueError("Initial end-effector position is outside end_effector_box")
            return
        if self.plane_type == "axis_aligned":
            if abs(point[self.plane_fixed_axis] - self.plane_fixed_value) > tolerance:
                raise ValueError("Initial end-effector position is outside end_effector_plane")
            return
        relative = tuple(
            point[axis] - self.plane_origin[index]
            for index, axis in enumerate(self.CARTESIAN_FIELDS)
        )
        normal = self._cross(self.plane_u, self.plane_v)
        if abs(sum(left * right for left, right in zip(relative, normal))) > tolerance * self._norm(normal):
            raise ValueError("Initial end-effector position is outside end_effector_plane")

    def generate_waypoints(self):
        """Generate the configured, reproducible sequence of safe waypoints."""
        self.waypoints = []
        if self.automove_type in {"random_ee_box", "random_ee_plane"}:
            self._validate_current_end_effector_point()
        attempts = 0
        while len(self.waypoints) < self.waypoint_count:
            attempts += 1
            if attempts > self.max_sampling_attempts:
                raise RuntimeError("Unable to sample enough reachable AutoMove waypoints")
            if self.automove_type == "random_robot_position":
                waypoint = self._sample_robot_position()
                try:
                    self._validate_position(waypoint)
                except ValueError:
                    continue
            else:
                point = self._sample_end_effector_point()
                try:
                    self.from_end_effector_xyz(point)
                except ValueError:
                    continue
                waypoint = point
            self.waypoints.append(waypoint)

    def set_waypoints(self, waypoints):
        """Set waypoints externally for deterministic experiments."""
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError("AutoMove waypoints must be a nonempty list")
        self.waypoints = list(waypoints)

    def next_pos(self):
        """Return the next robot target, or None after all waypoints are reached."""
        if not self.waypoints:
            return None
        if self.automove_type == "random_robot_position":
            return self._next_robot_position()
        return self._next_end_effector_position()

    def _next_robot_position(self):
        waypoint = self.waypoints[0]
        if self.pos_current.empirical_distance(self.robot_controller.exp, waypoint) <= self.waypoint_reached_distance:
            del self.waypoints[0]
            if not self.waypoints:
                return None
            waypoint = self.waypoints[0]
        max_steps = {
            field: self.velocity[field] * self.robot_interval
            for field in RobotPosition.FIELDS
        }
        self.pos_target = move_position_towards(
            self.robot_controller.exp, self.pos_current, waypoint, max_steps
        )
        return self.pos_target

    def _next_end_effector_position(self):
        waypoint = self.waypoints[0]
        current = self.to_end_effector_xyz(self.pos_current)
        delta = {axis: waypoint[axis] - current[axis] for axis in self.CARTESIAN_FIELDS}
        distance = math.sqrt(sum(value * value for value in delta.values()))
        if distance <= self.waypoint_reached_distance:
            del self.waypoints[0]
            if not self.waypoints:
                return None
            return self._next_end_effector_position()
        step = min(self.linear_velocity * self.robot_interval, distance)
        point = {
            axis: current[axis] + delta[axis] * step / distance
            for axis in self.CARTESIAN_FIELDS
        }
        self.pos_target = self.from_end_effector_xyz(point)
        return self.pos_target

    def control(self):
        """Legacy interactive loop; collection uses AutoMoveLeaderParticipant instead."""
        self.generate_waypoints()
        self.autonomous_countdown = 0
        while self.max_timesteps > 0:
            start_time = time.monotonic()
            if self.camera_controller is not None:
                self.camera_controller.update()
            self.pos_current = self.robot_controller.get_position()
            self.max_timesteps -= 1
            target = self.next_pos()
            if target is None:
                break
            self.autonomous_countdown -= 1
            if self.interactive_confirm and self.autonomous_countdown <= 0:
                proceed = input(f"Proposed next target: {target}. Proceed? [stop/y/<number>]")
                if proceed == "stop":
                    break
                if proceed.isdigit():
                    self.autonomous_countdown = int(proceed)
            self.control_robot()
            self.update()
            time.sleep(max(0.0, self.controller_interval - (time.monotonic() - start_time)))
        self.stop()
