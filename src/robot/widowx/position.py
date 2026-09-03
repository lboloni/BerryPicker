"""High-level WidowX pose and command representations."""

from copy import copy
from math import isfinite, pi

import numpy as np


class WidowXPose:
    """An end-effector pose in meters and radians."""

    FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(self, exp, values=None):
        if exp["robot_name"] != "widowx" or exp["controller_type"] != "position_controller":
            raise ValueError("WidowXPose requires a WidowX position-controller experiment")
        self.values = copy(exp["POSE_DEFAULT"] if values is None else values)
        self.validate(exp)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        if key not in self.FIELDS:
            raise KeyError(f"Unknown WidowX pose field: {key}")
        self.values[key] = value

    def __copy__(self):
        pose = object.__new__(WidowXPose)
        pose.values = copy(self.values)
        return pose

    def validate(self, exp):
        if set(self.values) != set(self.FIELDS):
            raise ValueError(f"WidowX pose must contain exactly {list(self.FIELDS)}")
        for field in self.FIELDS:
            value = self.values[field]
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"WidowX pose field {field} must be finite")
            if not exp["POSE_MIN"][field] <= value <= exp["POSE_MAX"][field]:
                raise ValueError(
                    f"WidowX pose field {field}={value} is outside "
                    f"[{exp['POSE_MIN'][field]}, {exp['POSE_MAX'][field]}]"
                )

    @staticmethod
    def limit(exp, pose):
        """Return whether a pose passes the configured task-space limits."""
        try:
            pose.validate(exp)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def to_vector(self):
        return np.asarray([self.values[field] for field in self.FIELDS], dtype=np.float32)

    def to_normalized_vector(self, exp):
        self.validate(exp)
        for field in self.FIELDS:
            if exp["POSE_MAX"][field] <= exp["POSE_MIN"][field]:
                raise ValueError(f"WidowX pose range for {field} must be positive")
        return np.asarray([
            (self.values[field] - exp["POSE_MIN"][field])
            / (exp["POSE_MAX"][field] - exp["POSE_MIN"][field])
            for field in self.FIELDS
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, exp, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(cls.FIELDS),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {len(cls.FIELDS)} finite WidowX pose values")
        return cls(exp, dict(zip(cls.FIELDS, values.tolist())))

    @classmethod
    def from_normalized_vector(cls, exp, values):
        values = np.asarray(values, dtype=float)
        if (
            values.shape != (len(cls.FIELDS),)
            or not np.all(np.isfinite(values))
            or not np.all((0.0 <= values) & (values <= 1.0))
        ):
            raise ValueError("Expected six normalized WidowX pose values in [0, 1]")
        absolute = {
            field: exp["POSE_MIN"][field]
            + values[index] * (exp["POSE_MAX"][field] - exp["POSE_MIN"][field])
            for index, field in enumerate(cls.FIELDS)
        }
        return cls(exp, absolute)

    def empirical_distance(self, exp, other):
        self.validate(exp)
        other.validate(exp)
        deltas = []
        for field in self.FIELDS:
            span = exp["POSE_MAX"][field] - exp["POSE_MIN"][field]
            if span <= 0:
                raise ValueError(f"WidowX pose range for {field} must be positive")
            delta = abs(self[field] - other[field])
            if field in ("roll", "pitch", "yaw"):
                delta = abs((self[field] - other[field] + pi) % (2.0 * pi) - pi)
            deltas.append(delta / span)
        return float(np.mean(deltas))

    def as_dict(self):
        return copy(self.values)

    def __str__(self):
        return "WidowX pose:\n" + "".join(
            f" {field}: {self.values[field]:.4f}\n" for field in self.FIELDS
        )


class WidowXCommand:
    """A WidowX pose target and a native Interbotix gripper action."""

    GRIPPER_ACTIONS = ("hold", "grasp", "release")

    def __init__(self, pose, gripper_action="hold", gripper_pressure=None):
        if not isinstance(pose, WidowXPose):
            raise TypeError("WidowXCommand pose must be a WidowXPose")
        if gripper_action not in self.GRIPPER_ACTIONS:
            raise ValueError(f"Unsupported WidowX gripper action: {gripper_action}")
        if gripper_pressure is not None:
            if not isinstance(gripper_pressure, (int, float)) or not isfinite(gripper_pressure):
                raise ValueError("WidowX gripper pressure must be finite")
            if not 0.0 <= gripper_pressure <= 1.0:
                raise ValueError("WidowX gripper pressure must be in [0, 1]")
        self.pose = copy(pose)
        self.gripper_action = gripper_action
        self.gripper_pressure = gripper_pressure

    def __copy__(self):
        return WidowXCommand(self.pose, self.gripper_action, self.gripper_pressure)

    def as_dict(self):
        return {
            "pose": self.pose.as_dict(),
            "gripper_action": self.gripper_action,
            "gripper_pressure": self.gripper_pressure,
        }
