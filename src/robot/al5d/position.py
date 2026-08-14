"""High-level AL5D position representation."""

from copy import copy
import logging

import numpy as np

from exp_run_config import Experiment

from .helper import RobotHelper

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RobotPosition:
    """A high-level AL5D position represented by its six control fields."""

    FIELDS = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]

    def __init__(self, exp: Experiment, values: dict = None):
        assert exp["robot_name"] == "al5d" and exp["controller_type"] == "position_controller"
        self.values = copy(exp["POS_DEFAULT"] if values is None else values)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __copy__(self):
        position = object.__new__(RobotPosition)
        position.values = copy(self.values)
        return position

    @staticmethod
    def _vector(values, normalized=False):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(RobotPosition.FIELDS),):
            raise ValueError(f"Expected {len(RobotPosition.FIELDS)} values, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Position values must be finite")
        if normalized and not np.all((0.0 <= values) & (values <= 1.0)):
            raise ValueError("Normalized position values must be in [0, 1]")
        return values

    @staticmethod
    def limit(exp: Experiment, posc):
        """Return whether ``posc`` is inside the configured AL5D position limits."""
        for field in RobotPosition.FIELDS:
            if posc.values[field] > exp["POS_MAX"][field]:
                logger.warning("RobotPosition.limit value %s too big %s", field, posc.values[field])
                return False
            if posc.values[field] < exp["POS_MIN"][field]:
                logger.warning("RobotPosition.limit value %s too small %s", field, posc.values[field])
                return False
        return True

    def to_normalized_vector(self, exp: Experiment):
        if not RobotPosition.limit(exp, self):
            raise ValueError(f"Unsafe robot position:\n{self}")
        return np.asarray([
            RobotHelper.map_ranges(self.values[field], exp["POS_MIN"][field], exp["POS_MAX"][field])
            for field in RobotPosition.FIELDS
        ], dtype=np.float32)

    @staticmethod
    def from_normalized_vector(exp: Experiment, values):
        values = RobotPosition._vector(values, normalized=True)
        rp = RobotPosition(exp)
        for i, field in enumerate(RobotPosition.FIELDS):
            rp.values[field] = RobotHelper.map_ranges(
                values[i], 0.0, 1.0, exp["POS_MIN"][field], exp["POS_MAX"][field])
        if not RobotPosition.limit(exp, rp):
            raise ValueError(f"Unsafe robot position:\n{rp}")
        return rp

    @staticmethod
    def from_vector(exp: Experiment, values):
        values = RobotPosition._vector(values)
        rp = RobotPosition(exp)
        for i, field in enumerate(RobotPosition.FIELDS):
            rp.values[field] = values[i]
        if not RobotPosition.limit(exp, rp):
            raise ValueError(f"Unsafe robot position:\n{rp}")
        return rp

    def empirical_distance(self, exp: Experiment, other):
        weights = np.ones(len(RobotPosition.FIELDS)) / len(RobotPosition.FIELDS)
        return np.inner(weights, np.abs(
            self.to_normalized_vector(exp) - other.to_normalized_vector(exp)))

    def __str__(self):
        return "Position: \n" + "".join(
            f" {field}:{self.values[field]:.2f}\n" for field in RobotPosition.FIELDS)
