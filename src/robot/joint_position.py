"""The base class of the positions of robots described by a vector of named joints.

A robot of this kind (robot/franka, robot/g1, robot/inspire_hand) subclasses
JointPosition and sets ROBOT_NAME, as WXAIPose subclasses WidowXPose. The
joints, their order and their limits come from the robot exp/run, not from
the code: JOINTS (ordered list), POS_MIN, POS_MAX and POS_DEFAULT (dicts
keyed by joint name), and enforce_limits.
"""

from copy import copy
import logging

import numpy as np

from exp_run_config import Experiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class JointPosition:
    """A position of a robot described by the joints of its exp/run."""

    # the robot_name of the experiments this position class accepts, set by the subclasses
    ROBOT_NAME = None

    def __init__(self, exp: Experiment, values: dict = None):
        assert exp["robot_name"] == self.ROBOT_NAME, f"{type(self).__name__} requires robot_name {self.ROBOT_NAME}"
        self.fields = list(exp["JOINTS"])
        values = exp["POS_DEFAULT"] if values is None else values
        # keep only the joints of the exp/run, in their order
        self.values = {field: values[field] for field in self.fields}

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __copy__(self):
        position = object.__new__(type(self))
        position.fields = copy(self.fields)
        position.values = copy(self.values)
        return position

    @staticmethod
    def limit(exp: Experiment, posc):
        """Return whether ``posc`` is inside the joint limits of the exp/run."""
        for field in posc.fields:
            if posc.values[field] > exp["POS_MAX"][field]:
                logger.warning("JointPosition.limit value %s too big %s", field, posc.values[field])
                return False
            if posc.values[field] < exp["POS_MIN"][field]:
                logger.warning("JointPosition.limit value %s too small %s", field, posc.values[field])
                return False
        return True

    def to_normalized_vector(self, exp: Experiment, enforce_limits=None):
        """The joints mapped from [POS_MIN, POS_MAX] to [0, 1], in the order of JOINTS.
        With enforce_limits false, values outside the limits map outside [0, 1] instead of
        raising: for reproductions of runs that did not check the limits (the exp/run sets
        enforce_limits), and for predicted positions (the caller passes enforce_limits=False).
        By default, the enforce_limits of the exp/run applies."""
        if enforce_limits is None:
            enforce_limits = exp["enforce_limits"]
        if enforce_limits and not JointPosition.limit(exp, self):
            raise ValueError(f"Joint position outside the limits:\n{self}")
        return np.asarray([
            (self.values[field] - exp["POS_MIN"][field]) / (exp["POS_MAX"][field] - exp["POS_MIN"][field])
            for field in self.fields
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, exp: Experiment, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(exp["JOINTS"]),):
            raise ValueError(f"Expected {len(exp['JOINTS'])} values, got {values.shape}")
        return cls(exp, dict(zip(exp["JOINTS"], values.tolist())))

    @classmethod
    def from_normalized_vector(cls, exp: Experiment, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(exp["JOINTS"]),):
            raise ValueError(f"Expected {len(exp['JOINTS'])} values, got {values.shape}")
        return cls(exp, {
            field: exp["POS_MIN"][field] + values[i] * (exp["POS_MAX"][field] - exp["POS_MIN"][field])
            for i, field in enumerate(exp["JOINTS"])
        })

    def empirical_distance(self, exp: Experiment, other):
        return float(np.mean(np.abs(
            self.to_normalized_vector(exp) - other.to_normalized_vector(exp))))

    def __str__(self):
        return f"{type(self).__name__}: \n" + "".join(
            f" {field}:{self.values[field]:.3f}\n" for field in self.fields)
