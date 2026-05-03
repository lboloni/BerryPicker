"""
mobile_camera_decision.py

Decision policies for choosing mobile camera observation dome points.
"""

from abc import ABC, abstractmethod


class AbstractMobileCameraDecision(ABC):
    """Choose an observation dome point for a given robot position."""

    @abstractmethod
    def get_lat_long_for_robot_position(self, robot_position):
        """Return a `(lat_index, long_index)` point on the observation dome."""
        raise NotImplementedError


class SimpleHighLowMobileCameraDecision(AbstractMobileCameraDecision):
    """
    Choose a high or middle observation dome point based on robot height.

    If the robot height is above `height_threshold`, the policy chooses the dome
    top if reachable, otherwise the highest reachable latitude/longitude point.
    Lower robot positions use a configurable middle dome point.
    """

    def __init__(
        self,
        dome_driver,
        height_threshold=0.5,
        middle_lat_index=None,
        middle_long_index=0,
        reachability_kwargs=None,
    ):
        self.dome_driver = dome_driver
        self.height_threshold = height_threshold
        self.middle_lat_index = middle_lat_index
        self.middle_long_index = middle_long_index
        self.reachability_kwargs = reachability_kwargs or {}

    def get_lat_long_for_robot_position(self, robot_position):
        """Return a `(lat_index, long_index)` point on the observation dome."""
        robot_height = self._get_robot_height(robot_position)
        if robot_height > self.height_threshold:
            return self._get_highest_accessible_lat_long()
        return self._get_middle_lat_long()

    def _get_robot_height(self, robot_position):
        if robot_position is None:
            return 0.0
        try:
            print(f"Height threshold is {self.height_threshold}, robot height is {robot_position['height']}")
            return robot_position["height"]
        except (KeyError, TypeError):
            return robot_position.height

    def _get_highest_accessible_lat_long(self):
        dome = self.dome_driver.dome
        for lat_index in range(dome.no_lat + 1):
            long_indices = [0] if lat_index == 0 else range(dome.no_long)
            for long_index in long_indices:
                if self._is_lat_long_reachable(lat_index, long_index):
                    return lat_index, long_index
        return 0, 0

    def _get_middle_lat_long(self):
        dome = self.dome_driver.dome
        lat_index = self.middle_lat_index
        if lat_index is None:
            lat_index = dome.no_lat // 2
        return lat_index, self.middle_long_index

    def _is_lat_long_reachable(self, lat_index, long_index):
        try:
            return self.dome_driver.is_lat_long_reachable(
                lat_index,
                long_index,
                **self.reachability_kwargs,
            )
        except Exception:
            return False
