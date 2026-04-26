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
