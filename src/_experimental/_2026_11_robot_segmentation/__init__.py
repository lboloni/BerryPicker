"""Standalone real-time robot segmentation experiment (2026-11).

All code lives inside ``src/_experimental/_2026_11_robot_segmentation`` with no
imports from BerryPicker core packages or ExpRunFlow.
"""

from .data import BG, FG, UNKNOWN
from .models import RobotSegmenter, load_segmenter

__all__ = ["BG", "FG", "UNKNOWN", "RobotSegmenter", "load_segmenter"]
