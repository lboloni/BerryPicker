"""The Franka Emika Panda arm. For now only its positions (as recorded in the DREAM dataset); a controller would go here, as for robot/wxai."""

from .position import FrankaPosition

__all__ = ["FrankaPosition"]
