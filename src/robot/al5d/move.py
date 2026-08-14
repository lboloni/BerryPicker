"""Pure position-motion helpers for the fixed AL5D controller."""

from math import isfinite
from typing import Mapping

from .position import RobotPosition


def move_towards(current: float, target: float, max_step: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_step``."""
    current, target, max_step = float(current), float(target), float(max_step)
    if not all(isfinite(value) for value in (current, target, max_step)):
        raise ValueError("Movement values must be finite")
    if max_step < 0:
        raise ValueError("Maximum movement step must not be negative")
    if abs(target - current) <= max_step:
        return target
    return current + max_step if target > current else current - max_step


def move_position_by(exp, current: RobotPosition,
                     deltas: Mapping[str, float]) -> RobotPosition:
    """Return a copied, limit-checked position after applying ``deltas``."""
    unknown = set(deltas) - set(RobotPosition.FIELDS)
    if unknown:
        raise ValueError(f"Unknown robot position fields: {sorted(unknown)}")
    target = current.__copy__()
    for field, delta in deltas.items():
        delta = float(delta)
        if not isfinite(delta):
            raise ValueError("Position deltas must be finite")
        target[field] += delta
    if not RobotPosition.limit(exp, target):
        raise ValueError(f"Unsafe robot target:\n{target}")
    return target


def move_position_towards(exp, current: RobotPosition, target: RobotPosition,
                          max_steps: Mapping[str, float]) -> RobotPosition:
    """Return a limit-checked position moved toward ``target`` per field."""
    expected = set(RobotPosition.FIELDS)
    if set(max_steps) != expected:
        raise ValueError("Maximum steps must specify every robot position field")
    deltas = {
        field: move_towards(current[field], target[field], max_steps[field]) - current[field]
        for field in RobotPosition.FIELDS
    }
    return move_position_by(exp, current, deltas)
