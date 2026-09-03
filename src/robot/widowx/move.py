"""Pure motion helpers for WidowX pose targets."""

from math import isfinite, pi

from .position import WidowXPose


ANGULAR_FIELDS = {"roll", "pitch", "yaw"}


def _angular_delta(current, target):
    return (target - current + pi) % (2.0 * pi) - pi


def _wrap_angle(value):
    return (value + pi) % (2.0 * pi) - pi


def move_towards(current, target, max_step, angular=False):
    current, target, max_step = float(current), float(target), float(max_step)
    if not all(isfinite(value) for value in (current, target, max_step)):
        raise ValueError("WidowX movement values must be finite")
    if max_step < 0:
        raise ValueError("WidowX maximum movement step must not be negative")
    delta = _angular_delta(current, target) if angular else target - current
    if abs(delta) <= max_step:
        return target
    result = current + max_step if delta > 0 else current - max_step
    return _wrap_angle(result) if angular else result


def move_pose_by(exp, current, deltas):
    unknown = set(deltas) - set(WidowXPose.FIELDS)
    if unknown:
        raise ValueError(f"Unknown WidowX pose fields: {sorted(unknown)}")
    target = current.__copy__()
    for field, delta in deltas.items():
        if not isinstance(delta, (int, float)) or not isfinite(delta):
            raise ValueError("WidowX pose deltas must be finite")
        target[field] += delta
        if field in ANGULAR_FIELDS:
            target[field] = _wrap_angle(target[field])
    target.validate(exp)
    return target


def move_pose_by_clamped(exp, current, deltas):
    """Apply pose deltas while saturating at the configured task-space limits."""
    unknown = set(deltas) - set(WidowXPose.FIELDS)
    if unknown:
        raise ValueError(f"Unknown WidowX pose fields: {sorted(unknown)}")
    target = current.__copy__()
    for field, delta in deltas.items():
        if not isinstance(delta, (int, float)) or not isfinite(delta):
            raise ValueError("WidowX pose deltas must be finite")
        value = target[field] + delta
        if field in ANGULAR_FIELDS:
            value = _wrap_angle(value)
        target[field] = min(
            exp["POSE_MAX"][field], max(exp["POSE_MIN"][field], value)
        )
    target.validate(exp)
    return target


def move_pose_towards(exp, current, target, max_steps):
    if set(max_steps) != set(WidowXPose.FIELDS):
        raise ValueError("WidowX maximum steps must specify every pose field")
    values = {
        field: move_towards(
            current[field], target[field], max_steps[field], field in ANGULAR_FIELDS
        )
        for field in WidowXPose.FIELDS
    }
    return WidowXPose(exp, values)
