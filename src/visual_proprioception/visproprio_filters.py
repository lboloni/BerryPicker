"""
visproprio_filters.py

Causal filters over the output of a visual-proprioception regressor.

The regressor estimates a normalized robot position from one frame at a time.
Consecutive frames of a demonstration are highly redundant, so a recursive
filter over the predicted positions recovers accuracy without changing the
encoder or the regressor. See DESIGN-TemporalVisualProprioception.md.

These filters operate on the six-element normalized position vector, after the
sensor processor and after the regressor. They therefore combine with any
sensor processing family without per-family code. Filtering learned latent
coordinates instead would require a dynamics model those coordinates do not
have.

Each filter is a stream: `reset()` starts a new sequence and `update()` advances
it by one observation. `dt` is elapsed observation time in seconds, not
inference time.
"""

import numpy as np


class PositionFilter:
    """A causal filter over normalized position vectors."""

    def reset(self):
        """Start a new observation sequence."""
        raise NotImplementedError

    def update(self, measurement, dt):
        """Advance by one observation and return the current estimate."""
        raise NotImplementedError


class NoFilter(PositionFilter):
    """Pass measurements through, so the unfiltered path is the same code path."""

    def reset(self):
        pass

    def update(self, measurement, dt):
        return np.array(measurement, dtype=float)


class EMAPositionFilter(PositionFilter):
    """Time-aware exponential moving average with a per-field time constant.

    ``tau`` is in seconds. The first observation of a sequence initializes the
    estimate directly.
    """

    def __init__(self, tau):
        self.tau = np.asarray(tau, dtype=float)
        self.reset()

    def reset(self):
        self.filtered = None

    def update(self, measurement, dt):
        measurement = np.asarray(measurement, dtype=float)
        if self.filtered is None:
            self.filtered = measurement.copy()
        else:
            alpha = 1.0 - np.exp(-dt / self.tau)
            self.filtered = alpha * measurement + (1.0 - alpha) * self.filtered
        return self.filtered.copy()


class KalmanPositionFilter(PositionFilter):
    """Six independent constant-velocity Kalman filters, one per field.

    Each field has state [position, velocity], transition [[1, dt], [0, 1]] and
    measurement [1, 0]. ``process_noise`` is the acceleration noise density and
    ``measurement_noise`` the variance of the regressor's error, both per field.

    The fields are filtered independently because no cross-field correlation of
    the regressor's error has been measured. A joint state over all six degrees
    of freedom is a later extension.
    """

    def __init__(self, process_noise, measurement_noise, initial_velocity_variance=1.0):
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.measurement_noise = np.asarray(measurement_noise, dtype=float)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None

    def _transition(self, dt):
        return np.array([[1.0, dt], [0.0, 1.0]])

    def _process_covariance(self, dt):
        """Continuous acceleration noise integrated over one interval."""
        base = np.array([
            [dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 2 / 2.0, dt],
        ])
        return self.process_noise[:, None, None] * base

    def _initialize(self, measurement):
        fields = measurement.size
        self.state = np.stack([measurement, np.zeros(fields)], axis=1)
        self.covariance = np.zeros((fields, 2, 2))
        self.covariance[:, 0, 0] = self.measurement_noise
        self.covariance[:, 1, 1] = self.initial_velocity_variance

    def update(self, measurement, dt):
        measurement = np.asarray(measurement, dtype=float)
        if self.state is None:
            self._initialize(measurement)
            return self.state[:, 0].copy()

        transition = self._transition(dt)
        self.state = self.state @ transition.T
        self.covariance = (
            transition @ self.covariance @ transition.T + self._process_covariance(dt)
        )

        innovation = measurement - self.state[:, 0]
        innovation_variance = self.covariance[:, 0, 0] + self.measurement_noise
        gain = self.covariance[:, :, 0] / innovation_variance[:, None]
        self.state = self.state + gain * innovation[:, None]
        # (I - K H) P with H = [1, 0] reduces to P - outer(K, P[0, :]).
        self.covariance = (
            self.covariance - gain[:, :, None] * self.covariance[:, 0, :][:, None, :]
        )
        return self.state[:, 0].copy()


def _create_none(exp):
    return NoFilter()


def _create_ema(exp):
    return EMAPositionFilter(exp["filter_tau"])


def _create_kalman(exp):
    return KalmanPositionFilter(
        exp["filter_process_noise"], exp["filter_measurement_noise"]
    )


_FILTER_CLASSES = {
    "none": _create_none,
    "ema": _create_ema,
    "kalman": _create_kalman,
}


def create_position_filter(exp):
    """Get the position filter specified by a visual_proprioception exp/run."""
    name = exp.get("position_filter", "none")
    try:
        create = _FILTER_CLASSES[name]
    except KeyError as error:
        available = ", ".join(_FILTER_CLASSES)
        raise ValueError(
            f"Unknown position filter: {name!r}. Available filters: {available}"
        ) from error
    return create(exp)


def estimate_noise(predictions, targets, lengths, dt):
    """Starting Kalman noise terms measured from a training set.

    Measurement noise is the variance of the regressor's residual. Process
    noise is the acceleration noise density of the target trajectory, taken
    within each sequence so that the jump between two demonstrations does not
    register as acceleration.

    Returns:
        (process_noise, measurement_noise), both per field
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    measurement_noise = np.var(predictions - targets, axis=0)
    changes = []
    index = 0
    for length in lengths:
        segment = targets[index:index + length]
        velocity = np.diff(segment, axis=0) / dt
        changes.append(np.diff(velocity, axis=0))
        index += length
    process_noise = np.vstack(changes).var(axis=0) / dt
    return process_noise, measurement_noise


def _sweep(build, candidates, predictions, targets, lengths, dt):
    """Per-field RMSE for each candidate parameter value."""
    errors = np.empty((len(candidates), predictions.shape[1]))
    for row, candidate in enumerate(candidates):
        filtered = filter_sequence(build(candidate), predictions, lengths, dt)
        errors[row] = np.sqrt(np.mean((filtered - targets) ** 2, axis=0))
    return errors


def _best(candidates, errors):
    fields = errors.shape[1]
    best = errors.argmin(axis=0)
    return {
        "parameters": np.asarray(candidates)[best],
        "rmse": errors[best, np.arange(fields)],
        "candidates": np.asarray(candidates),
        "errors": errors,
    }


def tune_ema(predictions, targets, lengths, dt, candidates=None):
    """Per-field EMA time constant minimizing RMSE on the supplied data.

    The fields are filtered independently, so sweeping one shared time constant
    and taking the per-field minimum gives each field its own optimum.
    """
    fields = predictions.shape[1]
    if candidates is None:
        candidates = np.geomspace(0.02, 20.0, 60)
    errors = _sweep(
        lambda tau: EMAPositionFilter(np.full(fields, tau)),
        candidates, predictions, targets, lengths, dt)
    return _best(candidates, errors)


def tune_kalman(predictions, targets, lengths, dt,
                process_noise, measurement_noise, candidates=None):
    """Per-field multiplier on the measurement noise minimizing RMSE.

    The estimator is sensitive to the ratio of the two noise terms rather than
    their absolute scale, so the sweep moves the measurement noise and leaves
    the process noise at its measured value.
    """
    if candidates is None:
        candidates = np.geomspace(0.01, 1000.0, 60)
    errors = _sweep(
        lambda scale: KalmanPositionFilter(
            process_noise, np.asarray(measurement_noise, dtype=float) * scale),
        candidates, predictions, targets, lengths, dt)
    return _best(candidates, errors)


def filter_sequence(position_filter, predictions, lengths, dt):
    """Filter stacked predictions, resetting at every sequence boundary.

    ``predictions`` holds the frames of several demonstrations concatenated in
    load order and ``lengths`` how many frames each contributed. Filtering
    across a boundary would treat the jump between two demonstrations as robot
    motion.
    """
    predictions = np.asarray(predictions, dtype=float)
    assert sum(lengths) == len(predictions), (
        f"sequence lengths sum to {sum(lengths)} but there are "
        f"{len(predictions)} predictions"
    )
    filtered = np.empty_like(predictions)
    index = 0
    for length in lengths:
        position_filter.reset()
        for _ in range(length):
            filtered[index] = position_filter.update(predictions[index], dt)
            index += 1
    return filtered
