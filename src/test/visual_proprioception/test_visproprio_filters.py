import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from visual_proprioception.visproprio_filters import (
    EMAPositionFilter,
    KalmanPositionFilter,
    NoFilter,
    create_position_filter,
    estimate_noise,
    filter_sequence,
    tune_ema,
    tune_kalman,
)


def run(position_filter, measurements, dt=0.1):
    """Filter a sequence from a fresh context and stack the estimates."""
    position_filter.reset()
    return np.array([position_filter.update(m, dt) for m in measurements])


class TestNoFilter(unittest.TestCase):
    def test_reproduces_its_input_exactly(self):
        measurements = np.linspace(0.0, 1.0, 30).reshape(5, 6)

        filtered = run(NoFilter(), measurements)

        np.testing.assert_array_equal(filtered, measurements)


class TestEMAPositionFilter(unittest.TestCase):
    def test_first_observation_initializes_the_estimate(self):
        first = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        filtered = run(EMAPositionFilter(np.full(6, 0.5)), [first])

        np.testing.assert_allclose(filtered[0], first)

    def test_second_estimate_uses_the_time_aware_alpha(self):
        tau = np.full(6, 0.5)
        dt = 0.1
        first = np.zeros(6)
        second = np.ones(6)

        filtered = run(EMAPositionFilter(tau), [first, second], dt=dt)

        alpha = 1.0 - np.exp(-dt / tau)
        np.testing.assert_allclose(filtered[1], alpha * second)

    def test_constant_input_converges_to_that_constant(self):
        constant = np.full(6, 0.42)

        filtered = run(EMAPositionFilter(np.full(6, 0.3)), [constant] * 200)

        np.testing.assert_allclose(filtered[-1], constant, atol=1e-9)

    def test_small_time_constant_approaches_passthrough(self):
        measurements = np.random.default_rng(0).random((20, 6))

        filtered = run(EMAPositionFilter(np.full(6, 1e-6)), measurements)

        np.testing.assert_allclose(filtered, measurements, atol=1e-6)

    def test_per_field_time_constants_smooth_independently(self):
        tau = np.array([1e-6, 1e-6, 1e-6, 10.0, 10.0, 10.0])
        measurements = np.tile(np.arange(2.0).reshape(2, 1), (1, 6))

        filtered = run(EMAPositionFilter(tau), measurements)

        # The fast fields track the step; the slow fields barely move.
        self.assertGreater(filtered[1, 0], 0.99)
        self.assertLess(filtered[1, 3], 0.02)

    def test_steady_state_variance_matches_the_analytic_factor(self):
        dt, tau = 0.1, 0.4
        noise = np.random.default_rng(1).normal(0.0, 1.0, (40000, 1))

        filtered = run(EMAPositionFilter(np.full(1, tau)), noise, dt=dt)

        alpha = 1.0 - np.exp(-dt / tau)
        expected = np.sqrt(alpha / (2.0 - alpha))
        self.assertAlmostEqual(filtered[1000:].std(), expected, delta=0.02)


class TestKalmanPositionFilter(unittest.TestCase):
    def make_filter(self, process_noise=1e-4, measurement_noise=0.03, fields=6):
        return KalmanPositionFilter(
            np.full(fields, process_noise), np.full(fields, measurement_noise)
        )

    def test_first_observation_initializes_the_estimate(self):
        first = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        filtered = run(self.make_filter(), [first])

        np.testing.assert_allclose(filtered[0], first)

    def test_covariance_stays_symmetric_and_positive_definite(self):
        position_filter = self.make_filter()
        measurements = np.random.default_rng(2).random((50, 6))

        run(position_filter, measurements)

        np.testing.assert_allclose(
            position_filter.covariance,
            position_filter.covariance.transpose(0, 2, 1),
        )
        self.assertTrue((np.linalg.eigvalsh(position_filter.covariance) > 0).all())

    def test_position_variance_decreases_on_a_static_target(self):
        position_filter = KalmanPositionFilter(np.zeros(1), np.full(1, 0.04))
        constant = np.full(1, 0.5)

        run(position_filter, [constant] * 40)

        self.assertLess(position_filter.covariance[0, 0, 0], 0.04)

    def test_reduces_error_on_a_noisy_static_target(self):
        rng = np.random.default_rng(3)
        truth = np.full(6, 0.4)
        measurements = truth + rng.normal(0.0, 0.18, (400, 6))

        filtered = run(self.make_filter(measurement_noise=0.18 ** 2), measurements)

        raw_error = np.sqrt(np.mean((measurements - truth) ** 2))
        filtered_error = np.sqrt(np.mean((filtered[50:] - truth) ** 2))
        self.assertLess(filtered_error, raw_error / 2.0)

    def test_tracks_a_constant_velocity_ramp_without_systematic_lag(self):
        rng = np.random.default_rng(4)
        steps = 600
        truth = 0.2 + 0.004 * np.arange(steps)
        measurements = (truth + rng.normal(0.0, 0.18, steps)).reshape(-1, 1)

        filtered = run(
            KalmanPositionFilter(np.full(1, 1e-4), np.full(1, 0.18 ** 2)),
            measurements,
        )

        residual = filtered[200:, 0] - truth[200:]
        # A velocity state should leave no systematic offset on a steady ramp.
        self.assertLess(abs(residual.mean()), 0.02)
        self.assertLess(residual.std(), 0.18)


class TestCausalityAndResets(unittest.TestCase):
    def test_output_does_not_depend_on_later_samples(self):
        measurements = np.random.default_rng(5).random((20, 6))
        changed = measurements.copy()
        changed[12:] = 0.0

        for position_filter in (
            EMAPositionFilter(np.full(6, 0.4)),
            KalmanPositionFilter(np.full(6, 1e-4), np.full(6, 0.03)),
        ):
            original = run(position_filter, measurements)
            perturbed = run(position_filter, changed)
            np.testing.assert_array_equal(original[:12], perturbed[:12])

    def test_reset_makes_sequences_independent(self):
        rng = np.random.default_rng(6)
        first = rng.random((15, 6))
        second = rng.random((25, 6))
        stacked = np.vstack([first, second])

        for position_filter in (
            EMAPositionFilter(np.full(6, 0.4)),
            KalmanPositionFilter(np.full(6, 1e-4), np.full(6, 0.03)),
        ):
            together = filter_sequence(position_filter, stacked, [15, 25], 0.1)
            separately = np.vstack([
                run(position_filter, first),
                run(position_filter, second),
            ])
            np.testing.assert_allclose(together, separately)

    def test_filter_sequence_rejects_lengths_that_do_not_describe_the_input(self):
        predictions = np.zeros((10, 6))

        with self.assertRaises(AssertionError):
            filter_sequence(NoFilter(), predictions, [4, 4], 0.1)


class TestCreatePositionFilter(unittest.TestCase):
    def test_none_reproduces_its_input(self):
        predictions = np.random.default_rng(7).random((12, 6))

        position_filter = create_position_filter({"position_filter": "none"})
        filtered = filter_sequence(position_filter, predictions, [12], 0.1)

        np.testing.assert_array_equal(filtered, predictions)

    def test_missing_setting_defaults_to_no_filtering(self):
        self.assertIsInstance(create_position_filter({}), NoFilter)

    def test_builds_the_configured_filters(self):
        ema = create_position_filter({
            "position_filter": "ema",
            "filter_tau": [0.5] * 6,
        })
        kalman = create_position_filter({
            "position_filter": "kalman",
            "filter_process_noise": [1e-4] * 6,
            "filter_measurement_noise": [0.03] * 6,
        })

        self.assertIsInstance(ema, EMAPositionFilter)
        self.assertIsInstance(kalman, KalmanPositionFilter)

    def test_unknown_filter_names_the_available_ones(self):
        with self.assertRaises(ValueError) as caught:
            create_position_filter({"position_filter": "particle"})

        self.assertIn("kalman", str(caught.exception))


class TestEstimateNoise(unittest.TestCase):
    def test_measurement_noise_recovers_the_residual_variance(self):
        rng = np.random.default_rng(8)
        targets = np.zeros((6000, 2))
        scale = np.array([0.2, 0.05])
        predictions = targets + rng.normal(0.0, 1.0, targets.shape) * scale

        process_noise, measurement_noise = estimate_noise(
            predictions, targets, [6000], 0.1)

        np.testing.assert_allclose(measurement_noise, scale ** 2, rtol=0.1)
        self.assertEqual(process_noise.shape, (2,))

    def test_a_constant_velocity_target_has_no_process_noise(self):
        steps = 300
        targets = np.stack([
            0.004 * np.arange(steps),
            0.001 * np.arange(steps),
        ], axis=1)

        process_noise, _ = estimate_noise(targets, targets, [steps], 0.1)

        np.testing.assert_allclose(process_noise, np.zeros(2), atol=1e-12)

    def test_boundaries_are_not_read_as_acceleration(self):
        ramp = (0.004 * np.arange(150)).reshape(-1, 1)
        # Two demonstrations, each a clean ramp, but with a jump between them.
        targets = np.vstack([ramp, ramp + 5.0])

        split, = estimate_noise(targets, targets, [150, 150], 0.1)[:1]
        joined, = estimate_noise(targets, targets, [300], 0.1)[:1]

        np.testing.assert_allclose(split, np.zeros(1), atol=1e-12)
        self.assertGreater(joined[0], 0.0)


class TestTuning(unittest.TestCase):
    def make_data(self, seed=9, steps=400, noise=0.18):
        rng = np.random.default_rng(seed)
        truth = (0.3 + 0.004 * np.arange(steps)).reshape(-1, 1)
        return truth + rng.normal(0.0, noise, (steps, 1)), truth

    def test_tune_ema_returns_the_documented_shape(self):
        predictions, targets = self.make_data()

        tuned = tune_ema(predictions, targets, [400], 0.1)

        self.assertEqual(set(tuned), {"parameters", "rmse", "candidates", "errors"})
        self.assertEqual(tuned["parameters"].shape, (1,))
        self.assertEqual(tuned["errors"].shape, (len(tuned["candidates"]), 1))

    def test_tune_ema_beats_the_unfiltered_predictions(self):
        predictions, targets = self.make_data()

        tuned = tune_ema(predictions, targets, [400], 0.1)

        unfiltered = np.sqrt(np.mean((predictions - targets) ** 2))
        self.assertLess(tuned["rmse"][0], unfiltered)
        # The reported RMSE must be what that parameter actually achieves.
        filtered = filter_sequence(
            EMAPositionFilter(tuned["parameters"]), predictions, [400], 0.1)
        np.testing.assert_allclose(
            np.sqrt(np.mean((filtered - targets) ** 2)), tuned["rmse"][0])

    def test_tune_ema_reports_the_minimum_of_its_own_sweep(self):
        predictions, targets = self.make_data()

        tuned = tune_ema(predictions, targets, [400], 0.1)

        self.assertAlmostEqual(tuned["rmse"][0], tuned["errors"][:, 0].min())

    def test_tune_kalman_beats_the_unfiltered_predictions(self):
        predictions, targets = self.make_data()
        process_noise, measurement_noise = estimate_noise(
            predictions, targets, [400], 0.1)

        tuned = tune_kalman(predictions, targets, [400], 0.1,
                            process_noise, measurement_noise)

        unfiltered = np.sqrt(np.mean((predictions - targets) ** 2))
        self.assertLess(tuned["rmse"][0], unfiltered)

    def test_noisier_fields_prefer_longer_time_constants(self):
        rng = np.random.default_rng(10)
        steps = 3000
        truth = np.tile((0.3 + 0.0005 * np.arange(steps)).reshape(-1, 1), (1, 2))
        predictions = truth + rng.normal(0.0, 1.0, (steps, 2)) * np.array([0.05, 0.4])

        tuned = tune_ema(predictions, truth, [steps], 0.1)

        self.assertLess(tuned["parameters"][0], tuned["parameters"][1])


if __name__ == "__main__":
    unittest.main()
