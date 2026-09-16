# Temporal visual proprioception

## Purpose and scope

Visual proprioception currently estimates the robot position from one frame at a
time. Consecutive frames of a demonstration are highly redundant: the arm moves
slowly relative to the camera rate, so each frame is nearly a repeated
measurement of the same position. This document examines which recursive
estimators can exploit that redundancy, where such an estimator belongs, and
what the repository would need before one can be configured.

Nothing described here is implemented. The temporal execution framework already
exists and is documented in
[DESIGN-SensorProcessingMemory.md](../sensorprocessing/DESIGN-SensorProcessingMemory.md);
this document does not restate its context, timing, or reset semantics. What is
missing is any concrete temporal component: no operation in `src/` is registered
with `temporal=True`, and the only such registration in the repository is a test
double in `src/test/sensorprocessing/test_composite.py`. The one composite
exp/run on disk, `sensorprocessing_composite/appearance_foreground_256`, is
purely spatial.

This document is a proposal and a measurement record. It does not claim that any
estimator below improves BerryPicker performance; it establishes which ones the
data justifies testing, and in what order.

## Measured signal characteristics

The numbers below come from the `random-both-cameras-video` demopack, which the
`visual_proprioception` defaults use for training and validation. They are
recorded here because the choice of estimator depends on them entirely.

Observation rate is 10 Hz: `tick_interval: 0.1` in the
`demonstration_collector` exp/runs, `fps: 10` for both cameras in the demopack.

The `time` field in each demonstration's `_action.yaml` is a **frame counter**,
not elapsed seconds: it runs 0, 1, 2, ... with a difference of exactly 1. There
is therefore no recorded wall-clock timing for these demonstrations, consistent
with the memory design's statement that no timestamp convention exists. A
temporal component must take `dt` from its configured `sample_interval`
(0.1 seconds) or from an explicit `timestep_interval` callback.

Per-frame change of the target position, expressed in the normalized `[0, 1]`
coordinates that the regressor actually predicts, measured over the two
validation demonstrations (`2025_03_08__14_23_19`, 459 frames and
`2025_03_08__14_24_52`, 472 frames; 929 within-demonstration transitions):

| Field | mean change | 99th percentile | max |
|---|---|---|---|
| height | 0.0078 | 0.0250 | 0.0250 |
| distance | 0.0068 | 0.0143 | 0.0143 |
| heading | 0.0056 | 0.0083 | 0.0083 |
| wrist_angle | 0.0070 | 0.0167 | 0.0167 |
| wrist_rotation | 0.0034 | 0.0083 | 0.0083 |
| gripper | 0.0082 | 0.0500 | 0.0500 |

The complete six-vector moves 0.0241 on average and never more than 0.0612 in
L2 norm between consecutive frames. The commanded trajectory is smooth and
rate-limited rather than piecewise-constant with jumps.

Note that the recorded target is `rc-position-target`, the controller setpoint.
Only target fields are stored (`rc-position-target`, `rc-angle-target`,
`rc-pulse-target`); no measured arm position exists in the demonstrations. The
physical arm lags the setpoint, and this document's error figures inherit that
convention rather than correcting for it.

## Why a recursive estimator is justified

Normalized per-field RMSE of the strongest existing single-view model,
`ptun-vgg19-128`, taken from the saved comparison
`msecomparison_values.txt` produced by
[Compare_VisualProprioception.ipynb](Compare_VisualProprioception.ipynb). The
metric there is `sqrt(mean((y - ypred) ** 2))` over the concatenated validation
set.

| Field | RMSE | mean motion/frame | ratio |
|---|---|---|---|
| height | 0.184 | 0.0078 | 24 |
| distance | 0.111 | 0.0068 | 16 |
| heading | 0.067 | 0.0056 | 12 |
| wrist_angle | 0.163 | 0.0070 | 23 |
| wrist_rotation | 0.149 | 0.0034 | 44 |
| gripper | 0.320 | 0.0082 | 39 |

Estimation error exceeds the per-frame motion of the estimated quantity by a
factor of 12 to 44. On the timescale over which the error fluctuates, the
position is nearly constant. This is the regime in which recursive estimation
gives the largest return, and it is a stronger argument for temporal filtering
than for any additional single-frame architecture.

A first-order estimate of what plain exponential averaging can recover, trading
the white-noise variance floor `sigma^2 * alpha / (2 - alpha)` against the
constant-velocity lag bias `v * (1 - alpha) / alpha`:

| Field | RMSE | best alpha | tau (frames) | predicted RMSE | gain |
|---|---|---|---|---|---|
| height | 0.184 | 0.17 | 6 | 0.068 | 2.7x |
| distance | 0.111 | 0.21 | 5 | 0.046 | 2.4x |
| heading | 0.067 | 0.25 | 4 | 0.030 | 2.2x |
| wrist_angle | 0.163 | 0.17 | 6 | 0.060 | 2.7x |
| wrist_rotation | 0.149 | 0.12 | 9 | 0.045 | 3.3x |
| gripper | 0.320 | 0.13 | 8 | 0.101 | 3.2x |

Time constants land at 4 to 9 frames, or 0.4 to 0.9 seconds at 10 Hz.

Treat these as an upper bound, not a prediction. The derivation assumes the
per-frame error is temporally white. A convolutional encoder makes correlated
mistakes on visually similar consecutive frames, and correlated error averages
out more slowly than independent error. The gap between this bound and the
measured result is itself informative: it estimates how much of the error is
systematic rather than observation noise, and a systematic component cannot be
removed by any causal filter.

## Measured result: the bound does not hold

`Flow_FilteredVsUnfiltered.ipynb` has since been run end to end on the
random-projection VGG19 encoder (`vgg19_rademacher_128`, 1807 training frames
in four demonstrations, evaluated on two held-out ones). Parameters were tuned
on the training demonstrations by `tune_ema` / `tune_kalman`. Normalized RMSE
on the held-out set:

| Field | unfiltered | EMA | Kalman | best gain |
|---|---|---|---|---|
| height | 0.2298 | 0.2186 | 0.2229 | 1.05x |
| distance | 0.1380 | 0.1342 | 0.1362 | 1.03x |
| heading | 0.1179 | 0.1131 | 0.1157 | 1.04x |
| wrist_angle | 0.2104 | 0.1976 | 0.1939 | 1.08x |
| wrist_rotation | 0.1615 | 0.1588 | 0.1605 | 1.02x |
| gripper | 0.3260 | 0.3238 | 0.3276 | 1.01x |

**One to eight percent, not two to three times.** The tuner also chose time
constants of 0.13 to 0.47 seconds, that is 1.3 to 4.7 frames, well below the
4 to 9 frames the white-noise model predicted: given the real error, heavier
smoothing costs more in lag than it recovers in noise.

The cause is the correlation the section above flagged, and it is much stronger
than that caveat implied. Autocorrelation of the residual within a
demonstration:

| Lag (frames) | height | distance | heading | wrist_angle | wrist_rotation | gripper |
|---|---|---|---|---|---|---|
| 1 | 0.764 | 0.714 | 0.638 | 0.889 | 0.940 | 0.943 |
| 5 | 0.530 | 0.492 | 0.390 | 0.788 | 0.872 | 0.788 |
| 10 | 0.369 | 0.344 | 0.229 | 0.689 | 0.787 | 0.560 |
| 20 | 0.205 | 0.158 | 0.079 | 0.511 | 0.594 | 0.197 |

White error would show roughly zero at lag 1. Splitting the residual into a
five-frame moving average and the remainder, the fast part carries only 4% of
the variance for `gripper`, 9% for `wrist_angle` and at most 29% for `heading`.
The other 71 to 96% is a slowly varying bias: the encoder is not making
independent mistakes about a static arm, it is confidently wrong about what it
is looking at for stretches of many frames. No causal filter removes that.

The per-field ordering confirms the mechanism: `heading` has the largest fast
share (29%) and the largest EMA gain, `gripper` the smallest (4%) and the
smallest gain. The achievable gain is bounded by `1 / sqrt(1 - fast share)`,
which is 1.18x for `heading` and 1.02x for `gripper`. The filters are already
close to that ceiling, so there is nothing left for a better-tuned filter, a
longer window, or a smarter estimator of the same kind to recover.

By this document's own criterion, the answer is a better encoder rather than a
filter. Two consequences:

- Filtering is cheap and slightly positive, so it is reasonable to leave on.
  It is not a route to a materially better proprioceptor.
- The LSTM and switching-model proposals below inherit this result. Their
  advantage over EMA would have to come from modelling the *systematic* error,
  not from averaging observation noise, and nothing measured here suggests they
  can. Establishing that the residual bias is predictable from the observation
  stream should precede implementing them.

This was measured on the training-free random-projection encoder, whose
unfiltered error is larger than the proprioception-tuned CNN's. A stronger
encoder has less error to remove and, unless its error is markedly less
correlated, less to gain.

## Where the estimator belongs

Two placements are possible, and they are not equivalent.

```text
Image -> encoder -> z -> regressor -> position -> FILTER -> filtered position
```

```text
Image -> encoder -> z -> FILTER -> filtered z -> regressor -> position
```

Position-space filtering operates on physically meaningful, separately bounded
coordinates with known limits and a plausible dynamics model. Latent-space
filtering operates on learned coordinates with no physical interpretation and no
per-dimension scale.

The memory design already rules on this: arbitrary learned latent coordinates
must not be interpreted as physical positions, and when a downstream model
produces positions, the filter belongs after that model unless the position
estimator is itself incorporated into the composite. That places a Kalman
filter, and any estimator with a dynamics model, in position space.

This creates a structural mismatch worth stating plainly. The composite
framework's temporal slot sits inside the sensor processor, before the
regressor, so the placement it supports today is exactly the one that suits the
fewest estimators. Exponential averaging and a recurrent encoder are defensible
on latents; a Kalman filter is not. Filtering in position space within the
framework requires the regressor to become a composite operation, so that one
composite spans encoder, regressor, and filter. The regressor is currently
constructed only in notebooks
([visproprio_models.py](visproprio_models.py), instantiated in
`Train_`/`Verify_`/`Compare_VisualProprioception*.ipynb`) and is not reachable
from a composite step.

## Candidate estimators

| Estimator | Placement | Justified by the data | Priority |
|---|---|---|---|
| Exponential moving average | Position, or latent | Yes; rate-limited smooth trajectory | First |
| Kalman filter, constant velocity | Position | Yes; also yields covariance | Second |
| Sliding-window LSTM | Latent | Plausible; needs sequence training | Third |
| Switching state space / continuous HMM | Position | Weakly, and not on this demopack | Later |
| Particle filter | Position | Not by this data | Not recommended |

### Exponential moving average

The highest value for the least machinery, and the baseline every other method
must beat. The memory design gives the time-aware form directly:

```text
alpha_t = 1 - exp(-dt / tau)
filtered_t = alpha_t * measurement_t + (1 - alpha_t) * filtered_previous
```

Runtime context is the previous filtered vector; the first observation
initializes it. One configured parameter, `tau`. Per-field time constants differ
by more than a factor of two in the table above, so `tau` should be a
six-vector rather than a scalar, or the fields should be filtered by separate
steps.

### Kalman filter, constant velocity

State is position and velocity per degree of freedom, context is the state
estimate and its covariance, and `dt` enters the transition and process-noise
terms. The measured per-frame motion bounds give a defensible starting process
noise, and the per-field RMSE gives a starting measurement noise; both then need
tuning against complete trajectories rather than frames.

Beyond accuracy this is the only candidate that produces a calibrated
uncertainty. Nothing in the current pipeline does: the multiview fusion design
notes explicitly that learned gate weights are not automatically calibrated
confidence estimates. A per-field covariance would be usable by a downstream
controller and would also make disagreement between camera views measurable.

Six independent single-degree-of-freedom filters are the correct first version.
A full 12-dimensional state with cross-covariance assumes correlations between
degrees of freedom that have not been measured.

### Sliding-window LSTM

The repository already contains one, in the controller layer rather than this
one: [`RCCO_LSTM`](../robot_controller/rcco_lstm.py) is a residual LSTM stack
over a bounded `deque` of latent vectors, with `reset_context()`, an
architecture signature, and staged training through the robot-controller recipe.
It consumes `z` and produces a feature for the MDN action head. It is not used
anywhere in the visual-proprioception path.

Two routes exist: register an analogous temporal operation for composite sensor
processing, or reuse the RCCO component pattern. Either way the architecture is
not the obstacle; training is. See the prerequisites below.

An LSTM subsumes exponential averaging in principle, but it has to learn what
EMA gets for free from one parameter, and it can overfit the particular
trajectories of a small demonstration set. It should be evaluated against a
tuned EMA, not against the unfiltered regressor.

### Switching state space and continuous HMM

Worth considering only if the trajectory has discrete regimes, such as moving
versus stationary versus gripper actuation. The `gripper` field is the one
plausible candidate: it has both the largest per-frame steps (0.05 normalized)
and the worst RMSE (0.320).

The obstacle is the data rather than the method. `random-both-cameras-video`
records random-pose motion with no task structure, so regime boundaries are
weak in exactly the demopack the benchmarks use. A task demopack such as
`touch-apple` is where this hypothesis should be tested, and it should be tested
against a plain Kalman filter rather than against no filter.

### Particle filter

Not recommended for this problem. A particle filter earns its cost when the
belief is multimodal or the dynamics strongly nonlinear. Here the state is
six-dimensional, the trajectory is rate-limited and near-linear, and the
measurement is a single point estimate; a Kalman filter is the matched estimator
at a fraction of the cost.

It also lacks an input. A particle filter needs an observation likelihood, and
`VisProprio_SimpleMLPRegression` emits a bare point estimate with no variance
head. Representing genuine ambiguity, such as a bimodal arm-pose belief under
occlusion, would require a probabilistic output head first. That head, not the
filter, is the interesting change, and it should be justified on its own terms.

## Prerequisites

Ordered by what blocks what. None of these are satisfied today.

1. No temporal operation is registered, so no temporal composite can be
   configured. This blocks everything inside the framework.
2. The visual-proprioception training path shuffles frames:
   `DataLoader(..., shuffle=True)` in the training notebooks and
   `split_training_validation(..., shuffle=True)` in
   [visproprio_helper.py](visproprio_helper.py). The memory design forbids
   shuffling individual frames before a temporal model. Inference-time EMA and
   Kalman filtering are unaffected; anything learned over time is blocked.
3. Position-space filtering has no in-framework home until the regressor is
   reachable as a composite operation.
4. The regressor produces no uncertainty, so Kalman measurement noise must be
   calibrated offline and a particle filter has no likelihood.
5. `dt` must be configured as `sample_interval: 0.1` rather than derived from
   the data, because the recorded `time` field is a frame counter.
6. Temporal sensor processors bypass the latent cache, so every training epoch
   re-encodes every frame. Sequence training of a temporal component is
   substantially slower than the current cached-latent regressor training.

Items 1 and 2 are independent: the first experiment below needs neither.

## First experiment: offline post-filtering

The central question -- how much error is observation noise that a causal filter
can remove -- can be answered without changing the framework, and should be
answered before any operation is implemented.

Validation latents are loaded in demonstration order and `test_loader` uses
`shuffle=False`, so the `ypred` array produced by
[Verify_VisualProprioception.ipynb](Verify_VisualProprioception.ipynb) is
already chronological. Apply an exponential average and a scalar Kalman filter
to that array directly, sweep `tau` and the noise ratio per field, and recompute
the same RMSE that `Compare_VisualProprioception.ipynb` computes.

Reset the filter at the demonstration boundary. The validation set concatenates
two demonstrations, 459 frames followed by 472, so the transition after index
458 is a discontinuity rather than arm motion. Filtering across it measures the
wrong thing.

This yields a measured per-field gain to compare against the 2.2x to 3.3x
estimate, distinguishes observation noise from systematic error, and produces
the `tau` values a registered operation would then be configured with. If the
measured gain is far below the bound, the error is largely systematic and the
right response is a better encoder rather than a filter.

## Measurement

Report per-field normalized RMSE, not only the aggregate: the ratios above span
a factor of nearly four across fields, and a filter that helps `wrist_rotation`
may do nothing for `heading`.

Compare against a tuned EMA rather than against the unfiltered regressor, so
that a complex estimator has to demonstrate an advantage over the simplest one.

Separate lag from noise. A causal filter trades one for the other, and a metric
averaged over a whole trajectory hides the cost at direction reversals. Report
error during fast motion separately from error while nearly stationary, using
the per-frame motion of the target to partition the frames.

Verify that filtered predictions remain inside `[0, 1]`. The regressor's output
head is linear and unbounded, and `RobotPosition.from_normalized_vector` rejects
values outside the range. A filter narrows the distribution and should reduce
these violations; confirm that rather than assuming it.

Check that streaming and batch evaluation agree for the same inputs, timing, and
initial context, as required of any temporal component by the memory design.

Measure added latency in the control loop, not only offline accuracy. EMA and
Kalman costs are negligible; a recurrent or windowed component is not
automatically so.
