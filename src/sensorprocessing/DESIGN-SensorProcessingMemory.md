# Sensor-processing memory

## Purpose and scope

Extend sensor processing with runtime memory for temporal filters, recurrent
networks, and transformers. The same architecture should support exponential
averaging of estimated positions, Kalman filtering, LSTM encodings, and bounded
histories of sensor features.

The minimal framework is implemented: temporal operation registration, explicit
context routing, inference context/reset, timing forwarding, and loader sequence
boundaries. Specific temporal components, sequence trainers, context persistence,
and temporal cache formats remain unimplemented. The README documents the
implemented interfaces and limitations. This design builds on
[DESIGN-CompositeSensorProcessing.md](DESIGN-CompositeSensorProcessing.md)
and its distinction between inference composition and family-specific training.
Candidate compositions are described in
[DESIGN-ProposedCompositeSPs.md](DESIGN-ProposedCompositeSPs.md).

Use the terminology already present in `AbstractRCComponent`: trained model
**state** is distinct from runtime **context** accumulated during execution.

A temporal SP computes:

```text
(z_t, context_t) = model(observation_t, context_previous, dt)
```

The exp/run still describes one deployable SP. Runtime context belongs to an
observation stream, not to the exp/run's trained weights.

## Tensor-model interface and context ownership

Make context explicit at the tensor-model level:

```python
z, next_context = model.advance(sensor_readings, context, dt=dt)
```

The model does not retain context internally. The caller can use the same model
weights with separate contexts for independent demonstrations or streams. A
context of `None` denotes the beginning of a sequence; each operation defines
its initialization rule. Operations return new context without modifying the
supplied context in place.

The public SP inference wrapper owns one current stream context:

```python
sp.reset_context()
z0 = sp.process(frame0, dt=dt)
z1 = sp.process(frame1, dt=dt)
z2 = sp.process(frame2, dt=dt)
```

`process()` continues to return the current NumPy representation. It calls the
tensor model under inference/no-gradient execution and stores the returned
context after a successful update. Loading model weights starts with fresh
context. Do not retain training computation graphs in the public wrapper.

Existing stateless processors retain their behavior and have a no-op
`reset_context()`. Fixed-rate temporal processors may use a sampling interval
specified by their exp/run; variable-rate callers supply `dt` explicitly.
File, capture, and demonstration helpers must forward timing to the temporal
path without resetting context on every call.

One wrapper represents one stream. Callers handling several streams either use
separate wrappers or manage explicit contexts around the shared tensor model.
Do not introduce a global context registry as part of the initial extension.

## Temporal operations in composites

Ordinary operations retain their existing interface:

```python
output = operation(*inputs)
```

Temporal operations expose:

```python
output, next_context = operation.advance(
    *inputs, context=context, dt=dt
)
```

Operation registration identifies temporal operations. Each temporal step has
one entry in the composite context, keyed by its configured step name:

```python
{
    "position_filter": ...,
    "temporal_encoder": ...,
}
```

The executor retains the current ordered graph. Ordinary steps call `forward`;
temporal steps receive their own previous context and contribute their next
context. Context is separate from named within-timestep results and structured
outputs. Initialize all temporal entries when the composite receives `None`.
Subsequent contexts are expected to contain the correct entries; do not silently
reinitialize missing state.

Memory may appear at any point:

```text
Camera views -> encoders -> fusion -> LSTM -> z
```

```text
Image -> position estimator -> Kalman filter -> position representation
```

```text
                         +-> appearance encoder ----------------+
Image -> shared features +                                      +-> fusion -> z
                         +-> position estimator -> filter ------+
```

Temporal feedback is carried by context, not by cycles in the processing graph.
A temporal equivalent of `forward_steps()` can return named results and next
context for auxiliary training losses. Temporal execution uses the explicit
context path; the existing framewise path must not silently initialize and
discard memory on each call.

## Forms of memory

| Operation | Runtime context | Start-of-sequence behavior |
|---|---|---|
| Exponential averaging | Previous filtered vector | Initialize from first measurement |
| Kalman filter | State estimate and covariance | Use configured prior or first-measurement initialization |
| LSTM | Hidden and cell states | Use zeros or learned initial states |
| Windowed transformer | Recent feature tokens and timing/position information | Start with empty history |
| Cached causal transformer | Attention cache and position information | Start with empty cache |

LSTM input/output state naturally fits this interface.
[PyTorch LSTMCell documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTMCell.html).
Transformer recurrence can retain representations across segments, but this is
a model-specific choice rather than an automatic property of a transformer.
[Transformer-XL](https://arxiv.org/abs/1901.02860).

The first transformer component should retain a bounded window of encoded
features and recompute its temporal encoding each step. This is simpler to train
and compare with streaming inference than a specialized attention cache. Its
exp/run specifies window length, positional encoding, and causal behavior.
Window eviction and initialization must match between training and deployment.

A Kalman component defines its measurement and dynamics semantics explicitly.
For example, measurements may be positions while its context contains position,
velocity, and uncertainty. Do not interpret arbitrary learned latent coordinates
as physical positions. If a downstream proprioception model currently produces
positions, their filter belongs after that model unless the position estimator
is incorporated into the composite SP.

The SP's exposed output size remains `exp["latent_size"]`; context may have a
different shape and need not be exposed downstream. An operation can explicitly
output uncertainty or velocity as part of a structured result when needed.

## Sequence boundaries and reset semantics

Each successful `process()` call advances one timestep. Repeated calls on the
same frame are repeated updates, not repeated reads of a cached output. A
controller should read its stored output without reexecuting the SP.

Reset context explicitly:

- At the start of each demonstration or deployment episode.
- When switching to an independent stream.
- Before evaluating an independent sequence.
- After replacing model weights.

Sequential `process_file()` or `process_demonstration()` calls accumulate
history. Requesting a later frame directly does not reconstruct its preceding
history: the caller replays a prefix, supplies a saved context, or explicitly
evaluates from a reset state. Seeking backwards likewise requires a deliberate
reset/replay or context restoration.

Current proprioception and behavior-cloning loaders reuse an SP across
demonstrations without a reset. Their loops must become sequence-aware:

```python
for demonstration in demonstrations:
    sp.reset_context()
    for timestep in chronological_order:
        z = sp.process(...)
```

Controller `reset_context()` must propagate to the contained SP. Training,
validation, and separate episodes never share runtime context accidentally.

For synchronized multiview input, one complete ordered camera set represents
one timestep. Camera order remains fixed. Asynchronous sensors or per-camera
update rates require a separate extension rather than implicit partial updates.

## Timing

`dt` is elapsed observation time, not inference execution time. Subsampling
frames changes the elapsed time; variable capture intervals must be reflected
in the values passed to temporal operations.

For time-aware exponential averaging, a component can use:

```text
alpha_t = 1 - exp(-dt / tau)
filtered_t = alpha_t * measurement_t + (1 - alpha_t) * filtered_previous
```

`tau` is the configured time constant. The first observation initializes the
filter directly. A Kalman component uses elapsed time in its transition and
process-noise calculations. Transformer components may retain timestamps or
positions when required by their chosen encoding.

Dataset gaps, dropped observations, and prediction-only Kalman updates have
explicit recipe-specific semantics. The executor does not invent measurements
or silently reset context. Assume supplied inputs and timing are correctly
formatted, consistent with the repository's research-code conventions; failures
raise exceptions rather than invoking fallback behavior.

## Training and batching

Training remains specific to each composite family. Explicit context permits
differentiable unrolling:

```python
context = None
outputs = []
for frame, dt in sequence:
    z, context = model.advance(frame, context, dt=dt)
    outputs.append(z)

loss = training_objective(outputs, targets)
loss.backward()
```

Unlike public inference, this path preserves gradients through time. Possible
recipes include:

- Train frame encoders separately, freeze them, then train an LSTM on features.
- Tune temporal-filter parameters against complete trajectories.
- Train transformers on causal feature windows.
- Fine-tune selected frame encoders jointly with temporal modules.
- Alternate objectives or use truncated backpropagation with explicit context
  detachment at chunk boundaries.

Freezing parameters does not freeze runtime context. A frozen LSTM still updates
its memory, and a frozen filter still updates its estimate. As in the composite
design, frozen differentiable operations may transmit gradients upstream.

Independent windows start with fresh context or a defined warm-up prefix.
Consecutive chunks may carry context only when batch lanes retain their stream
identity. Initially use batches with clear sequence boundaries; partial lane
resets and padding require explicit treatment by the family's sequence adapter.
Padded timesteps must not inadvertently advance context or contribute losses.

Do not shuffle individual frames before a temporal model. Independent sequences
or windows may be shuffled if their context initialization remains correct.
For deployment models that are causal, training must not expose future frames
through attention, bidirectional recurrence, or warm-up construction.

A shared unrolling helper can provide a reference implementation. Optimized
LSTM/transformer sequence paths remain family-specific and should match streaming
`advance()` in evaluation for the same inputs, initial context, timing, and
history policy. Family-specific checkpoints record the actual training stage,
optimizers, and context policy as described in the composite design.

## Checkpoints and caches

The composite exp/run owns architecture, temporal settings, learned weights
(including learned initialization parameters), and training artifacts. Ordinary
model loading begins with empty runtime context. Context must not become a
persistent PyTorch buffer accidentally included in the deployed `state_dict()`.

Exact mid-sequence runtime resumption can save a separate context artifact with
stream position, timing information, and model identity. Exact training
resumption additionally needs the recipe's data/optimizer and other relevant
state. A detached context snapshot does not preserve an unfinished autograd
graph; truncated-backpropagation recipes should checkpoint at their defined
boundaries or replay the necessary prefix.

Temporal output caches include sequence identity, reset/warm-up policy, timing,
and model version. The same frame can yield different outputs after different
histories. Frozen stateless frame features may be cached independently, but
temporal features must be generated with their declared sequence semantics.
Changing upstream weights, timing, or the history policy invalidates dependent
caches.

## Minimal architectural changes

- Add explicit tensor-model `advance()` and temporal operation registration.
- Route per-step context through the ordered composite executor.
- Add context ownership and `reset_context()` to SP inference wrappers, with
  no-op reset behavior for stateless processors.
- Forward timing through temporal inference helpers.
- Reset at sequence boundaries in data loaders and controller integrations.
- Provide a reference sequence-unrolling path without imposing one training
  algorithm on all model families.

EMA, Kalman, LSTM, and transformer components remain separate implementations.
The existing framewise execution path remains available for stateless models.
No implicit global memory, graph scheduler, or automatic episode inference is
required.

Verify reset reproducibility, independence of stream contexts, multiview
timestep semantics, bounded transformer history, gradient flow through time,
explicit detachment, and matching streaming/sequence evaluation. Check that
ordinary model exports exclude runtime context and that temporal cache and
resume behavior reflect the actual observation history.
