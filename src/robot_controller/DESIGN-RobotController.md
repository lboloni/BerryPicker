# Robot Controller Architecture

The robot controller is a configuration-driven directed acyclic graph of robot
controller components (RCCOs). Each component has named input and output ports,
trained state when applicable, and runtime context. A complete controller is
assembled by an exp/run that references the component exp/runs and connects
their ports.

The flows that train, verify, and compare controllers are described in
`DESIGN-BehaviorCloningFlow.md`; the staged training recipe in
`data/expruns/robot_controller/DESIGN_Training_Recipe.md`.

## Main abstractions

- `AbstractRobotController` owns the components and connections.
  `reset_context()` resets every component and marks an episode boundary.
- `GraphRobotController` validates the graph, computes its topological order,
  transfers values between ports, and propagates changed inputs through it.
- `AbstractRCComponent` owns named inputs and outputs, port dimensions, a dirty
  flag, and resettable runtime context.
- `rcco_factory` constructs components from the canonical `rcco-type` field.
- `roco_factory` constructs the complete controller selected by the top-level
  `class` field.

Invalid component references, ports, dimensions, multiple writers, cycles, or
missing required model files raise exceptions.

## Components

| `rcco-type` | Class | Ports | Role |
|---|---|---|---|
| `Input` | `RCCO_Input` | → `input` | an externally supplied, preprocessed image tensor |
| `SP_CNN` | `RCCO_SP_CNN` | `image` → `z` | the latent of a proprioception-tuned VGG19 or ResNet-50 CNN, without its regression head |
| `SP_VAE` | `RCCO_SP_VAE` | `image` → `z` | the deterministic latent mean of a Conv-VAE-Neo or VAE-GAN encoder; the decoder is not used |
| `LSTM` | `RCCO_LSTM` | `z` → `h` | a recurrent feature; see "LSTM components and their state" |
| `MLP` | `RCCO_MLP` | `z` → `a` | a deterministic action; a sigmoid output keeps it in `[0, 1]` |
| `MDN` | `RCCO_MDN` | `h` → `mu`, `sigma`, `pi`, `a` | mixture parameters, and an action selected by sampling, expected value, or the most probable component |
| `Output` | `RCCO_Output` | `output` → | the normalized action for the robot-facing caller |

The encoder components accept only the encoder architectures above, because
their trainable and bundle-loadable modules are built from them
(`ProprioTunedCNNSensorProcessing.encoder_classes`, `ConvVAENeoEncoder`).
Other sensor processors (ViT, random projection, legacy Conv-VAE, multi-view)
cannot enter a controller. `RCCO_MLP` reads its input on port `z`, so it can
follow an encoder or an LSTM.

Torch tensors remain on the configured runtime device while moving through the
graph.

## Controller shapes

The supported controllers are linear chains:

```text
image_input -> encoder -> [lstm] -> head -> robot_output
    image     (SP_CNN |    (LSTM)   (MLP |
               SP_VAE) z        h    MDN)   a
```

Examples:

- `roco_cnn_mlp_sample`: ResNet-50 encoder → MLP (one frame, no temporal
  context).
- `roco_vae_neo_lstm_mdn_sample`: Conv-VAE-Neo encoder → residual LSTM
  (sliding window of 10) → MDN with five Gaussian components.
- The flows generate the controller types of `CONTROLLER_TYPES` in
  `rcco_flow.py` (`mlp`, `lstm_mlp`, `lstm_residual_mlp`,
  `lstm_residual_mdn`) on any of the four encoders, with
  `build_controller_graph`. Their component labels are `encoder`, `lstm`, and
  `head`.

The runtime graph itself is a DAG; the restriction to chains comes from
training (`ChainTrainingModel`). A non-chain graph can run, but cannot be
trained by the recipe.

## LSTM components and their state

The recurrent network of `RCCO_LSTM` is one of the registered
`RECURRENT_CORES` in `rcco_lstm.py`, selected by `architecture`:

- `plain`: a stacked `nn.LSTM` with `num_layers` (and optional `dropout`);
- `residual`: `num_layers` single-layer LSTMs, where every layer after the
  first adds its input as a residual.

Every core maps `(sequence [B, T, input], state)` to
`(outputs [B, T, hidden], state)`, with `None` as the zero state. A new
variation (GRU, layer norm, another residual pattern) is one class added to
the registry.

The component owns the runtime context, never exposes it on a port, and clears
it in `reset_context()`. Two `context_mode`s:

| `context_mode` | Runtime | Cost per frame | Output from |
|---|---|---|---|
| `sliding_window` | keeps the last `sequence_length` latents; re-runs the core over them from the zero state | `sequence_length` × layers LSTM steps | the frame that fills the window |
| `stateful` | carries the recurrent state; feeds the core one latent per frame | one step per layer | the first frame |

A component without an output yet (a filling window) returns `False` from
`propagate()`; downstream components keep `None`, and `read_output` returns
`None`. A controller runs one episode at a time (batch size 1). The runtime
state is never saved; checkpoints and bundles hold weights only. The
architecture signature records the core and the context mode, so a bundle
cannot be loaded into a different configuration.

## Training

`StagedControllerTrainingRecipe` (`training_recipe.py`) trains a chain
through `ChainTrainingModel` (`chain_training_model.py`):

- The model follows the connections from the single encoder through an
  optional LSTM to an MLP or MDN head. It builds every module through its
  component with `load_state=False`, so the exported states load into the same
  components at runtime.
- `forward(inputs, state)` takes images `[B, T, C, H, W]` or cached latents
  `[B, T, Z]`, and returns per-step outputs and the recurrent state.
- The head decides the loss: MSE for an MLP, the MDN negative log-likelihood
  for an MDN (with its expected value as the prediction for the MSE and MAE
  metrics).
- Sliding-window and LSTM-free controllers train on windows from
  `RobotControllerSequenceDataset` (`training_data.py`), with the loss on the
  last step. Stateful controllers train with truncated backpropagation through
  time on `RobotControllerChunkLoader`: each batch row follows one
  demonstration over consecutive chunks of `chunk_length` steps; the state is
  carried between the chunks and detached, reset at the start of the
  demonstrations, and the loss covers every existing step. With a
  `chunk_length` at least as long as the demonstrations this is full-sequence
  training.
- In stages whose trainable components exclude the encoder, the recipe
  encodes every frame once at the start of the stage and trains on the cached
  latents.

The trained components are exported as a self-contained bundle;
`GraphRobotController(..., bundle_path=...)` loads all neural component states
from it and validates their architecture signatures before use. Without a
bundle, the encoders load the checkpoint of their sensor-processing exp/run,
and the LSTM, MLP, and MDN components load their own `model_file` from their
result directories.

## Inspection

`load_controller_spec()` resolves and validates the complete graph without
constructing models or loading checkpoints. `RCCOVisualizer` converts this
resolved specification into a Graphviz graph. This allows
`Visualize_roco.ipynb` to inspect and render the architecture independently of
trained artifacts or robot hardware.
