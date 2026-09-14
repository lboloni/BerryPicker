# Robot Controller Architecture

The robot controller is a configuration-driven directed acyclic graph of robot
controller components (RCCOs). Each component has named input and output ports,
trained state when applicable, and runtime context. A complete controller is
assembled by an exp/run that references the component exp/runs and connects
their ports.

## Main abstractions

- `AbstractRobotController` owns the components and connections.
- `GraphRobotController` validates the graph, computes its topological order,
  transfers values between ports, and propagates changed inputs through it.
- `AbstractRCComponent` owns named inputs and outputs, port dimensions, a dirty
  flag, and resettable runtime context.
- `rcco_factory` constructs components from the canonical `rcco-type` field.
- `roco_factory` constructs the complete controller selected by the top-level
  `class` field.

Invalid component references, ports, dimensions, multiple writers, cycles, or
missing required model files raise exceptions.

## VAE–LSTM–MDN controller

The sample controller has the following data flow:

```text
image_input -> vae_encoder -> lstm -> mdn -> robot_output
    image          z           h       a
```

- `RCCO_Input` exposes an externally supplied, preprocessed image tensor.
- `RCCO_SP_VAE` loads a separately trained sensor-processing VAE and emits its
  deterministic latent mean. The decoder is not used during controller
  inference.
- `RCCO_LSTM` maintains a fixed-length sliding window of latent vectors and
  emits the final feature of a residual LSTM stack. It produces no output until
  the window is full.
- `RCCO_MDN` maps the LSTM feature to mixture parameters `mu`, `sigma`, and
  `pi`, and emits an action selected by sampling, expected value, or the most
  probable mixture component.
- `RCCO_Output` exposes the normalized action to the robot-facing caller.

Torch tensors remain on the configured runtime device while moving through the
graph. `reset_context()` clears inputs, outputs, and the LSTM window without
changing trained weights.

## CNN–MLP controller

The deterministic alternative is configured by
`robot_controller/roco_cnn_mlp_sample`:

```text
image_input -> cnn_encoder -> mlp -> robot_output
                    z          a
```

`RCCO_SP_CNN` exposes the latent from a separately trained VGG-19 or ResNet-50
proprioception-tuned CNN while keeping tensors batched. `RCCO_MLP` maps that
latent directly to a normalized action. Its default sigmoid output guarantees
values in `[0, 1]`, matching the robot position normalization contract. This
controller produces an action from one image and therefore has no temporal
context or probabilistic output distribution.

## Configuration and model state

The example is specified by
`robot_controller/roco_vae_neo_lstm_mdn_sample`. Its component runs configure a
256-dimensional VAE latent, a 32-dimensional LSTM feature, and a
six-dimensional MDN action with five Gaussian components.

The VAE checkpoint belongs to its sensor-processing exp/run. The LSTM and MDN
load their own required `model_file` values from their component result
directories. Alternatively, `GraphRobotController(..., bundle_path=...)`
loads all three neural component states from the self-contained artifact
exported by a controller training recipe and validates their architecture
signatures before use.

`load_controller_spec()` resolves and validates the complete graph without
constructing models or loading checkpoints. `RCCOVisualizer` converts this
resolved specification into a Graphviz graph. This allows
`Visualize_roco.ipynb` to inspect and render the architecture independently of
trained artifacts or robot hardware.
