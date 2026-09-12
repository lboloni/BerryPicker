# Training Recipe Design

The robot-controller training recipe is a staged, resumable state machine built
around an existing controller graph. The runtime controller exp/run remains
unchanged. A separate training exp/run references that controller and owns all
source states, checkpoints, metrics, and final component artifacts.

## Staged training

The VAE–LSTM–MDN example uses the following default progression:

| Stage | Trainable components | Frozen components | Purpose |
| --- | --- | --- | --- |
| Initialization | None | All | Resolve, validate, and copy source states |
| MDN warm-up | MDN | VAE encoder and LSTM | Adapt a new action-distribution head |
| Policy fine-tuning | LSTM and MDN | VAE encoder | Jointly adapt the temporal policy |
| End-to-end | VAE encoder, LSTM, and MDN | None | Final joint optimization |

The stages are configured explicitly. If the LSTM is not pretrained, the
MDN-only stage can be omitted or configured to train both the LSTM and MDN.
After each stage, its best validation checkpoint is restored before the next
stage begins.

## Training exp/run

Training runs should live in a separate `robot_controller_training` experiment
family. A representative configuration is:

```yaml
name: "Staged VAE-LSTM-MDN behavior cloning"
class: StagedRobotControllerTrainingRecipe

controller:
  exp: robot_controller
  run: roco_vae_neo_lstm_mdn_sample

model_file: controller_bundle.pth

initial_states:
  vae_encoder:
    mode: configured
    required: true
  lstm:
    mode: configured
    required: true
  mdn:
    mode: random

stages:
  - name: mdn_warmup
    trainable_components: [mdn]
    epochs: 20
    optimizer: Adam
    learning_rates:
      mdn: 0.001
    grad_clip_norm: 1.0
    monitor: validation_nll

  - name: policy_finetune
    trainable_components: [lstm, mdn]
    epochs: 100
    optimizer: Adam
    learning_rates:
      lstm: 0.0003
      mdn: 0.0003
    grad_clip_norm: 1.0
    monitor: validation_nll

  - name: end_to_end
    trainable_components: [vae_encoder, lstm, mdn]
    epochs: 50
    optimizer: Adam
    learning_rates:
      vae_encoder: 0.00001
      lstm: 0.0001
      mdn: 0.0001
    grad_clip_norm: 1.0
    monitor: validation_nll

checkpoint_interval: 1
keep_checkpoints: 3

training_data:
  - ["demonstration-run", "training-demo", "dev2"]

validation_data:
  - ["demonstration-run", "validation-demo", "dev2"]
```

`mode: configured` loads the component's state from its existing exp/run.
Random initialization must always be requested explicitly. A future source
mode can name another training bundle or an explicit exp/run.

## Training model and component state

Training must not use the stateful, inference-oriented
`GraphRobotController.propagate()` path. A batched
`RobotControllerTrainingModel` should reference the same underlying neural
modules and implement:

```text
images [B,T,C,H,W]
    -> deterministic VAE encoder [B,T,Z]
    -> residual LSTM [B,H]
    -> MDN parameters [B,A,K]
```

The controller objective is MDN negative log-likelihood against the normalized
next action. Validation should also record expected-action MSE and MAE.

Model construction and state loading must be separate operations. Each
trainable RCCO should expose a consistent interface such as:

```python
component.model
component.load_state(path)
component.save_state(path)
component.trainable_parameters()
component.architecture_signature()
```

The VAE controller artifact should contain only the deterministic encoder
state (`encoder.*` and `fc_mu.*`). The VAE decoder and `fc_logvar` are not part
of controller training.

At each stage, the recipe derives `requires_grad` and training/evaluation mode
from `trainable_components`. Frozen components remain in evaluation mode. A
new optimizer and scheduler are constructed at every stage because parameter
groups and learning rates can change.

## Training data

A lazy sequence dataset should load images through `Demonstration`, apply the
configured VAE preprocessing, and enforce:

```text
[image(t-T+1), ..., image(t)] -> action(t+1)
```

Complete demonstrations must belong exclusively to training or validation.
The initial implementation should use image sequences for every stage so the
same path works when the VAE encoder is unfrozen. Frozen-encoder latent caching
can be added later as an optimization.

## Recipe implementation

`StagedRobotControllerTrainingRecipe`, derived from
`AbstractTrainingRecipe`, should:

1. Resolve and validate the controller graph.
2. Copy and validate every configured source state.
3. Construct the batched training model and datasets.
4. Read the persistent status and determine the current stage.
5. Configure trainable components and stage-specific optimization.
6. resume or execute the current stage.
7. Restore the stage's best state before advancing.
8. Export all final component states and the aggregate bundle.

A `trec_factory.py` should construct the recipe selected by `exp["class"]`.

## Exp/run data directory

All inputs and outputs needed to reproduce or deploy the trained controller
must be gathered under the training exp/run's `exp.data_dir()`:

```text
robot_controller_training/<run>/
|-- recipe_status.json
|-- resolved_controller.yaml
|-- source_manifest.json
|-- metrics.jsonl
|-- sources/
|   |-- vae_encoder.pth
|   `-- lstm.pth
|-- stages/
|   |-- 00_mdn_warmup/
|   |   |-- checkpoints/
|   |   |-- best_model.pth
|   |   `-- completed.json
|   |-- 01_policy_finetune/
|   `-- 02_end_to_end/
|-- components/
|   |-- vae_encoder.pth
|   |-- lstm.pth
|   `-- mdn.pth
|-- component_manifest.json
`-- controller_bundle.pth
```

Source checkpoints are copied rather than symlinked. The source manifest
records their original exp/run, filename, architecture signature, and
checksum. The final bundle contains the resolved controller identity, every
component state dictionary, architecture signatures, and provenance.

`GraphRobotController(..., bundle_path=...)` accepts this bundle and uses its
component artifacts instead of the component exp/runs' original model paths.

## Interruption and restart

The persistent lifecycle is:

```text
not_started -> initializing -> running -> stage_complete -> completed
                                  |
                                  |-> interrupted -> running
                                  `-> failed ------> running
```

Each epoch checkpoint contains:

- Stage index and name.
- Next epoch.
- Complete aggregate model state.
- Optimizer and scheduler states.
- Best validation metric and early-stopping state.
- Python, NumPy, Torch, and CUDA random-number states.
- Recipe and controller configuration fingerprints.

Status and checkpoint writes are atomic. On `KeyboardInterrupt`, the recipe
records `interrupted`, preserves the latest valid checkpoint, and re-raises.
On another exception, it records `failed`, saves diagnostic state, and
re-raises. The first implementation resumes at epoch boundaries, so a partial
epoch is repeated. A configuration or architecture fingerprint mismatch must
abort recovery.

## Visualization

`TrainingRecipeVisualizer`, `Visualize_trec.ipynb`, and `Train_RCCO.ipynb`
provide the visualization. It
combines the controller graph with a stage timeline and reads the current state
from `recipe_status.json` without loading model weights.

Controller nodes use the following states:

- Green: trainable in the current stage.
- Blue: frozen pretrained component.
- Gray: non-trainable input or output.
- Dashed: not yet initialized.
- Bold border: changed by the current stage.

The timeline shows completed stages with their best metrics, the current stage
and epoch, pending stages, and interrupted or failed status. Before training,
all stages appear as pending. Re-running the visualization cell refreshes the
display.

## Implementation order

1. Separate RCCO model construction from state loading.
2. Add encoder-only VAE state support.
3. Add the batched training model and lazy sequence dataset.
4. Implement staged execution, checkpointing, and recovery.
5. Implement final component gathering and bundle loading.
6. Add persistent status and metrics.
7. Add recipe visualization and its notebook.
8. Test freezing, stage transitions, interruption, recovery, bundle
   completeness, configuration mismatches, and visualization.
