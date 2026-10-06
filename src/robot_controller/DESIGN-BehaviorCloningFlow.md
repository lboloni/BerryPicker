# Design: a flow for training and verifying an RCCO behavior cloning controller

Status: Phase 1 (ResNet-50 CNN + MLP) and Phase 2 (other sensor processors,
separate controller and comparison flows) implemented.

The sections from "Goal" to "Phase 1 known limitations" describe Phase 1;
"Phase 1 implementation record" lists what was built and how it differs
from the design. Phase 2 follows at the end.

## Goal

Create a flow notebook that, starting from a demopack, trains and verifies
an RCCO-based behavior cloning controller with the existing flow machinery
(`src/flow.py`, `data/expruns/DESIGN-Flows.md`).

The first version is deliberately narrow:

- architecture: CNN encoder + deterministic MLP (`SP_CNN -> MLP`), trained by
  the existing `StagedCNNMLPTrainingRecipe`;
- robot: AL5D only (`robot_al5d/position_controller_00`), action type
  `rc-position-target`, normalized to [0, 1];
- data: designed for the `touch-apple` demopack (11 demonstrations, camera
  `dev0`, 189 steps each, `rc-position-target` actions present), which the
  rest of this document uses as the example.

  **Current default: `automove-pack-01` (camera `dev2`).** About 10% of the
  touch-apple position targets (9 of 11 demonstrations) lie outside the
  AL5D `POS_MIN`/`POS_MAX` (height 0.65–6.24 vs. [1, 5], distance up to 11.06
  vs. 10). They were recorded before commit `19e32a3` made
  `RobotPosition.to_normalized_vector` reject unsafe positions, so Stage 1
  fails on them. `automove-pack-01` (10 demonstrations of random motion) is
  within the limits; it validates the pipeline but does not learn a task.
  The touch-apple parameters remain in the flow notebook as comments.

LSTM/MDN controllers, other robots, proprioception inputs, and running the
controller on the robot are out of scope; the structure below is chosen so
that they can be added as further generators and stages.

## Flow overview

```text
touch-apple demopack
   │ import_demopack(group_chooser_sp_bc_standard)
   ▼
results/demonstration/touch-apple/{sp_*, bc_*}_000NN
   │
   ├─ Stage 1  Train_ProprioTuned_CNN   sensorprocessing_propriotuned_cnn/_flow_sp_resnet50_256
   │           (trained on sp_training, validated on sp_validation)
   │
   ├─ Stage 2  Train_RCCO               robot_controller_training/_flow_trec_cnn_mlp
   │           (cnn_encoder initialized from Stage 1; trained on bc_training,
   │            validated on bc_validation; exports controller_bundle.pth)
   │
   └─ Stage 3  Verify_RCCO (new)        robot_controller_verify/_flow_verify_cnn_mlp
               (loads the bundle through GraphRobotController; teacher-forced
                predictions on bc_testing; per-joint errors and plots)
```

The final report links the verification directory and previews its PNGs.

## Data split

`import_demopack(demopack_path, group_chooser_sp_bc_standard)` already
produces the groups needed. For 11 demonstrations (sorted by name):

| Group | Demonstrations | Used by |
|---|---|---|
| `sp_training` | 0–3 (4) | Stage 1 training |
| `sp_validation` | 4–5 (2) | Stage 1 validation |
| `bc_training` | 4–7 (4) | Stage 2 training |
| `bc_validation` | 0–1 (2) | Stage 2 validation |
| `bc_testing` | 8–10 (3) | Stage 3 verification |

Entries take the usual form `["touch-apple", "<group>_000NN", "dev0"]`.
Training and validation of each stage are disjoint by demonstration, as
`make_controller_dataloaders` requires. `bc_testing` is held out from both
stages; `bc_validation` reuses SP training demonstrations, which is
acceptable because the SP is trained on proprioception, not on actions.

## New and changed files

| File | Change |
|---|---|
| `src/robot_controller/Flow_RCCO_BehaviorCloning.ipynb` | New flow notebook. |
| `src/robot_controller/Verify_RCCO.ipynb` | New stage notebook (Stage 3). |
| `src/robot_controller/controller_verification.py` | New, small: teacher-forced prediction helper used by `Verify_RCCO`. |
| `data/expruns/robot_controller_verify/_defaults_robot_controller_verify.yaml` | New experiment family; declares `input-to-notebook: [robot_controller/Verify_RCCO.ipynb]` and the graph labels `sp_component`, `input_label`, `output_label`. |
| `data/expruns/robot_controller_verify/verify_cnn_mlp_sample.yaml` | New: verifies `trec_cnn_mlp_sample` outside a flow. |
| `src/test/robot_controller/test_controller_verification.py` | New test. |
| `data/expruns/DESIGN-Flows.md` | Add the flow to "Current flows". |
| `src/test/test_exprun_notebooks.py` | Add the flow notebook to `FLOW_NOTEBOOKS`. |

No change is needed to `flow.py`, the training recipe, the training model,
the graph controller, or `Train_RCCO.ipynb`. `Train_RCCO` already has the
standard stage parameters (`experiment`, `run`, `creation_style`,
`expruns_path`, `results_path`), ends with `exp.done()`, and is declared as
`input-to-notebook[0]` of `robot_controller_training`.

## The flow notebook

`Flow_RCCO_BehaviorCloning.ipynb` follows the structure of
`Flow_BehaviorCloning.ipynb` and the contract in `DESIGN-Flows.md`.

**Parameters cell** (tagged `parameters`):

```python
flow_name = "RCCO-BC-touch-apple"
demopack_name = "touch-apple"
demonstration_cam = "dev0"
epochs_sp = 100        # Stage 1 epochs
epochs_warmup = 20     # Stage 2, mlp_warmup
epochs_end_to_end = 20 # Stage 2, end_to_end
creation_style = "exist-ok"
```

**Setup cell:**

```python
expruns_path, results_path, notebooks_path = setup_flow(flow_name, [
    "demonstration", "robot_al5d", "sensorprocessing_propriotuned_cnn",
    "robot_controller", "robot_controller_training",
    "robot_controller_verify"])
demopack_path = pathlib.Path(Config()["demopacks_path"], demopack_name).expanduser()
selection = import_demopack(demopack_path, group_chooser_sp_bc_standard)
data = {group: [[demopack_name, demo, demonstration_cam]
                for demo in selection[group]]
        for group in ["sp_training", "sp_validation", "bc_training",
                      "bc_validation", "bc_testing"]}
```

**Generator functions.** Each writes one exprun YAML into
`Config().get_exprun_path()/<experiment>/<run>.yaml` and, where it is a
stage, returns `flow_entry(...)`. All generated runs are prefixed `_flow_`.

1. `generate_sp_cnn(...)` writes
   `sensorprocessing_propriotuned_cnn/_flow_sp_resnet50_256.yaml`: the
   values of `resnet50_256.yaml` with `epochs = epochs_sp` and
   `training_data`/`validation_data` from `sp_training`/`sp_validation`.
   Returns the entry for `input-to-notebook[0]` (Train_ProprioTuned_CNN).

2. `generate_controller(...)` writes three `robot_controller` runs and
   returns nothing (they are configuration, not stages):
   - `_flow_rcco_sp_cnn.yaml`: `rcco-type: SP_CNN`,
     `sp_experiment: sensorprocessing_propriotuned_cnn`,
     `sp_run: _flow_sp_resnet50_256`;
   - `_flow_roco_cnn_mlp.yaml`: `roco_cnn_mlp_sample.yaml` with the
     `cnn_encoder` component pointing to `_flow_rcco_sp_cnn`; the input,
     MLP (`rcco_mlp_256_6`), and output components and the connections are
     unchanged.

3. `generate_trec(...)` writes
   `robot_controller_training/_flow_trec_cnn_mlp.yaml`: the values of
   `trec_cnn_mlp_sample.yaml` with `controller.run = _flow_roco_cnn_mlp`,
   the stage epochs from the parameters, and `training_data` /
   `validation_data` from `bc_training` / `bc_validation`. Returns the entry
   for `input-to-notebook[0]` (Train_RCCO).

   The recipe's `initial_states.cnn_encoder.mode: configured` loads the
   encoder weights from Stage 1 (`model_file` of the SP run), so Stage 1 must
   precede Stage 2 in the queue.

4. `generate_verify(...)` writes
   `robot_controller_verify/_flow_verify_cnn_mlp.yaml`:

   ```yaml
   name: "Verify CNN-MLP controller on touch-apple"
   trec_experiment: robot_controller_training
   trec_run: _flow_trec_cnn_mlp
   testing_data:
     - ["touch-apple", "bc_testing_00000", "dev0"]
     - ...
   ```

   Returns the entry for `input-to-notebook[0]` (Verify_RCCO).

**Run and report cells** are the standard ones:

```python
flow_error = None
try:
    run_flow(entries, expruns_path, results_path, notebooks_path)
except Exception as error:
    flow_error = error
```

```python
final_results_path = pathlib.Path(
    results_path, "robot_controller_verify", "_flow_verify_cnn_mlp")
report = display_flow_report(
    entries, results_path, notebooks_path, final_results_path,
    flow_error=flow_error,
    preview_files=sorted(final_results_path.glob("*.png")))
if flow_error is not None:
    raise flow_error
if not report["all-results-present"]:
    raise Exception("Flow finished without all expected stage results.")
```

## Stage 3: Verify_RCCO

Verification exercises the deployed path: the exported bundle loaded by
`GraphRobotController`, not the training model. A mismatch between the
training graph and the deployed graph therefore shows up here.

**Notebook cells:**

1. Standard parameters cell (`experiment = "robot_controller_verify"`,
   `run`, `creation_style`, `expruns_path = None`, `results_path = None`),
   then the standard path-setting and `get_experiment` cell.
2. Load the trained controller:

   ```python
   exp_trec = Config().get_experiment(exp["trec_experiment"], exp["trec_run"])
   exp_roco = Config().get_experiment(
       exp_trec["controller"]["exp"], exp_trec["controller"]["run"])
   exp_robot = Config().get_experiment(
       exp_trec["robot"]["exp"], exp_trec["robot"]["run"])
   controller = create_controller(
       exp_roco, bundle_path=exp_trec.data_dir() / exp_trec["model_file"])
   ```

3. For each `testing_data` entry, compute teacher-forced predictions with
   `controller_verification.teacher_forcing(controller, entry, ...)`.
4. Save results into `exp.data_dir()`:
   - `predictions_<demo>.npz` with `predicted` and `target` arrays
     (normalized, shape `[T, 6]`);
   - `errors.csv`: one row per demonstration plus an `all` row; columns
     are the MSE of each of the 6 joints and their mean;
   - `prediction_<demo>.pdf` and `.png`: 6 subplots (one per joint),
     predicted vs. target over time;
   - `errors.pdf` and `.png`: per-joint MSE bar chart over all test
     demonstrations.

   Every figure is saved as a PDF and a same-basename PNG, as
   `DESIGN-Flows.md` requires for flow previews.
5. `exp.done()` as the last cell.

**`controller_verification.py`** holds the part worth testing:

```python
def teacher_forcing(controller, entry, sensor_exp, robot_exp, output_size,
                    input_label="image_input", output_label="robot_output",
                    **dataset_kwargs):
    """Return (predicted, target) arrays of shape [T, output_size] for one
    demonstration, feeding each recorded frame to the controller."""
```

It builds a `RobotControllerSequenceDataset([entry], sensor_exp, robot_exp,
sequence_length=1, output_size=output_size)`, so frames are preprocessed exactly as in
training, and the target at step t is the recorded action at t+1. For each
sample it calls `controller.reset_context()` once at the start, then
`receive_input(input_label, image, t)`, `propagate()`, and
`read_output(output_label)`. The notebook takes `sensor_exp` from the
controller spec (`controller.spec["components"][sp_component]["sensor_exp"]`),
that is, the SP experiment the controller's `SP_CNN` component was built
from. The output size is `len(RobotPosition.FIELDS)`, and the plots and
`errors.csv` are labeled with the AL5D field names.

The input and output labels are those of `roco_cnn_mlp_sample.yaml`
(`image_input`, `robot_output`); they are parameters so that later
controllers with other labels can reuse the helper.

## Tests

- `test_controller_verification.py`: a recording controller (only the
  controller interface is used) over a fake demonstration factory; checks
  that `teacher_forcing` returns one prediction per sample, aligns targets
  to t+1, and resets context once.
- `test_exprun_notebooks.py` covers the new notebooks once the flow is
  added to `FLOW_NOTEBOOKS`: the flow notebook must use `setup_flow(` and `run_flow(` and end with
  `display_flow_report(` and `raise flow_error`; `Verify_RCCO` must have
  the standard parameters and end with `exp.done()`; both notebooks it
  queues must be declared in an `input-to-notebook`.

## Verification of the implementation

1. `PYTHONPATH=src python -m unittest src.test.test_exprun_notebooks src.test.test_flows src.test.robot_controller.test_controller_verification -v`
2. Run `Flow_RCCO_BehaviorCloning` with small epochs (`epochs_sp = 2`,
   `epochs_warmup = 1`, `epochs_end_to_end = 1`): all three stages
   complete, the stage table shows their durations, and the per-joint
   prediction plots appear inline.
3. Run it with the default epochs and check that the predicted trajectories
   on `bc_testing` follow the targets.

## Phase 1 implementation record

Implemented as designed, in:

- `src/robot_controller/Flow_RCCO_BehaviorCloning.ipynb` (flow);
- `src/robot_controller/Verify_RCCO.ipynb` (Stage 3);
- `src/robot_controller/controller_verification.py` with
  `src/test/robot_controller/test_controller_verification.py`;
- `data/expruns/robot_controller_verify/` (`_defaults_...`,
  `verify_cnn_mlp_sample.yaml`);
- `data/expruns/DESIGN-Flows.md` ("Current flows") and
  `src/test/test_exprun_notebooks.py` (`FLOW_NOTEBOOKS`).

Differences from the design above:

- **Default data.** The flow defaults to `automove-pack-01` / `dev2`, not
  `touch-apple` (see the data bullet under "Goal"). touch-apple remains in
  the parameters cell as comments.
- **`teacher_forcing` signature** takes `output_size` and passes extra
  keyword arguments to `RobotControllerSequenceDataset` (used by the test
  to inject a fake demonstration factory).
- **Verify_RCCO** takes `sensor_exp` from the controller spec
  (`controller.spec["components"][sp_component]["sensor_exp"]`) and labels
  plots and `errors.csv` with `RobotPosition.FIELDS`. The graph labels
  (`sp_component`, `input_label`, `output_label`) are defaults of the
  `robot_controller_verify` family.
- **Demopack path** uses `.expanduser()`, since `demopacks_path` may be
  given as `~/...` (the older `Flow_BehaviorCloning` lacks this).
- **Test** uses a recording controller instead of a graph controller.

Results: a run on `automove-pack-01` with debug epochs (2 / 1 / 1) completed
all three stages in 1 min 40 s (TrainSP 24 s, TrainRCCO 57 s, VerifyRCCO
20 s); test MSE 0.064 averaged over the six normalized fields. A run with
the default epochs has not been completed yet.

## Phase 1 known limitations

- **Single SP architecture.** The SP stage trains ResNet-50/256 only
  (addressed by Phase 2).
- **Encoder cost.** With `freeze_feature_extractor: True` in Stage 1 and an
  `end_to_end` stage in Stage 2, Stage 2 runs the CNN on every frame of
  every epoch; latent caching does not exist yet.
- **Teacher forcing only.** Verification feeds recorded frames; it does not
  measure closed-loop behavior. Running on the AL5D needs a separate
  Run_RCCO notebook with step clamping, as in `Run_BehaviorCloning`.
- **No comparison stage.** With one controller there is nothing to compare
  (addressed by Phase 2).

# Phase 2: other sensor processors (VGG19, Conv-VAE-Neo, VAE-GAN)

Status: implemented; see "Phase 2 implementation record". Phase 2 renames
the Phase 1 generated runs: `_flow_sp_resnet50_256`, `_flow_trec_cnn_mlp`,
and `_flow_verify_cnn_mlp` become `_flow_sp_resnet50`,
`_flow_trec_resnet50`, and `_flow_verify_resnet50`.

## Goal

Train and verify the encoder + MLP controller with any of four sensor
processors, and compare controllers built on them. This is done by two
separate flows:

- a **controller flow** that trains and verifies one controller, for one
  chosen sensor processor (the Phase 1 flow, generalized); and
- a **comparison flow** that trains and verifies several controllers and
  compares them.

The supported sensor processors:

| `sp_type` | SP experiment / base run | RCCO type | SP stage notebook |
|---|---|---|---|
| `resnet50` | `sensorprocessing_propriotuned_cnn/resnet50_256` | `SP_CNN` | Train_ProprioTuned_CNN |
| `vgg19` | `sensorprocessing_propriotuned_cnn/vgg19_256` | `SP_CNN` | Train_ProprioTuned_CNN |
| `vae` | `sensorprocessing_conv_vae_neo/sp_vae_neo_256_256px` | `SP_VAE` | Train_Conv_VAE_Neo |
| `vae_gan` | `sensorprocessing_vae_gan/sp_vae_gan_256_256px` | `SP_VAE` | Train_VAE_GAN |

All four produce a 256-dimensional latent from 256×256 images, so the MLP
(`rcco_mlp_256_6`) is unchanged. Still AL5D only, still MLP only.

## What the RCCO model supports today

RCCO does **not** accept arbitrary sensor processors. An encoder enters a
controller graph through one of two component types, each tied to one
encoder architecture:

- `SP_CNN` (`rcco_sp_cnn.py`) builds its trainable and bundle-loadable
  module from `ProprioTunedCNNSensorProcessing.encoder_classes`, i.e. the
  proprioception-tuned VGG19 and ResNet-50 encoders.
- `SP_VAE` (`rcco_sp_vae.py`) builds `ConvVAENeoEncoder`, the encoder half
  of `ConvVAENeo`. Both Conv-VAE-Neo and VAE-GAN qualify, because
  `VAEGANSensorProcessing` wraps a `ConvVAENeo`
  (`sensorprocessing/sp_vae_gan.py`), and `load_vae_state_dict` takes only
  the `encoder.*` and `fc_mu.*` weights from either checkpoint.

With `load_state=True` both components call `create_sp(exp_sp).enc`, so a
frozen runtime controller could be built around other sensor processors.
The training and bundle path (`load_state=False`), however, only builds
the encoders above. Other sensor processors (legacy
`sensorprocessing_conv_vae`, ViT, random projection, ArUco, multi-view)
cannot be trained into, or loaded from, a controller bundle.

The training models add a second restriction: the encoder type fixes the
rest of the graph.

- `CNNMLPTrainingModel` requires exactly `SP_CNN.z -> MLP.z`.
- `RobotControllerTrainingModel` requires exactly
  `SP_VAE.z -> LSTM.z -> LSTM.h -> MDN.h`.

Consequences for this flow:

- **VGG19** works today. It is an `SP_CNN` encoder, and only the flow's
  generator needs to choose `vgg19_256`.
- **Conv-VAE-Neo and VAE-GAN** with an MLP cannot be trained, because no
  training model accepts `SP_VAE.z -> MLP.z`. The recipe is already
  ready for them: `_source_path` takes the encoder weights from
  `model_file(sensor_exp)` for both `SP_CNN` and `SP_VAE`, and
  `_materialize_sources` loads `SP_VAE` sources with `full_vae=True`.

## Code change: one encoder + MLP training model

Generalize `CNNMLPTrainingModel` (`cnn_mlp_training_model.py`) to accept
exactly one encoder component of type `SP_CNN` **or** `SP_VAE`, plus
exactly one `MLP`, connected `encoder.z -> MLP.z`:

- Find the encoder as the single component whose type is in
  `{"SP_CNN", "SP_VAE"}`. Build it with the matching component class and
  `load_state=False, sensor_exp=...` (`RCCO_SP_CNN` or `RCCO_SP_VAE`), and
  use its `.model` as the trainable module.
- `load_component_state(label, payload, full_vae=False)`: for the encoder
  with `full_vae=True`, call `module.load_vae_state_dict(payload)`
  (`ConvVAENeoEncoder` has it); otherwise use `load_state_dict(strict=True)`.
  This replaces today's "does not accept VAE checkpoints" error.
- `forward` is unchanged: both encoder modules map `[B, 3, H, W]` to
  `[B, latent]`.
- The architecture signature comes from the component, as today.

Rename the class `EncoderMLPTrainingModel` and the recipe
`StagedEncoderMLPTrainingRecipe`, updating `create_training_recipe`,
`trec_cnn_mlp_sample.yaml`, and `test_cnn_mlp_controller.py`. The component
label `cnn_encoder` in `roco_cnn_mlp_sample.yaml` and the recipe stages is
kept, so the existing sample and the flow keep working; a VAE encoder is
merely placed under that label.

`GraphRobotController` needs no change: it builds bundle components with
`load_state=False`, which already works for both encoder types.

New test in `test_cnn_mlp_controller.py`: a tiny `ConvVAENeo` sensor
experiment as `SP_VAE` + MLP. Train one epoch from a configured full-VAE
checkpoint, export the bundle, load it through `GraphRobotController`, and
check that the bundle output equals the training model's output. This
mirrors the existing CNN round trip.

## Shared generators: `robot_controller/rcco_flow.py`

Both flows generate the same exp/runs for a controller, so the generator
functions move from the Phase 1 flow notebook into a small module, instead
of being copied into two notebooks:

```python
SP_TYPES = {
    "resnet50": {"experiment": "sensorprocessing_propriotuned_cnn",
                 "base_run": "resnet50_256", "rcco_type": "SP_CNN"},
    "vgg19":    {"experiment": "sensorprocessing_propriotuned_cnn",
                 "base_run": "vgg19_256", "rcco_type": "SP_CNN"},
    "vae":      {"experiment": "sensorprocessing_conv_vae_neo",
                 "base_run": "sp_vae_neo_256_256px", "rcco_type": "SP_VAE"},
    "vae_gan":  {"experiment": "sensorprocessing_vae_gan",
                 "base_run": "sp_vae_gan_256_256px", "rcco_type": "SP_VAE"},
}

def generate_controller_stages(sp_type, data, epochs_sp, epochs_warmup,
                               epochs_end_to_end, creation_style):
    """Write the SP, controller, recipe, and verify exp/runs for one
    controller into the active exprun path, and return the flow entries
    [TrainSP, TrainRCCO, VerifyRCCO]."""

def generate_compare(sp_types, name, creation_style):
    """Write the compare exp/run over the verify runs of sp_types and
    return its flow entry."""
```

`data` is the dict of group entries (`sp_training`, ..., `bc_testing`)
built in the flow notebook after `import_demopack`. The run names are
derived from `sp_type`:

| Exp/run | Name |
|---|---|
| SP run (in the SP family from `SP_TYPES`) | `_flow_sp_<sp_type>` |
| RCCO encoder component | `robot_controller/_flow_rcco_sp_<sp_type>` |
| Controller graph | `robot_controller/_flow_roco_<sp_type>` |
| Training recipe | `robot_controller_training/_flow_trec_<sp_type>` |
| Verification | `robot_controller_verify/_flow_verify_<sp_type>` |
| Comparison | `robot_controller_compare/_flow_compare` |

The function bodies are those of the Phase 1 generators (`load_run`,
`save_run`, `generate_sp_cnn`, `generate_controller`, `generate_trec`,
`generate_verify`), parameterized by `sp_type`. The SP stage entry uses
`input-to-notebook[0]` of the SP family, so each encoder runs its own
training notebook (Train_ProprioTuned_CNN, Train_Conv_VAE_Neo, or
Train_VAE_GAN). All three families take `epochs`, `training_data`, and
`validation_data`, and their stage notebooks have the standard parameters.

## The controller flow: `Flow_RCCO_BehaviorCloning.ipynb`

The Phase 1 flow, with one controller chosen by a parameter:

```python
flow_name = "RCCO-BC-automove"
demopack_name = "automove-pack-01"
demonstration_cam = "dev2"
sp_type = "resnet50"   # resnet50, vgg19, vae, vae_gan
epochs_sp = 100
epochs_warmup = 20
epochs_end_to_end = 20
creation_style = "exist-ok"
```

- Queue: `generate_controller_stages(sp_type, ...)`, that is,
  `TrainSP -> TrainRCCO -> VerifyRCCO`.
- Final results: `robot_controller_verify/_flow_verify_<sp_type>`, with its
  prediction and error plots previewed.
- `setup_flow` copies the SP family of the chosen `sp_type` (from
  `SP_TYPES`) besides the robot-controller families.
- Its generator cells are replaced by the import from `rcco_flow`.

## The comparison flow: `Flow_RCCO_Compare.ipynb` (new)

Trains and verifies one controller per entry of `sp_types` and compares
them:

```python
flow_name = "RCCO-Compare-automove"
demopack_name = "automove-pack-01"
demonstration_cam = "dev2"
sp_types = ["resnet50", "vgg19", "vae", "vae_gan"]
epochs_sp = {"resnet50": 100, "vgg19": 100, "vae": 300, "vae_gan": 300}
epochs_warmup = 20
epochs_end_to_end = 20
creation_style = "exist-ok"
```

- Setup and demopack import are the same as in the controller flow. All
  four controllers use the same demonstration split, so they are compared
  on the same `bc_testing` demonstrations.
- Queue: for each `sp_type`, `generate_controller_stages(sp_type,
  epochs_sp=epochs_sp[sp_type], ...)`; then
  `generate_compare(sp_types, ...)`. With all four types, 13 stages.
- Final results: `robot_controller_compare/_flow_compare`, with the
  comparison chart previewed.
- Each controller is trained in this flow's own workspace. A controller
  already trained by a controller flow is not reused (see "Future work").

## New stage: Compare_RCCO

- New experiment family `robot_controller_compare`, whose `_defaults`
  declares `input-to-notebook: [robot_controller/Compare_RCCO.ipynb]`.
- The compare run lists the verify runs to compare:

  ```yaml
  name: "Encoder comparison on automove-pack-01"
  verify_experiment: robot_controller_verify
  verify_runs: [_flow_verify_resnet50, _flow_verify_vgg19,
                _flow_verify_vae, _flow_verify_vae_gan]
  ```

- `Compare_RCCO.ipynb` (standard parameters, ends with `exp.done()`) reads
  each verify run's `errors.csv` (its `all` row) and writes:
  - `comparison.csv`: one row per verify run, one column per AL5D field
    plus the mean;
  - `comparison.pdf` and `.png`: a grouped bar chart of per-field MSE by
    encoder.

  It recomputes nothing. Verify_RCCO stays the single source of the
  numbers.

## Tests and verification

- `test_cnn_mlp_controller.py`: the existing CNN round trip, with the
  renamed classes, and the new SP_VAE round trip.
- New `src/test/robot_controller/test_rcco_flow.py`: run
  `generate_controller_stages` and `generate_compare` against a temporary
  exprun directory holding copies of the needed families. Check that the
  entries come in stage order with the SP family's training notebook, that
  every generated exp/run resolves with `Config().get_experiment`, that the
  generated controller passes `load_controller_spec`, and that run names
  of different `sp_type`s do not collide.
- `test_exprun_notebooks.py`: add `Flow_RCCO_Compare.ipynb` to
  `FLOW_NOTEBOOKS`; Compare_RCCO is covered through its `input-to-notebook`
  declaration.
- Run the controller flow once per `sp_type` with debug epochs (SP 2,
  MLP 1 / 1), then the comparison flow with all four: 13 stages complete,
  and the comparison chart shows four encoders.

## Phase 2 implementation record

Implemented as designed, in:

- `src/robot_controller/encoder_mlp_training_model.py` (renamed from
  `cnn_mlp_training_model.py`), `EncoderMLPTrainingModel`;
  `StagedEncoderMLPTrainingRecipe` in `training_recipe.py`;
  `trec_cnn_mlp_sample.yaml` uses the new recipe class;
- `src/robot_controller/rcco_flow.py`, with `flow_families(sp_types)`
  added: the families `setup_flow` must copy for the chosen encoders;
- `Flow_RCCO_BehaviorCloning.ipynb` (`sp_type`), the new
  `Flow_RCCO_Compare.ipynb`, and `Compare_RCCO.ipynb`;
- `data/expruns/robot_controller_compare/` (`_defaults_...`, and
  `compare_sample.yaml` over `verify_cnn_mlp_sample`, for running
  Compare_RCCO outside a flow);
- tests: `test_cnn_mlp_controller.py` (renamed classes; new SP_VAE + MLP
  round trip from a full ConvVAENeo checkpoint),
  `test_rcco_flow.py`, and `FLOW_NOTEBOOKS` in `test_exprun_notebooks.py`;
- docs: `DESIGN-Flows.md`, `DESIGN-RobotController.md`,
  `data/expruns/robot_controller/DESIGN_Training_Recipe.md`.

Fix outside the design: `Config.set_exprun_path` and
`Config.set_results_path` (`exp_run_config.py`) now accept strings.
`Train_VAE_GAN.ipynb` passed the papermill path parameters (strings)
unconverted and failed when first run as a flow stage.

Results on `automove-pack-01` with debug epochs (SP 2, MLP 1 / 1):

- the comparison flow completed all 13 stages. Teacher-forced test MSE,
  averaged over the six normalized fields: ResNet-50 0.064, VGG19 0.050,
  Conv-VAE-Neo 0.067, VAE-GAN 0.065. These numbers only show that the
  pipeline works; at two epochs they say nothing about the encoders;
- the controller flow with `sp_type = "vae"` completed its three stages
  (TrainSP 3 min 16 s, TrainRCCO 22 s, VerifyRCCO 9 s).

Runs with the default epochs have not been done.

## Out of scope for Phase 2 (possible Phase 3)

- **Arbitrary frozen sensor processors.** A generic component (e.g.
  `SP_Frozen`) could wrap `create_sp(exp_sp).process()` for any SP,
  including ViT, random projection, and legacy Conv-VAE. It would be
  non-trainable, and its latents could be precomputed and cached by the
  training model, which also solves the encoder-cost limitation. The bundle
  would record the SP exp/run instead of weights.
- **Encoders with LSTM/MDN heads** (CNN -> LSTM -> MDN): this needs the
  graph-to-training-model generalization, not just an encoder choice.
- **Multi-view encoders.** The dataset and `RobotControllerSequenceDataset`
  entries carry one camera.

## Future work: sharing pre-trained components between flows

The controller flow and the comparison flow train everything in their own
workspace. A comparison over four encoders therefore retrains every encoder
and controller, even when a controller flow has already trained the same
one, and two comparisons that share an encoder each train it again. Within
one comparison, controllers that differ only in their head (for example a
future MLP vs. LSTM-MDN comparison on the same encoder) could share one
trained SP run, but the naming above gives every controller its own.

It would be useful to let a flow reuse a pre-trained component (an SP run,
a controller bundle, or a verify run) instead of training it. Points a
design would have to settle:

- **Identity.** When two components are the same: the same exp/run
  values (with `training_data` / `validation_data` naming the same
  demopack demonstrations) and the same code. The training recipe's
  fingerprint is a starting point.
- **Location.** Either a shared results area under `flows_path` that all
  flows read and write, or a parameter naming another flow workspace to
  import from (copying or linking its `results/<exp>/<run>` directories
  and their exp/runs).
- **Demonstration names.** `import_demopack` renames demonstrations by
  group (`bc_training_00000`, ...), so the same split must be used for a
  reused component to mean the same data.
- **Provenance.** The flow report should mark reused stages, and link to
  the workspace that produced them, rather than show them as trained by
  this flow.
- **Staleness.** A reused component trained by an older code version
  should be detectable, for example through an architecture or recipe
  version recorded in its `exprun.yaml`.

