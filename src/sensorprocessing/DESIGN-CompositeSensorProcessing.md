# Composite sensor processing

## Purpose and scope

Introduce a sensor-processing model composed of an ordered series of named
steps. Steps can filter images, perform segmentation, encode observations, and
combine representations. A step can consume any earlier result, allowing
branches and merges without a graph scheduler.

The architectural core is implemented in `composite.py` and `sp_composite.py`:
ordered execution, operation lifecycle registration, freezing, resolved
configuration snapshots, and inference wrappers. Concrete processing operations,
external adapters, caches, and task training wrappers remain proposed.
Configuration examples below include proposed operations and illustrative runs.
See the README for the implemented registration and checkpoint interfaces.

The composite remains one SP model described by one experiment/run obtained
through `Config().get_experiment()`. Its model files belong to that experiment's
data directory. Steps are internal components, not necessarily separate runs.

## Existing architecture

- `sensor_processing.py` separates the public inference interface from tensor
  encoders. `EncoderSensorProcessing.process()` enters evaluation mode, disables
  gradients, and converts the encoding to NumPy.
- `sp_helper.py` prepares RGB image tensors in `[0, 1]`; model-specific
  normalization belongs to the encoder.
- `sp_factory.py` constructs processors from an experiment/run.
- `multiview_fusion.py` already supplies `concat_proj`, `indiv_proj`, `attention`,
  `weighted_sum`, and `gated` fusion heads.
- The Neo multiview fusion model provides an existing example of initializing
  encoders from another run and training fusion with a proprioception objective.

Composition belongs inside the tensor-model layer. Calling an existing SP's
`process()` between steps would disable gradients and repeatedly convert tensors
to NumPy. Encoder adapters instead call the underlying tensor encoding method,
respecting the processor's `encoder_method`.

## Execution model

`CompositeModel` is a `torch.nn.Module` containing operations in an
`nn.ModuleDict`. The configuration gives each step a unique name, an operation,
and an ordered list of inputs. Inputs refer to the initial observation or to
earlier step outputs. Execution follows configuration order.

```text
Image -> filter ---------------------> appearance encoder ---+
           |                                                |
           +-> segmentation -> mask                         +-> fusion -> z
           |                    |                           |
           +---------------> apply mask -> foreground encoder
```

The executor is conceptually:

```python
def forward_steps(self, sensor_readings):
    values = {"input": sensor_readings}
    for step in self.steps:
        inputs = [resolve(values, name) for name in step["inputs"]]
        values[step["name"]] = self.operations[step["name"]](*inputs)
    return values

def encode(self, sensor_readings):
    values = self.forward_steps(sensor_readings)
    return resolve(values, self.output)
```

`resolve` supports a step name and, for a structured result, a named field such
as `segmentation.masks`. No expression language, automatic scheduling, or
implicit conversion is needed. Intermediate results exist for one invocation;
the executor does not retain computation graphs between calls.

Use ordinary modules implementing `forward(*inputs)`. Most operations return a
tensor. Operations such as prompted segmentation may return a small dictionary
of named results, for example `masks`, `scores`, and `boxes`. Selection and
conversion remain explicit steps.

Assume experiment fields are present and correctly formatted. Use direct field
access such as `exp["steps"]`. Missing references, unavailable dependencies,
and incompatible operations raise exceptions. Do not add silent substitutions,
fallback representations, or a general configuration-validation framework.

## Representation contracts

| Representation | Shape or contract |
|---|---|
| RGB image or spatial features | `[B, C, H, W]` |
| Class probabilities | `[B, K, H, W]` |
| Foreground mask | `[B, 1, H, W]` |
| Feature vector | `[B, D]` |
| Token sequence | `[B, N, D]` |
| Multiple segmentation candidates | Named masks and scores; adapter documents candidate and batch dimensions |
| Final representation | `[B, exp["latent_size"]]` |

Resizing, flattening, pooling, projection, mask selection, and conversion from
logits to probabilities are explicit operations. Addition requires matching
shapes; matching lengths alone do not imply compatible feature meanings.
Mask application deliberately broadcasts one mask channel across image channels.

`CompositeSensorProcessing` is the public single-view inference wrapper. It
exposes the same `process()` and `process_file()` behavior as existing encoders,
including the NumPy inference result. Training calls the tensor model directly.

`CompositeMultiViewSensorProcessing` uses the existing multiview base and
receives the ordered list of camera tensors as `input`. `select_view` operations
expose individual views. Preserve `exp["cameras"]` order for demonstrations,
captures, training, and inference.

## Configuration example

```yaml
class: CompositeSensorProcessing
model_file: composite.pth
image_size: [256, 256]
latent_size: 256

steps:
  - name: filtered
    operation: gaussian_blur
    inputs: [input]
    kernel_size: 5
    sigma: 1.0

  - name: probabilities
    operation: segmentation
    inputs: [filtered]
    source_experiment: sensorprocessing_segmentation
    source_run: foreground_model
    frozen: true

  - name: mask
    operation: select_class
    inputs: [probabilities]
    class_index: 1

  - name: foreground
    operation: apply_mask
    inputs: [filtered, mask]

  - name: appearance
    operation: encoder
    inputs: [filtered]
    source_experiment: sensorprocessing_conv_vae_neo
    source_run: sp_vae_neo_128_256px_dev2_dev3
    frozen: true

  - name: foreground_features
    operation: encoder
    inputs: [foreground]
    source_experiment: sensorprocessing_conv_vae_neo
    source_run: sp_vae_neo_128_256px_dev2_dev3
    frozen: true

  - name: combined
    operation: concat
    inputs: [appearance, foreground_features]
    dim: 1

output: combined
```

Two 128-dimensional encodings produce a 256-dimensional representation. Each
encoder step above owns a separate module initialized from the same source.
Weight sharing, if later needed, should be an explicit module reference rather
than inferred from equal source runs.

Steps may also declare an architecture directly and initialize it from scratch;
they need not refer to an existing SP run. Applying a pretrained encoder to
masked images changes its input distribution and needs empirical evaluation.

## Fusion operations

| Operation | Behavior | Constraints |
|---|---|---|
| `concat` | Join along a configured dimension | Vector widths may differ |
| `sum`, `mean` | Elementwise sum or average | Matching shape and aligned feature semantics |
| `multiply` | Elementwise interaction | Matching shapes, or deliberate mask broadcasting |
| `stack` | Add a branch/view/token dimension | Matching input shapes |
| `concat_proj` | Concatenation followed by learned projection | Reuse existing head |
| `indiv_proj` | Per-branch projections followed by learned fusion | Reuse existing head |
| `weighted_sum` | Project branches and learn input-independent softmax weights | Reuse existing head |
| `gated` | Project branches and learn input-dependent weights | Reuse existing head |
| `attention` | Interact across branch tokens, pool, and project | Reuse existing head |

The existing `MultiViewFusion` accepts feature tensors regardless of whether
they originate from cameras or different representations of one image. Its
interface requires a common feature width. Explicit projection steps can align
heterogeneous widths before this adapter; plain concatenation requires no such
alignment. Preserve input order, including the meaning of learned view/branch
embeddings.

Concatenation is the default baseline for independently trained encoders.
Their latent coordinates need not align even when their dimensions agree.
Learned projections provide a route to additive fusion.

Possible later operations include:

- FiLM: one representation generates feature-wise scale and offset for another,
  `gamma(condition) * features + beta(condition)`.
- Bilinear/tensor fusion: multiplicative interactions across feature coordinates,
  preferably factorized when full tensor products would be too large.

References for these families, not claims of superiority on BerryPicker:

- [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992).
  The existing gate is related, not necessarily this exact architecture.
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871).
- [Tensor Fusion Network for Multimodal Sentiment Analysis](https://aclanthology.org/D17-1115/).
- [Efficient Low-rank Multimodal Fusion With Modality-Specific Factors](https://aclanthology.org/P18-1209/).

## Preprocessing and training

Keep file loading, RGB conversion, common geometry, batching, and device
placement in `SensorPreprocessor`. Execute configured filtering and segmentation
inside the composite so training and deployment apply the same operations.
Branch-specific geometry is explicit; model-specific normalization stays with
the branch adapter/encoder. External segmentation adapters must map outputs and
prompts between their model's coordinates and the pipeline input coordinates.

The processing graph defines inference, not a training algorithm. Different
composite SP families are expected to have different, potentially complex
training procedures. One may only assemble independently pretrained components;
another may alternate several objectives, train branches separately, then
fine-tune a subset jointly. A single end-to-end loss is one option, not the
default contract of the architecture.

Each family supplies its own training entry point (Python function/script or
notebook), configured by its exp/run. It owns datasets, training-only modules,
objectives, optimizers, stage transitions, and model selection. Reuse the
existing harness for individual compatible stages. Custom epoch callbacks or
loops handle procedures that need different optimization schedules. The
composite executor and `create_sp()` do not orchestrate training.

A simple task-specific training wrapper can contain the composite and a
prediction head. For the existing proprioception objective:

```text
Images -> composite -> z -> proprioception head -> predicted position
                                                   |
                                        loss against recorded position
```

At deployment, only the graph through `z` is used. Reuse the existing training
harness and task loss rather than introducing a general loss language.

```python
z = model.composite.encode(images)
prediction = model.proprioception_head(z)
loss = criterion(prediction, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The optimizer includes only parameters with `requires_grad=True`.

### Separate and staged component training

A representative procedure is:

```text
Segmentation data -> train segmenter ---------> selected segmenter checkpoint --+
Unlabeled views ---> train multiview VAE ------> selected VAE checkpoint --------+-> assemble composite
                                                                                       |
Task demonstrations -------------------------------------------------------> train fusion/head
                                                                                       |
                                                                     optional selective fine-tuning
                                                                                       |
                                                                            export inference SP
```

Stages can use different datasets, supervision, input representations, and
training-only modules. For example, train a multiview VAE with its reconstruction
and KL losses before importing its `encode_views()` path into a composite.
Its encoding adapter alone does not expose the decoder or VAE training outputs.
A later task stage may freeze this encoder, fine-tune it with a task loss, or
use a custom wrapper retaining VAE outputs for auxiliary supervision.

Resolve source exp/runs with `Config().get_experiment()`. Independently reusable
components should be trained in their own exp/runs, with their own datasets,
selected checkpoints, and training entry points. Composite-specific branch
pretraining may instead be a stage owned by the composite run. There is no
requirement for every internal step to have a separate exp/run.

Source training is explicit work by those training entry points, never an
implicit side effect of building a composite or discovering a missing checkpoint.
The operation `resolve` and `initialize` hooks resolve configuration and import
already selected weights; they are not component-training hooks. Pure assembly
of pretrained components and parameter-free fusion may require no composite
optimization at all.

The recipe records stage-specific trainable/frozen modules, losses and weights,
optimizers, stopping/selection criteria, selected input checkpoints, and cache
dependencies. Prefer ordinary family-specific code and exp fields over a generic
training-stage language or scheduler. The inference graph remains stable even
when the training procedure is substantially different from another family's.

The current executor's `frozen` list is fixed at construction. A later stage
that changes freezing must explicitly configure both parameter gradients and
the persistent evaluation-mode policy; changing `requires_grad` alone does not
change that list. The family training code must also rebuild/configure its
optimizers for the new parameter groups. Stage scheduling and automatic
freeze/unfreeze transitions are not implemented by the architectural core.

| Training arrangement | Trainable parts |
|---|---|
| Frozen feature extraction | Fusion and prediction head |
| Partial fine-tuning | Selected encoders, fusion, and prediction head |
| Joint training | All selected differentiable steps and prediction head |

If the graph consists entirely of frozen encoders and parameter-free fusion,
only the task head learns; the representation itself does not change.

For differentiable modules, `frozen: true` disables parameter gradients and
keeps the module in evaluation mode even when the containing model enters
training mode. It does not automatically apply `torch.no_grad()`: a frozen
module may need to transmit gradients to a trainable upstream step.

Inference-only adapters explicitly form a gradient boundary. API operations,
OpenCV operations, and hard mask decisions cannot be trained through ordinary
PyTorch backpropagation. Soft mask multiplication can propagate gradients into
a differentiable segmenter. A downstream proprioception loss alone does not
ensure meaningful segmentation; a segmenter could learn an all-pass mask.
Freeze a pretrained segmenter or provide segmentation supervision when mask
meaning must be preserved.

`forward_steps()` makes intermediate results available for auxiliary losses:

```python
outputs = model.composite.forward_steps(images)
prediction = model.proprioception_head(outputs["combined"])
loss_proprio = proprioception_loss(prediction, robot_positions)
loss_seg = segmentation_loss(outputs["segmentation_logits"], masks)
loss = loss_proprio + exp["segmentation_loss_weight"] * loss_seg
```

A VAE reconstruction/KL objective similarly requires its decoder and mean/log
variance outputs; the final encoding alone is insufficient. Those modules and
losses belong to the appropriate training wrapper or explicitly configured
steps. Auxiliary targets must exist in the training data.

Train from images when upstream components change. Outputs of fixed subgraphs
may be cached when they do not depend on trainable upstream operations or
varying augmentations. Cached downstream SP encodings must be regenerated when
the composite changes.

## Experiment/run ownership and model lifecycle

The composite exp/run owns the final graph, its training recipe/settings, data
selection for its stages, and exported SP model. One SP per exp/run means one
deployable model, not one optimizer, dataset, loss, or training session. Use
`model_file(exp)` and the existing training-harness conventions for paths and
checkpoint format. Source runs are read-only initialization sources; training
a composite never updates their files.

Keep composite-owned stage artifacts in separate subdirectories of the run's
data directory, for example `stages/branch_pretraining/` and
`stages/fusion_training/`, each with its own checkpoints. The family training
entry point supplies these paths explicitly to the harness. Independently
trained source components retain their training artifacts in their own runs.

Record the selected checkpoint identities/digests and resolved configurations,
not only a mutable source run name or a pointer to its latest checkpoint. Save
enough stage state to resume the actual procedure: completed stages, current
stage, applicable model/optimizer/scheduler state, and any additional state
required by that recipe. Do not silently rerun component pretraining on resume.
The core snapshot and model state alone do not provide multistage resumption.

Composite-owned modules are registered beneath the composite, giving state
dictionary keys such as:

```text
operations.appearance.encoder....
operations.foreground_features.encoder....
operations.projection.weight
operations.projection.bias
```

Include frozen composite-owned weights as well as trainable weights in the final
export. Stage training checkpoints retain any task heads, decoders, and other
state needed to resume that stage. Export only the final inference composite's
`state_dict()` to `model_file(exp)`; a task wrapper's prefixed state dictionary
is not directly loadable by the SP wrapper. Keep full stage checkpoints separate
from this export. The inference loader need not know how many stages ran or
which objectives trained its components.

| Lifecycle | Initialization |
|---|---|
| New training | Family recipe trains prerequisites explicitly, selects their checkpoints, assembles the composite, and runs any remaining stages |
| Resume | Family recipe restores stage progress and the appropriate saved architectures, models, and training state |
| Deployment | Construct saved architecture; restore composite state and enter inference mode |

Separate module construction from source-weight initialization. Do not call
`create_sp(source_exp)` unconditionally during resume/deployment because its
constructor may require the original source checkpoint.

At initial training, save resolved architecture settings for imported modules
alongside the model in the composite run's data directory. Resume and deployment
use that snapshot instead of rereading mutable source architecture settings.
Source experiment/run names remain provenance. Composite-owned source weights
are no longer external dependencies after saving the complete composite.

Downstream experiments still select one SP experiment/run:

```yaml
sp_experiment: sensorprocessing_composite
sp_run: appearance_foreground_256
```

```python
spexp = Config().get_experiment(exp["sp_experiment"], exp["sp_run"])
sp = create_sp(spexp)
```

## Pretrained foundation models and API steps

Not every model should be copied into every composite checkpoint. Distinguish
weight ownership from freezing:

| Ownership/execution | Saved by the composite run |
|---|---|
| Composite-owned local module | Resolved architecture and all module weights |
| Frozen external local model | Exact model identity, checkpoint reference/digest, adapter settings |
| API operation | Service/model version and request settings; no remote weights |

Make external ownership explicit in the step configuration. An externally
managed local model is inference-only in the initial design. If fine-tuning is
required, use a differentiable, composite-owned adapter instead. External
weights can live in a shared model cache and must not be accidentally included
in the composite's state dictionary through module registration. The adapter
owns the runtime handle and its device placement; only composite-owned learned
modules participate in checkpoint serialization and optimization.

The run's data directory therefore contains the composite checkpoint, resolved
configuration, and optional cached intermediate results. A composite with
external steps intentionally retains those runtime dependencies. Credentials
come from the runtime environment, never the configuration snapshot/checkpoint.

### SAM example

Treat SAM as a model-specific segmentation adapter, with explicit prompts and
output selection. Prompts may originate from a detector, fixed regions, or
annotations. The choice of prompt source is part of the graph.

```text
Image -> detector -> boxes ----+
  |                           |
  +-------------------------> SAM -> candidate masks -> select mask
  |                                                        |
  +--------------------------------------------------> apply mask -> encoder
```

Illustrative step fields:

```yaml
- name: segmentation
  operation: sam
  inputs: [input, boxes]
  execution: local
  ownership: external
  model_id: <specific-variant-and-version>
  checkpoint: <external-checkpoint-reference>
  frozen: true

- name: foreground
  operation: select_mask
  inputs: [segmentation.masks, segmentation.scores]
  selection: <explicit-selection-rule>

- name: masked_image
  operation: apply_mask
  inputs: [input, foreground]
```

Document prompt coordinate conventions, mask type, candidate ordering, and
batch behavior in the adapter. Selection must define which object(s) constitute
foreground. An empty valid detection result needs an explicit task-defined
policy; it is not grounds for silently returning an unrelated representation.
Different SAM variants or automatic-mask-generation modes require their own
documented adapter settings, not assumptions hidden in the generic executor.

A remote segmentation service implements the same result contract, with
service-specific request construction inside its adapter. This does not assume
SAM itself supplies a hosted API. API failures raise exceptions. Deployment
feasibility depends on measured latency and service availability.

Precompute fixed segmentation results for repeated training epochs. Cache
identity includes observation identity/content, prompts, exact model version
or checkpoint digest, and relevant step settings, including preprocessing.
Changed inputs/settings require regeneration. Cached API outputs preserve what
training used; they do not guarantee future remote outputs if the provider
cannot pin the model version.

## Minimal implementation and verification

Proposed files:

- `composite.py`: tensor model, ordered execution, and step construction.
- `processing_steps.py`: basic transforms, mask operations, encoder adapters,
  and adapters to existing fusion heads. Model-specific SAM/API adapters may
  live in separate modules and load dependencies only when selected.
- `sp_composite.py`: single-view and multiview inference wrappers.
- `sp_factory.py`: register the two wrappers and multiview classification.
- One composite experiment family and a proprioception training entry point
  using the existing harness.

Start with filtering, mask selection/application, encoder reuse, concatenation,
addition, and existing fusion heads. Initially use frozen pretrained branches
with trainable fusion and a proprioception head. Integrate frozen local SAM
with explicit prompts and training caches before adding remote execution or
foundation-model fine-tuning. FiLM and tensor fusion can follow as needed.

Verify branching and numerical fusion, tensor/NumPy inference compatibility,
gradient flow and frozen evaluation behavior, mask geometry, and checkpoint
round trips. For composite-owned steps, verify reload without source model
files. For external steps, verify references and failure propagation using
local test doubles rather than live API calls. Compare a single-encoder
composite against its original processor on identical inputs.

Consumers using `create_sp()` and `process()` retain their interface. Controller
components that explicitly construct CNN/VAE architectures need a separate
integration change to support composites. No general controller refactor or
new workflow engine is part of this proposal.
