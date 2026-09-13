# Proposed composite sensor-processing models

These proposals adapt literature-supported representation patterns to
BerryPicker's composite sensor-processing architecture. They are candidate
experiments, not architectures already demonstrated to work on BerryPicker.
See [DESIGN-CompositeSensorProcessing.md](DESIGN-CompositeSensorProcessing.md)
for execution, training, and checkpoint ownership.

Each composition is one SP exp/run with a final representation of size
`exp["latent_size"]`. Concrete operations and adapters must be implemented
separately; this document does not introduce implementations or runnable runs.

The diagrams below describe inference compositions, not mandatory end-to-end
training graphs. Each family supplies its own training procedure. Components
may be trained independently with different data and objectives, then assembled
and optionally fine-tuned. The simple training suggestions below are starting
experiments, not a common training algorithm imposed on every composite.

## Objectives and priorities

Predicting robot joint positions and learning a representation for manipulation
are different objectives. A proprioception loss can reward encoding the arm
while ignoring the target object. Object-focused compositions should also be
evaluated through behavior cloning or supervised object-location tasks.

| Composition | Main purpose | Priority |
|---|---|---|
| Pretrained encoder followed by projection | Establish a strong, inexpensive baseline | First |
| Appearance plus spatial features | Preserve visual identity and precise location | High |
| RGB plus segmentation guidance | Reduce distraction while retaining scene context | High |
| Multi-camera feature fusion | Resolve occlusion and viewpoint ambiguity | High |
| Global scene plus local crop | Preserve small-object detail | Medium |
| Task-specific plus general pretrained features | Combine existing SP models with transferable features | Medium |

## 1. Pretrained encoder followed by learned projection

```text
Image -> frozen R3M or DINOv2 -> learned projection -> z
```

This is the simplest useful composition and an important control experiment.
R3M demonstrated frozen visual representations for downstream robot
manipulation. DINOv2 provides transferable image-level and dense visual
features, although its paper does not establish superiority for BerryPicker's
control task. [R3M](https://arxiv.org/abs/2203.12601),
[DINOv2](https://arxiv.org/abs/2304.07193).

Freeze the backbone initially and train a small projection plus the
proprioception or behavior-cloning head. Compare with existing CNN and VAE
models at the same final latent size.

For precise position estimation, also test spatial pooling of patch features
instead of relying exclusively on a global image embedding. The pooling rule
and resulting feature dimensions should be explicit steps/settings.

## 2. Appearance features plus explicit spatial features

```text
                           +-> global pooling -> appearance vector --+
Image -> spatial backbone -+                                         +-> concat -> projection -> z
                           +-> heatmaps -> spatial softmax ----------+
```

The spatial branch converts feature heatmaps into expected image coordinates.
The appearance branch retains information that coordinates alone cannot express.

Deep Spatial Autoencoders demonstrated compact feature-point representations
for visuomotor learning. This supports the spatial branch; combining it with a
separate appearance branch is a proposed adaptation.
[Deep Spatial Autoencoders for Visuomotor Learning](https://arxiv.org/abs/1509.06113).

For BerryPicker, this is attractive for locating the gripper, arm joints, and
objects while retaining appearance information. Learned feature points need
not correspond to named landmarks unless explicitly supervised.

Train the spatial branch and fusion with the downstream objective, initially
keeping any pretrained backbone frozen. Reconstruction or landmark supervision
can provide auxiliary objectives when the relevant data exists.

Use concatenation because appearance descriptors and coordinates have different
meanings. Normalize coordinates consistently before fusion. Evaluate whether
the spatial branch improves localization beyond the appearance branch alone.

## 3. RGB plus segmentation guidance

```text
Image + target prompt -> segmenter -> target mask --+
                                                   +-> channel concat -> encoder -> z
Image ---------------------------------------------+
```

An RGB image and one target mask produce a four-channel input. Multiple
explicitly identified masks, such as gripper and target, can occupy separate
channels.

RoboGround combines grounding masks with image inputs and uses mask-guided
attention for manipulation. Its masks identify target objects and placement
areas; this does not establish that arbitrary foreground segmentation is
sufficient.
[RoboGround, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_RoboGround_Robotic_Manipulation_with_Grounded_Vision-Language_Priors_CVPR_2025_paper.html).

SAM is a possible mask-producing component, with explicit prompt generation and
mask selection. [Segment Anything](https://arxiv.org/abs/2304.02643).

The first variant should retain the full RGB image and append the mask. Hard
background removal might discard arm or environmental information needed by
the task.

Initially freeze segmentation and train the downstream encoder. Four-channel
input requires adapting and training the encoder's input layer; it cannot be
passed unchanged to a pretrained RGB backbone. A separate mask encoder followed
by feature concatenation is an alternative that preserves the RGB interface.

Cache fixed segmentation results for training under the composite design's
cache rules. Evaluate segmentation guidance against RGB alone, and distinguish
mask quality from downstream representation quality.

## 4. Multi-camera encoders followed by learned fusion

```text
Camera 1 -> encoder -> projection --+
                                   +-> concatenate / gate / attention -> z
Camera 2 -> encoder -> projection --+
```

This extends the multiview models already in BerryPicker. RVT demonstrates
attention across views for robotic manipulation. RVT also reconstructs and
renders virtual views of the workspace; two-camera RGB fusion is a simpler
adaptation, not a reproduction of RVT.
[RVT](https://arxiv.org/abs/2306.14896).

Compare three fusion variants with fixed encoders and final latent size:

- `concat_proj`: baseline concatenation and projection.
- `gated`: input-dependent contributions from each camera.
- `attention`: interactions between camera representations.

A gate might learn to rely less on an occluded view, but that behavior must be
measured. Learned weights are not automatically calibrated confidence estimates.

Begin with frozen encoders and train fusion. Fine-tune encoders after
establishing that fusion improves held-out performance. Preserve the trained
camera order everywhere. Compare both individual cameras against their fusion.

Preserving multiple spatial tokens per camera is a later extension. The
existing fusion heads operate on one feature vector per view.

## 5. Global scene plus local crop and crop coordinates

```text
Image -> scene encoder ----------------------------------+
Image -> detector -> crop -> local encoder ---------------+-> concat -> projection -> z
                   +-> normalized crop coordinates ------+
```

This addresses cases where resizing the full camera image makes the gripper,
target, or contact region too small.

RVT-2 uses coarse-to-fine processing to improve precise manipulation. A
global/local crop composition borrows that principle without adopting its
complete 3D policy architecture.
[RVT-2](https://arxiv.org/abs/2406.08545).

Crop coordinates are essential: resizing each crop to the same dimensions
otherwise removes its original position and scale. Include normalized box
coordinates alongside the local features. The global branch supplies scene
context.

Initially use a fixed detector and train the local encoder and fusion. A fixed
workspace crop can first test whether extra spatial resolution helps before
introducing learned localization.

This is especially interesting if errors concentrate near grasping or contact
rather than during large arm movements. Compare global-only, local-only, and
combined representations; retain the same train/test crop-generation procedure.

## 6. Existing task-specific encoder plus general pretrained encoder

```text
Image -> existing proprioception-tuned encoder --+
                                                +-> branch projections -> concat -> fusion -> z
Image -> frozen DINOv2 or R3M encoder ------------+
```

The hypothesis is that the existing encoder captures robot configuration while
the pretrained branch contributes object and scene information.

The literature supports transferable pretrained visual features, particularly
R3M for manipulation. This exact pairing is an experimental hypothesis rather
than a demonstrated result. [R3M](https://arxiv.org/abs/2203.12601),
[DINOv2](https://arxiv.org/abs/2304.07193).

Freeze both branches initially and train fusion. Compare against each branch
individually at a controlled final latent size; otherwise improvement could
simply reflect a larger representation.

Avoid direct addition of independently trained latent vectors. Matching lengths
do not establish aligned coordinates. Branch projections and concatenation
provide an explicit way to learn the combination.

## Family-specific training procedures

The following are proposed recipes, not claims that the cited papers use these
exact BerryPicker training schedules.

| Composition | Separate preparation/training | Composite-owned training stages |
|---|---|---|
| Pretrained encoder + projection | Select a fixed pretrained checkpoint; optionally adapt the encoder in a separate source run | Train projection/task head; optionally fine-tune selected layers |
| Appearance + spatial features | Optionally pretrain spatial features using reconstruction or landmark supervision, separately from the appearance encoder | Train branch projections and fusion; optionally optimize spatial features with task and auxiliary objectives |
| RGB + segmentation | Train/adapt and select the segmenter on segmentation data; fix prompt and mask-selection procedures | Train the mask-conditioned encoder on task demonstrations; optionally fine-tune segmentation only with a suitable differentiable path and supervision |
| Multi-camera fusion | Train per-view encoders separately, or train a joint multiview VAE with reconstruction/KL losses | Assemble encoding paths, train fusion/head, then optionally fine-tune selected encoders |
| Global + local crop | Train/select the detector; optionally pretrain the local encoder on crops generated by the selected detector | Train global/local fusion; optionally fine-tune the local branch while keeping detection fixed |
| Task-specific + general features | Train the proprioception encoder in its own run and independently select the general pretrained model | Train projections/fusion with branches frozen; optionally fine-tune selected branches using a task-appropriate objective |

For appearance/spatial variants, decide explicitly whether branches share a
backbone. Independently trained backbone weights cannot both initialize one
shared module; either retain separate modules or choose one initialization and
train the shared representation with the relevant objectives.

A multiview VAE is a useful example of why these procedures differ. It can be
trained separately with its decoder and KL term, then incorporated through an
adapter calling `encode_views()`. Training only fusion needs neither the decoder
nor a VAE loss. Continuing VAE training inside a later stage needs a dedicated
wrapper exposing reconstruction, mean, and log variance. The current composite
executor supports tensor composition but supplies neither that adapter nor the
training recipe.

Each family has an explicit training entry point configured by its exp/run.
Reusable components have their own exp/runs; composite-specific pretraining may
instead use stage directories under the composite run. Building/loading an SP
must never implicitly launch prerequisite training. Source initialization hooks
import selected checkpoints only.

Use separate checkpoints and optimizer state for stages with different modules
or objectives, and record progress and selected artifacts for resumption.
Freezing changes must update the executor's persistent frozen-module policy as
well as gradients and optimizer groups; no automatic stage transitions are
provided by the current core. A recipe may also alternate updates or objectives
when a simple sequential schedule is insufficient.

Cache outputs only while their entire upstream computation is fixed. Switching
the segmenter, detector, crop procedure, or encoder checkpoint invalidates
dependent caches. Once upstream components are fine-tuned, train from the
appropriate raw inputs rather than stale cached representations. Cache identity
includes selected checkpoints, prompts, and preprocessing settings.

One final SP still belongs to one composite exp/run. Export its inference state
to `model_file(exp)` and retain the resolved architecture; keep full training
wrappers, decoders, optimizers, and stage progress in separate training artifacts.
Deployment restores the final composite without replaying its training history.

## Initial experiment set

Start with the pretrained baseline, appearance-plus-spatial features,
mask-guided RGB, and two-camera fusion. These test distinct hypotheses: better
pretraining, explicit location, segmentation guidance, and additional viewpoints.

Keep final latent size, downstream heads, and demonstration splits comparable.
Split by complete demonstrations rather than adjacent frames. Check whether
pretraining/source runs overlap evaluation demonstrations, and report any such
overlap when interpreting generalization.

Measure:

- Robot-position error for the visual-proprioception objective.
- Downstream manipulation or behavior-cloning performance for object-aware
  representations.
- Inference latency, including segmentation or detection rather than only the
  final encoder.
- Performance under relevant changes in lighting, clutter, and occlusion.

Use branch ablations to identify where improvements originate. Keep the
initial fusion simple, normally concatenation followed by projection, and test
more complex fusion only after establishing complementary branch information.

All trainable projection/fusion weights belong to the composite exp/run.
Pretrained sources follow the architectural design's explicit distinction
between composite-owned weights and frozen external dependencies. Task heads
belong to training; the deployed SP still returns only `z`.
