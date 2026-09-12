# Design: Separate MultiView Fusion and MultiView Concat Conv-VAE-Neo Models

Status: implemented.

## Goal

Extend BerryPicker's internal `ConvVAENeo` model to multi-camera input with two
independent model families:

1. A fused, proprioception-trained sensor processor that encodes every camera
   separately and combines the encodings.
2. A true multiview VAE that constructs one joint observation from synchronized
   camera images and learns it with reconstruction and KL losses.

Concat and Fusion must be separate `nn.Module` classes, Python modules,
sensor-processing wrappers, experiment families, training functions, and
notebooks. Neither model should contain a `mode` or `model_type` switch that
turns it into the other. They may share data validation and checkpoint
utilities, but they must not share a model superclass beyond normal PyTorch and
existing sensor-processing abstractions.

Names consistently place the multiview strategy last: files and experiment
families use `multiview_fusion` or `multiview_concat`, while class names use
`MultiViewFusion` or `MultiViewConcat`.

The legacy external Conv-VAE models and their experiment configurations should
remain unchanged so existing results can still be reproduced and compared.

## Current situation

[`conv_vae_neo.py`](conv_vae_neo.py) provides the internal,
resolution-scalable VAE. Its deterministic sensor-processing representation is
the mean of the approximate posterior, returned by `ConvVAENeo.encode()`.

The existing multiview models have two different meanings:

- [`sp_conv_vae_multiview.py`](sp_conv_vae_multiview.py)
  uses a small convolutional encoder per view, a fusion head, and a
  proprioception regressor. Despite its name, this is supervised and has no VAE
  decoder or KL loss.
- [`sp_conv_vae_concat_multiview.py`](sp_conv_vae_concat_multiview.py)
  concatenates views for the external Conv-VAE. It is a VAE, but it depends on
  the external library and reduces concatenated images back to the original
  width.

The Neo replacement should preserve both experimental questions without
conflating their training objectives.

The current multiview infrastructure already supplies several variations that
the Fusion model can reuse directly:

| Variation | Current support | Treatment in this design |
|---|---|---|
| Horizontal image concatenation | Legacy external Conv-VAE concat wrapper | Reimplement internally for the Neo Concat model without width reduction |
| Channel concatenation | Partial runtime path only; no compatible end-to-end VAE training | Exclude from the initial Neo models |
| Shared or separate view encoders | `ViewBackbones` | Reuse as Fusion configuration |
| Batched shared encoder | `ViewBackbones` | Reuse as Fusion configuration |
| Concatenation projection | `MultiViewFusion` | Reuse as `fusion_type: concat_proj` |
| Per-view projection | `MultiViewFusion` | Reuse as `fusion_type: indiv_proj` |
| Attention fusion | `MultiViewFusion` | Reuse as `fusion_type: attention` |
| Learned weighted sum | `MultiViewFusion` | Reuse as `fusion_type: weighted_sum` |
| Input-dependent gated fusion | `MultiViewFusion` | Reuse as `fusion_type: gated` |

The existing attention, weighted, and gated implementations combine
deterministic feature vectors. They do not combine Gaussian VAE posteriors.
Product-of-experts, mixture-of-experts, multi-decoder, cross-view
reconstruction, shared/private latent, hierarchical, missing-view, and temporal
VAEs are not currently implemented.

## Proposed model families

| Family | Input path | Objective | Resulting representation |
|---|---|---|---|
| Fusion model | Neo mean encoder per view, followed by `MultiViewFusion` | Supervised robot-position regression | A fused, task-tuned latent |
| Concat model | Horizontally concatenate synchronized views, then run one Neo VAE | Reconstruction plus KL divergence | A joint generative latent |

### Fusion model

The proposed model processes the views as follows:

```text
camera 1 -> Neo mean encoder -> z1 --+
camera 2 -> Neo mean encoder -> z2 --+-> MultiViewFusion -> fused latent
camera N -> Neo mean encoder -> zN --+                         |
                                                               +-> proprioceptor
```

The view encoder should contain the encoder blocks and `fc_mu` from a trained
`ConvVAENeo`. The decoder and `fc_logvar` are not needed in this model and
should not be retained in its deployed architecture.

The model should reuse the existing
[`ViewBackbones`](multiview_backbones.py) and
[`MultiViewFusion`](multiview_fusion.py). This retains the
existing support for shared or per-view encoders and for `concat_proj`,
`indiv_proj`, `attention`, `weighted_sum`, and `gated` fusion.

The default workflow is two-stage training:

1. Train one ordinary Conv-VAE-Neo on images from every camera that will be
   used by the multiview model. Camera streams are separate samples during this
   stage.
2. Initialize the multiview view encoders from that VAE, then train the fusion
   and proprioception heads on synchronized images and normalized robot
   positions.

The Neo view encoder should be frozen by default. An experiment may enable
supervised fine-tuning, but this changes the interpretation: the resulting
per-view representation is no longer solely VAE-trained.

A shared backbone should also be the default. Because Neo uses GroupNorm rather
than batch statistics, all views can safely be combined along the batch
dimension for one efficient shared-backbone call. A per-view mode can initialize
each encoder from the same source VAE and then allow the encoders to diverge
during fine-tuning.

The five fusion algorithms are configuration variants of this one Fusion model,
not five additional model classes. They have the same inputs, supervised
objective, output contract, and checkpoint structure.

The source VAE is a training dependency only. Once the Fusion model has been
trained, its final model file must contain all view-encoder, fusion, and
proprioceptor weights. Runtime loading must not require the source VAE
experiment.

Proposed names:

- Model and training module: `src/sensorprocessing/conv_vae_neo_multiview_fusion.py`
- Model class: `ConvVAENeoMultiViewFusionModel`
- Runtime wrapper: `src/sensorprocessing/sp_conv_vae_neo_multiview_fusion.py`
- Training notebook: `src/sensorprocessing/Train_Conv_VAE_Neo_MultiView_Fusion.ipynb`
- Verification notebook: `src/sensorprocessing/Verify_Conv_VAE_Neo_MultiView_Fusion.ipynb`
- Experiment: `sensorprocessing_conv_vae_neo_multiview_fusion`
- Runtime class: `ConvVaeNeoMultiViewFusionSensorProcessing`

The module owns a Fusion-specific `train(exp, epochs=None)` function. That
function builds the supervised data, initializes the Neo view encoder only for
a genuinely new training run, creates the fusion/regression optimizer, and
delegates checkpoint lifecycle to the training harness.

### Concat model

This is the true joint multiview VAE. Every example consists of synchronized,
ordered camera images. For two RGB views, the transformation is:

```text
[B, 3, H, W] + [B, 3, H, W] -> [B, 3, H, 2W] -> ConvVAENeo
```

The reconstruction has the same composite geometry and can be split back into
one panel per camera for inspection. The posterior mean is the deterministic
runtime latent, while training uses the existing Neo reconstruction and KL
losses.

The composite must not be resized back to width `W`. Keeping the full width
preserves the information from every view, and `ConvVAENeo` already derives its
network depth and bottleneck geometry from rectangular inputs.

In the exp/run, `image_size` should continue to mean the size of one camera
image. The model derives its internal VAE size as `[height, width * num_views]`.
This keeps training preprocessing and runtime `SensorPreprocessor` behavior
consistent.

Only horizontal concatenation should be supported initially. Channel stacking
would change the VAE input from three channels to `3 * num_views` and would
require different model, sampling, and visualization semantics. Unsupported
stacking modes should raise an exception.

The Concat model is independent of the Fusion model. It does not instantiate
`ViewBackbones`, `MultiViewFusion`, or a proprioceptor. Conversely, the Fusion
model does not use the Concat model's composite-image forward path.

Proposed names:

- Model and training module: `src/sensorprocessing/conv_vae_neo_multiview_concat.py`
- Model class: `ConvVAENeoMultiViewConcatModel`
- Runtime wrapper: `src/sensorprocessing/sp_conv_vae_neo_multiview_concat.py`
- Training notebook: `src/sensorprocessing/Train_Conv_VAE_Neo_MultiView_Concat.ipynb`
- Verification notebook: `src/sensorprocessing/Verify_Conv_VAE_Neo_MultiView_Concat.ipynb`
- Experiment: `sensorprocessing_conv_vae_neo_multiview_concat`
- Runtime class: `ConvVaeNeoMultiViewConcatSensorProcessing`

This module owns its own `train(exp, epochs=None)` function, VAE epoch steps,
composite construction, and reconstruction metrics. It uses `ConvVAENeoLoss`
and does not import any Fusion-model training behavior.

## Data contract

Both families should use the existing ordered multiview entry format:

```text
[demonstration run, demonstration name, [camera 1, camera 2, ...]]
```

Camera order is part of the trained model. Training, validation, demonstration
processing, and live captures must all use exactly that order. The Neo wrappers
should reject a different order rather than silently produce an invalid latent.

A shared lazy synchronized multiview dataset should be added for the two model
families. Sharing this dataset does not couple the models: the Concat trainer
uses only the ordered images, whereas the Fusion trainer also requests the
normalized robot-position target. The dataset should:

- index demonstrations by timestep without loading the complete dataset into
  memory;
- load every configured camera at the same timestep through `Demonstration`;
- use the same transform for training and runtime processing;
- support image-backed and video-backed demonstrations;
- honor `frame_stride`, `num_workers`, and `pin_memory`;
- return ordered views for VAE training and optionally the normalized robot
  position for supervised Fusion training;
- reject missing or duplicate cameras, unequal view counts, unreadable frames,
  empty datasets, and non-finite targets; and
- prevent the same demonstration from appearing in training and validation,
  even if the configured camera lists differ.

Lazy loading is important because materializing several 256-pixel camera streams
as cached tensors scales poorly in both memory and storage.

## Exp/run configuration

The Fusion experiment family should have the following conceptual groups of
fields:

- Neo architecture: `architecture_version`, `image_size`, `latent_size`,
  `base_channels`, `max_channels`, `bottleneck_max_size`, and
  `group_norm_groups`.
- Source initialization: `source_vae_experiment`, `source_vae_run`, and
  `freeze_vae_encoder`.
- Views: `num_views`, ordered `cameras`, `shared_backbone`, and
  `batched_backbone`.
- Fusion: `fusion_type` and the existing optional fusion fields.
- Supervised head: `output_size`, `proprio_step_1`, and `proprio_step_2`.
- Training and data fields following the existing multiview conventions.

The source VAE's architecture and latent size must match the Fusion experiment.
Initialization should fail immediately on a mismatch. Source weights should be
loaded only when starting a new model; resuming a checkpoint or loading a
completed Fusion model should not consult the source VAE.

The Concat experiment family should use the normal Neo VAE fields plus `num_views`,
ordered `cameras`, and `stack_mode: width`. Its `image_size` is explicitly the
per-view size; composite geometry is derived rather than independently
configured.

Each family has its own defaults, run files, data directory, and `model_file`.
There should be no exp/run field that selects between Concat and Fusion. Both
families should use the shared training harness. They should not use the
external library's JSON configuration,
timestamped `model_subdir`, manually copied checkpoint name, or materialized
JPEG training directory.

## Training notebooks

The notebooks should be thin entry points. Model construction, dataloaders,
losses, checkpoint handling, and training steps belong in Python modules so the
same behavior can be tested without executing notebooks.

Each training notebook should:

1. Provide Papermill-compatible exp/run and path parameters.
2. Load and display the exp/run.
3. Call a module-level `train(exp, epochs=epochs)` function.
4. Explicitly reload `checkpoints/best_model.pth` before validation.
5. Include a commented alternative for an `epoch_XXXXXX.pth` checkpoint.
6. Accept both the raw best-model state dictionary and the wrapped intermediate
   checkpoint format.

The Fusion notebook should report per-DOF RMSE and MAE, overall RMSE, latent
shape, backbone frozen state, and view weights for fusion methods that expose
them.

The Concat notebook should report reconstruction and KL losses, overall
and per-camera reconstruction metrics, latent statistics, camera-separated
reconstruction panels, and samples from the joint prior.

## Runtime integration

Both wrappers should derive from the existing
`MultiViewEncoderSensorProcessing`; no new sensor-processing root class is
needed. They should load their completed model with
`load_encoder_checkpoint(required=True)`, enter evaluation mode, validate the
number and order of views, and return a squeezed NumPy latent consistent with
the other sensor processors.

Both runtime class names must be registered independently in `sp_factory.py`
and in its multiview class set. The two experiment families must also be copied
by the visual-proprioception flow helper so they can be used as `sp_experiment`
and `sp_run` dependencies.

## Checkpoint behavior

The shared BerryPicker training harness should own checkpoint creation,
resumption, best-model selection, early stopping, and final model export.

Runtime always loads the exported `model_file`. Notebook inspection may load
either:

- `checkpoints/best_model.pth`, currently stored as a raw state dictionary; or
- `checkpoints/epoch_XXXXXX.pth`, stored as a full checkpoint containing
  `model_state_dict` and optimizer/training state.

Loading utilities must explicitly support both forms and use strict state-dict
loading so architecture mismatches abort.

## Validation and tests

The implementation should include tests for:

- rectangular and large composite image geometry;
- reconstruction and latent shapes for different view counts;
- deterministic evaluation encodings;
- synchronized camera loading and stable camera ordering;
- missing, duplicate, or incorrectly ordered cameras;
- no demonstration overlap between training and validation;
- exact extraction of Neo encoder and `fc_mu` weights;
- shared, per-view, frozen, and fine-tuned backbones;
- all five existing fusion heads;
- raw final-model and wrapped checkpoint loading;
- factory construction and multiview detection; and
- training-step loss finiteness and gradient propagation.

## Reuse and inheritance decision

A shared root model class for the Fusion and Concat families is not
recommended. Their input paths, objectives, forward results, and checkpoint
contents are materially different. A common root would contain little behavior
and would make those distinctions less obvious.

Reuse should instead occur through composition and small focused utilities:

- `ConvVAENeo` for the generative model and source encoder architecture;
- `ViewBackbones` and `MultiViewFusion` for the Fusion family;
- `MultiViewEncoderSensorProcessing` for both runtime wrappers;
- one synchronized multiview dataset; and
- one strict ordered-view validation/composition helper.

## Proposed implementation order

1. Add and test only the model-independent ordered-view validation and lazy
   synchronized multiview dataset.
2. Implement the Concat model, its VAE-specific training function, tests,
   wrapper, experiment family, and notebooks as one self-contained vertical
   slice.
3. Verify that the Concat model supports full-width rectangular inputs and does
   not depend on any Fusion module.
4. Implement the Neo mean-encoder adapter and the Fusion model as a second
   self-contained vertical slice, reusing the existing backbone and fusion
   utilities.
5. Add the Fusion-specific supervised training function, wrapper, experiment
   family, and notebooks.
6. Register both runtime classes separately in the factory and multiview class
   set.
7. Add visual-proprioception flow and comparison configurations for each family.
8. Retain all legacy model families unchanged for reproducibility and direct
   comparison.

## Deferred work

The following variations are deliberately out of the first implementation:

- channel-concatenation VAE;
- tiled or mosaic input VAE;
- product-of-experts or mixture-of-experts posterior fusion;
- a joint latent with one decoder per camera;
- cross-view reconstruction;
- shared and camera-private latent decomposition;
- hierarchical multiview VAE;
- missing-view inference and variable runtime view counts; and
- temporal multiview VAE.

Of these, product-of-experts posterior fusion with one decoder per camera is the
most natural future third model. It should be designed as a separate model
family rather than added as another `fusion_type`, because it changes the
probabilistic objective, decoder structure, missing-view behavior, and
checkpoint format. Channel concatenation or mosaic layout could later become
additional Concat-family models if experiments justify them, but they should
not be switches in the initial horizontal Concat model.
