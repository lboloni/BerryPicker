# Proprioception-tuned CNN sensor processing

## Purpose

The proprioception-tuned CNN converts one camera image into a compact latent
vector. It starts with a standard ImageNet-pretrained VGG19 or ResNet50 feature
extractor, reduces the wide CNN feature representation to
`exp["latent_size"]`, and trains that reduction using the auxiliary task of
predicting the robot position.

The robot-position predictor is a training head. Runtime sensor processing
returns the intermediate latent vector, not the predicted position. The design
therefore uses proprioception as supervision for representation learning:

```text
RGB image -> pretrained CNN -> wide visual features -> learned reduction -> z
                                                                    |
                                                                    v
                                             training-only proprioception head
                                                                    |
                                                                    v
                                                   normalized robot position
```

The reduction is not a separately fitted method such as PCA. It is a neural
network bottleneck trained jointly with the proprioception head. With the
current `freeze_feature_extractor: true` runs, the pretrained CNN is fixed and
the reduction and regression head learn. If that field is false, gradients can
also update the CNN backbone.

## Main implementation

[sp_propriotuned_cnn.py](sp_propriotuned_cnn.py) contains three model classes
and the runtime sensor-processing wrapper.

`_ProprioTunedCNNRegression` defines the shared execution split:

```python
def encode(self, x):
    features = self.feature_extractor(x)
    return self.encode_flat_features(self.flatten(features))

def forward(self, x):
    return self.predict_from_latent(self.encode(x))
```

`forward()` is used during proprioception training. `encode()` is used after
training to expose the learned representation. Both routes use exactly the
same backbone and reduction parameters.

The backbone uses Torchvision's default pretrained weights unless
`exp["pretrained_backbone"]` is false. This field defaults to true in code.
`exp["freeze_feature_extractor"]` also defaults to true.

### VGG19 variant

`VGG19ProprioTunedRegression` retains `models.vgg19(...).features` and removes
the classifier. For the configured 256 by 256 input, the feature tensor is
flattened to `512 * 8 * 8 = 32768` values. Its complete head is:

```text
32768 -> latent_size -> ReLU -> latent_size -> ReLU -> output_size
```

The latent returned by `encode()` is the output of the second `latent_size`
linear layer, before the following ReLU. The remaining ReLU and final linear
layer form the proprioception predictor.

The `512 * 8 * 8` input width is hard-coded, so the current VGG19 implementation
is coupled to 256 by 256 images. Changing `image_size` without changing the
head causes a matrix-shape error.

### ResNet50 variant

`ResNetProprioTunedRegression` removes the final ResNet50 classification layer
but retains its global average pooling. The flattened feature vector therefore
has 2048 elements. It uses two explicit networks:

```text
reductor:       2048 -> reductor_step_1 -> ReLU -> latent_size
proprioceptor:  latent_size -> proprio_step_1 -> ReLU
                            -> proprio_step_2 -> ReLU -> output_size
```

For the repository runs, `reductor_step_1` is 512, the proprioceptor widths are
64 and 16, and `output_size` is 6. `encode()` returns the reductor output;
`forward()` passes it through the proprioceptor.

## Proprioception supervision

[helper_training_data.py](helper_training_data.py) builds the single-view
training tensors. Every item in `exp["training_data"]` has this form:

```yaml
- [demonstration_run, demonstration_name, camera]
```

For every timestep in each selected demonstration, the loader obtains:

- the selected camera image, transformed to the configured image size; and
- the `rc-position-target` action recorded for that timestep.

The action is a `RobotPosition`. `RobotPosition.to_normalized_vector()` maps the
six AL5D fields independently from their configured robot limits to `[0, 1]`:

```text
height, distance, heading, wrist_angle, wrist_rotation, gripper
```

The training target is consequently a six-dimensional normalized robot pose.
Minimizing pose-regression loss encourages the latent to preserve visual
features that distinguish the robot's configuration. It does not directly
supervise object identity, reconstruction quality, or manipulation actions.

### Image preprocessing

[sp_helper.py](sp_helper.py) provides the common training and inference
transform:

```text
PIL RGB -> Resize(image_size) -> CenterCrop(image_size) -> ToTensor()
```

The resulting tensor is `[channels, height, width]` with values in `[0, 1]`.
Inference adds a batch dimension and moves it to the configured device. The
current CNN path does not apply ImageNet mean/std normalization or data
augmentation; training and runtime nevertheless use the same transform.

### Caching and partitioning

The complete image and target tensors are cached below `exp["data_dir"]` as
`proprioception_input_file` and `proprioception_target_file`. Both cache files
must exist to be reused; otherwise both are rebuilt from the demonstrations.

The single-view loader shuffles the cached examples and makes a frame-level
67/33 training/validation split. `Train_ProprioTuned_CNN.ipynb` supplies a
deterministically seeded generator, currently using seed 777.

Although the expruns define `validation_data`,
`proprioception_test_input_file`, and `proprioception_test_target_file`, the
single-view loader does not currently use them. Those fields are used by the
newer multiview loader. The single-view validation loss is therefore computed
on the held-out 33 percent of `training_data`, not on the demonstrations listed
in `validation_data`. Because the split is by frame rather than by complete
demonstration, nearby frames from one recording can occur in both partitions.

## Training notebook and artifacts

[Train_ProprioTuned_CNN.ipynb](Train_ProprioTuned_CNN.ipynb) is the training
entry point. It performs the following sequence:

1. Load a `sensorprocessing_propriotuned_cnn` exp/run and its robot exp/run.
2. Instantiate either `VGG19ProprioTunedRegression` or
   `ResNetProprioTunedRegression` from `exp["model"]`.
3. Select `MSELoss` or `L1Loss` from `exp["loss"]`.
4. Build an Adam optimizer using `exp["learning_rate"]`.
5. Build cached training data and DataLoaders unless a completed model will be
   loaded directly.
6. Call `training_harness.load_or_train()` to load, resume, or train the model.

Each training batch runs the complete image-to-position `forward()` path. The
validation loop uses the same loss without gradients. The harness retains
recent epoch checkpoints, tracks the best validation loss, and exports the
best complete `state_dict`.

The final file is:

```text
exp["data_dir"] / exp["proprioception_mlp_model_file"]
```

The current filename is `proprioception_mlp.pth`. Despite that legacy name, the
file contains the state of the complete model: backbone, reduction, and
proprioception head. This is why runtime can reconstruct the full class and
strictly load one checkpoint.

`creation_style: discard-old` is the training notebook's interactive default,
so a normal run removes the old result directory and its caches before
training. A flow can override it with `exist-ok`; when the final model already
exists and `reload_existing_model` is true, the harness loads it without
rebuilding the dataset.

## Runtime sensor-processing interface

`ProprioTunedCNNSensorProcessing` is the generic single-view wrapper. It selects
the model class from `exp["model"]`, builds the same architecture, and requires
the trained checkpoint. It inherits preprocessing and inference behavior from
`SingleViewEncoderSensorProcessing`:

```text
image or capture -> shared preprocessing -> model.encode() -> NumPy latent z
```

Inference switches the model to evaluation mode and runs without gradients.
The proprioception prediction is deliberately not returned. Downstream
behavior-cloning, visual-proprioception, and robot-controller components consume
`z` as the image representation.

[sp_factory.py](sp_factory.py) supports the generic current names:

- `class: ProprioTunedCNNSensorProcessing` or the legacy
  `class: ProprioTunedCNN`, both with an explicit `model`; and
- the older `VGG19ProprioTunedSensorProcessing` and
  `ResNetProprioTunedSensorProcessing` class names, which the factory maps to
  the corresponding model automatically.

[rcco_sp_cnn.py](../robot_controller/rcco_sp_cnn.py) embeds the same encoder in
a composite robot-controller graph. It propagates through `model.encode()` and
publishes a tensor named `z`; it does not expose the proprioception head. When
constructing a controller model for later state loading, it can explicitly
disable pretrained weight acquisition because the complete trained state will
replace the initial parameters.

## Exp/run configuration

The experiment family is
`data/expruns/sensorprocessing_propriotuned_cnn`. Its four current runs combine
two backbones and two latent widths:

| Run | Backbone model | Latent size | Epochs |
|---|---|---:|---:|
| `vgg19_128` | `VGG19ProprioTunedRegression` | 128 | 10 |
| `vgg19_256` | `VGG19ProprioTunedRegression` | 256 | 10 |
| `resnet50_128` | `ResNetProprioTunedRegression` | 128 | 40 |
| `resnet50_256` | `ResNetProprioTunedRegression` | 256 | 100 |

All four use 256 by 256 input, a frozen feature extractor, six-dimensional
proprioception output, and the training and verification notebooks declared in
`input-to-notebook`.

Important fields are:

| Field | Meaning |
|---|---|
| `class` | Factory-facing sensor-processing class or compatibility alias |
| `model` | Regression architecture constructed by the training/runtime wrapper |
| `image_size` | Shared training and inference geometry |
| `latent_size` | Width of the deployed representation `z` |
| `output_size` | Width of the training-only proprioception prediction |
| `freeze_feature_extractor` | Whether the pretrained CNN receives gradients |
| `pretrained_backbone` | Whether model construction requests default pretrained weights; defaults to true in code |
| `reductor_step_1` | ResNet reduction hidden width |
| `proprio_step_1`, `proprio_step_2` | ResNet proprioception-head widths |
| `loss` | `MSELoss` or `L1Loss` in the training notebook |
| `proprioception_mlp_model_file` | Legacy key naming the complete exported model state |

`model_name` remains in the family default but does not select the current
architecture; `model` is the operative field.

## Current implementation boundaries

- With the repository defaults, “proprioception tuned” describes the learned
  bottleneck rather than a fine-tuned CNN backbone, because the backbone is
  frozen.
- The VGG19 reduction assumes a 256 by 256 input and a `512 * 8 * 8` flattened
  feature tensor. The ResNet50 version is not tied to that spatial feature
  width because it retains global average pooling.
- Normal construction requests Torchvision's default pretrained weights before
  loading the complete local checkpoint. This can require the weights to exist
  in the Torch cache. The controller's state-loading construction path avoids
  that redundant requirement by setting `pretrained_backbone` false.
- The single-view cache has no manifest describing the demonstrations or image
  transform. Changing those exp/run fields does not invalidate existing cache
  files automatically.
- `Verify_ProprioTuned_CNN.ipynb` predates the generic wrapper refactor. It
  imports wrapper class names that no longer exist in
  `sp_propriotuned_cnn.py`, calls the deprecated `set_experiment_path()`, and
  expects `testing_data`, while current runs provide `validation_data`. The
  training and runtime implementation described above is current; that
  verification notebook needs a separate update before it can exercise it.

## Related multiview model

[sp_propriotuned_cnn_multiview.py](sp_propriotuned_cnn_multiview.py) applies the
same high-level idea to ordered camera views, but it is a separate model family.
It projects each view, fuses the projected features, and trains the fused latent
with a proprioception head. It derives the backbone feature width dynamically
and uses the multiview dataset/cache implementation. Those details should not
be assumed to describe the single-view VGG19/ResNet50 code in this document.
