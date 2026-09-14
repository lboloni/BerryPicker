# Random-projection CNN sensor processing

## Purpose and scope

Define a sensor-processing model that converts a camera image into a compact
latent vector without training or fitting a dimensionality reducer. The model
uses a frozen pretrained CNN as its visual feature extractor and a fixed random
linear projection as its bottleneck.

This is an alternative to the proprioception-tuned CNN described in
[DESIGN-Proprioception-Tuned-CNN.md](DESIGN-Proprioception-Tuned-CNN.md). The
proprioception-tuned model learns its reduction and regression head from robot
data. The random-projection model has no proprioception head, dataset, loss,
optimizer, or training process.

ResNet50 and VGG19 are separate sensor-processing models. They do not share a
backbone, projection matrix, checkpoint, or data. They share only the same
method:

```text
image -> frozen pretrained CNN -> fixed feature vector
      -> fixed random projection -> latent z
```

The initial scope is single-view RGB processing. Multiview fusion is separate.

## Motivation

A random projection provides a training-free control for learned
dimensionality reduction. It tests how much downstream performance comes from
the pretrained CNN representation itself rather than from tuning the
bottleneck with proprioception.

For a feature vector `x` of width `D` and desired latent width `K`, generate a
fixed matrix `R` with shape `[K, D]` and compute:

```text
z = R x
```

Random-projection results motivate this approach by showing that a sufficiently
wide random embedding can approximately preserve pairwise Euclidean geometry.
They do not guarantee that a small projection will preferentially retain robot
pose, object identity, or manipulation-relevant information. The absence of
task preference is both the value and the expected limitation of this
baseline.

## Model architecture

Use a small common abstraction around backbone-specific feature adapters:

```python
class RandomProjectionCNN(nn.Module):
    def __init__(self, feature_extractor, feature_size, exp):
        ...

    def encode(self, images):
        features = self.feature_extractor(images)
        features = self.flatten(features)
        features = self.normalize(features)
        return torch.nn.functional.linear(features, self.projection)

    def forward(self, images):
        return self.encode(images)
```

`projection` is a registered buffer, not an `nn.Parameter`. It moves with the
model and appears in a serialized `state_dict`, but it never receives a
gradient.

Concrete model classes supply independently constructed backbones and feature
geometry:

```text
ResNet50RandomProjection
VGG19RandomProjection
```

The model returns `[batch, exp["latent_size"]]`. There is no secondary output
or training-only head.

### Fixed preprocessing

Training and inference consistency is not an issue because there is no local
training, but input interpretation must still be fixed. Use the repository's
shared RGB resize and center-crop geometry, followed inside the model by the
normalization associated with the selected pretrained weights. This avoids
making the backbone consume a different value distribution from the one its
pretrained weights expect.

Feature-vector normalization is separate from image normalization. Make it an
exp/run choice:

- `none` preserves the backbone feature magnitude before projection.
- `l2` projects unit-length feature vectors and emphasizes angular geometry.

Use `l2` as the initial default, but compare it with `none` because feature
magnitude may contain useful information.

## Backbone-specific representations

The two backbones do not have the same internal feature width. Each model must
therefore create its own projection matrix.

### ResNet50

Remove the classification layer and retain ResNet50's global average pooling:

```text
RGB image -> ResNet50 convolutional body and average pool
          -> [batch, 2048, 1, 1]
          -> flatten to D = 2048
          -> independent [K, 2048] projection
          -> z
```

The pooled width is 2048 for the configured image geometry. This is the
simplest initial implementation.

### VGG19

Use `models.vgg19(...).features` and exclude VGG19's large classifier. At a
256 by 256 input size the feature extractor produces:

```text
[batch, 512, 8, 8]
```

The primary research baseline preserves this complete spatial feature map:

```text
RGB image -> VGG19 features
          -> flatten to D = 512 * 8 * 8 = 32768
          -> independent [K, 32768] projection
          -> z
```

This is not the ResNet projection reused at a different size. It is a separate
random draw for a separate VGG19 model.

The feature width should be derived by passing a dummy tensor with the
configured image geometry through the frozen feature extractor. Do not
hard-code `512 * 8 * 8` in the common projection class.

## Projection distributions

### Rademacher projection

Use a dense Rademacher matrix as the initial method:

```text
R[i, j] = +1 / sqrt(K) with probability 1/2
R[i, j] = -1 / sqrt(K) with probability 1/2
```

It is simple to generate, requires no learned parameters, and replaces a dense
Gaussian matrix with signs and one common scale. It should be the default
method for the first experiment set.

### Gaussian projection

A conventional alternative is:

```text
R[i, j] ~ Normal(0, 1/K)
```

This is a useful reference but is not expected to offer a compelling practical
advantage over the Rademacher baseline here.

### Sparse random projection

A sparse matrix with values in `{-a, 0, +a}` can reduce projection storage and
arithmetic. It is most relevant for the direct VGG19 projection. Sparse tensor
operations should be benchmarked on the actual device: mathematical sparsity
does not necessarily produce lower latency with every PyTorch backend.

### Structured projection

A signed Hadamard transform followed by subsampling, or another fast
Johnson--Lindenstrauss transform, avoids a full dense matrix. This becomes
attractive for much wider features or strict memory limits. It adds padding,
transform, and device-support complexity and is not recommended for the first
implementation.

Random nonlinear MLPs and random convolutions are possible, but they introduce
additional architectural choices without the clear distance-preservation
interpretation of a linear projection. They should not be initial baselines.

## Determinism and ownership

Each exp/run owns its projection through:

- the projection algorithm;
- the projection seed;
- the backbone and pretrained-weight identifier;
- the input feature geometry; and
- the output latent size.

Generate the matrix on CPU from a dedicated generator initialized with
`exp["projection_seed"]`, then register it as a buffer. Do not use or modify the
process-wide random-number generator.

The matrix can be regenerated when the model is constructed. For long-lived
experiments that require bit-for-bit reproduction across library versions,
also export the model `state_dict` or the projection buffer as an untrained
artifact. Persisting a sampled matrix does not turn it into a trained model.

Different backbone runs must have independent seeds or an explicit convention
for deriving a model-specific seed. Equal integer seeds do not imply equal
projections because their matrix shapes differ.

## Size and computation

Dense float32 projection sizes are:

| Backbone representation | Projection | Parameters | Storage |
|---|---:|---:|---:|
| ResNet50, `D=2048` | 2048 -> 128 | 262,144 | 1 MiB |
| ResNet50, `D=2048` | 2048 -> 256 | 524,288 | 2 MiB |
| VGG19, `D=32768` | 32768 -> 128 | 4,194,304 | 16 MiB |
| VGG19, `D=32768` | 32768 -> 256 | 8,388,608 | 32 MiB |

`vgg19.features` contains roughly 76 MiB of float32 parameters. Excluding its
classifier keeps the total approximate parameter storage to 92 MiB with the
128-dimensional projection or 108 MiB with the 256-dimensional projection.
This is practical for batch-one inference on a desktop or laptop. VGG19's
convolutional computation is more likely to dominate latency than the final
projection.

### Optional VGG spatial pooling

If VGG memory or projection latency is important, insert a deterministic
adaptive pooling operation before flattening:

| Pooling | Feature width | 256-dimensional projection |
|---|---:|---:|
| none, 8x8 | 32,768 | 32 MiB |
| 4x4 | 8,192 | 8 MiB |
| 2x2 | 2,048 | 2 MiB |
| 1x1 | 512 | 0.5 MiB |

Pooling is not merely an implementation optimization. Global pooling removes
explicit spatial layout, which may be important when an image contains the
robot and its position must be inferred. Treat each pooling geometry as a
separate exp/run. Use the unpooled projection as the clean baseline and 4x4 as
the first reduced deployment variant.

## Exp/run design

Use one experiment family, for example
`sensorprocessing_random_projection_cnn`, with independent runs for each
backbone, latent width, projection seed, and optional pooling geometry.

```yaml
class: RandomProjectionCNNSensorProcessing
model: ResNet50RandomProjection

image_size: [256, 256]
latent_size: 128

backbone_weights: DEFAULT
feature_pool_size: [1, 1]
feature_normalization: l2

projection: rademacher
projection_seed: 73017
```

A VGG19 run changes the model and pooling geometry:

```yaml
class: RandomProjectionCNNSensorProcessing
model: VGG19RandomProjection

image_size: [256, 256]
latent_size: 128

backbone_weights: DEFAULT
feature_pool_size: [8, 8]
feature_normalization: l2

projection: rademacher
projection_seed: 18493
```

All fields defining the representation belong in the exp/run. Downstream code
should only depend on `latent_size` and the ordinary single-view SP interface.

Because there is no training operation, this family has no Train notebook. Its
`input-to-notebook` should list a verification notebook that inspects latent
shape, finiteness, determinism, latency, and downstream evaluation results.

## Runtime integration

Implement a `RandomProjectionCNNSensorProcessing` wrapper derived from
`SingleViewEncoderSensorProcessing`. It should:

1. select the concrete model from `exp["model"]`;
2. construct the frozen pretrained backbone;
3. derive the backbone feature width for `exp["image_size"]` and the configured
   pooling geometry;
4. generate and register the fixed projection;
5. expose it as `self.enc`; and
6. use the inherited `process()` and `process_file()` entry points.

Unlike trained encoder wrappers, construction must not require a checkpoint.
The complete SP remains stateless and returns a NumPy latent through the
existing runtime interface. Add one factory entry in `sp_factory.py`; behavior
cloning, visual proprioception, and robot-controller consumers should require
no model-specific changes.

## Initial experiment matrix

Start with:

```text
resnet50_rademacher_128
resnet50_rademacher_256
vgg19_rademacher_128
vgg19_rademacher_256
vgg19_pool4_rademacher_128
vgg19_pool4_rademacher_256
```

Create several projection-seed runs for every architecture. A single random
draw is not sufficient to characterize the method.

Compare against:

- the native unprojected backbone feature;
- deterministic pooling without a random projection;
- the existing proprioception-tuned CNN at the same latent width; and
- PCA at the same width if a data-dependent but label-free baseline is useful.

PCA is not part of the training-free design because it must be fitted to a
feature dataset. It is nevertheless a valuable comparison for separating the
effects of data-dependent dimensionality reduction from proprioception
supervision.

Evaluation should include:

- output shape, finiteness, and repeatability;
- batch-one latency and peak memory on the deployment machine;
- approximate preservation of pairwise feature distances or cosine
  similarities;
- visual-proprioception and behavior-cloning performance; and
- mean and variance across projection seeds.

## Recommended first implementation

Implement dense Rademacher projection for ResNet50 and unpooled VGG19, at 128
and 256 dimensions. Generate a separate matrix for every exp/run, keep all
backbones frozen, and use no experiment data. Add the VGG19 4x4 pooling variant
only as the first memory/latency comparison.

This provides the most direct answer to the research question: how useful are
the fixed pretrained CNN features after a generic, completely untrained
dimensionality reduction?

## References

- Dimitris Achlioptas, “Database-friendly random projections:
  Johnson--Lindenstrauss with binary coins,” *Journal of Computer and System
  Sciences*, 2003. https://doi.org/10.1016/S0022-0000(03)00025-4
- Ping Li, Trevor Hastie, and Kenneth Church, “Very Sparse Random Projections,”
  *KDD*, 2006. https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
- Nir Ailon and Bernard Chazelle, “The Fast Johnson--Lindenstrauss Transform and
  Approximate Nearest Neighbors,” *SIAM Journal on Computing*, 2009.
  https://doi.org/10.1137/060673096
