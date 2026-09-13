# What is this

The ```sensorprocessing''' package contains code to process the robots' sensor input and output an __encoding vector z__. The sensor processing code is usually some learned encoding model. As of Feb 2025, this is a single camera vision input. The use of other type of sensory information is planned for the future.

## Composite architecture

`composite.py` implements ordered named steps, branching, structured result
references (`step.field`), and frozen modules that retain input gradient flow.
`sp_composite.py` exposes `CompositeSensorProcessing` and
`CompositeMultiViewSensorProcessing` through `create_sp()`.

No concrete processing operations are registered by default. Applications must
register their operation implementations before constructing or restoring a
composite:

```python
from sensorprocessing.composite import register_operation

register_operation(
    "my_operation",
    build=build_module,
    resolve=resolve_source_configuration,
    initialize=initialize_source_weights,
)
```

`build(step)` returns an `nn.Module` without reading source checkpoints.
`resolve(step)` returns a complete JSON-serializable step configuration with any
source architecture settings resolved; `initialize(module, step)` initializes
source weights. The latter two hooks are optional and run only for new training.
External-model/API adapters must manage external runtime handles without
registering externally owned weights as PyTorch children. Such adapters and
their caching are not implemented here.

For new training, call `create_composite(exp)` and move the returned model to the
training device. It writes `<model filename stem>.config.json` beside
`model_file(exp)`, containing the resolved graph, latent size, image geometry,
and (for multiview) camera order. It refuses to overwrite an existing snapshot.
Call `restore_composite(exp)` to reconstruct an existing run, then let the
training harness restore model and optimizer state. Restoration does not invoke
source-resolution or source-initialization hooks.

The model supports `forward()`, `encode()`, and `forward_steps()` for auxiliary
losses. Use an ordinary task wrapper and the existing training harness; no loss,
training head, filter, segmentation model, or fusion component is supplied by
this architecture. `frozen: true` disables parameter gradients and preserves
evaluation mode without detaching inputs. Omit `frozen` for trainable or
parameter-free operations.

Save a standalone composite's `state_dict()` to `model_file(exp)` for inference.
If training a wrapper with a task head, save the wrapper's complete state through
the harness for resumption and export **only** `wrapper.composite.state_dict()`
to the SP's final `model_file(exp)`. Keep the wrapper checkpoints separate from
that exported encoder state. Inference accepts either a bare composite state
dictionary or one under `model_state_dict`, not a task wrapper's prefixed state.

`create_sp(exp)` restores the saved configuration and final weights. It uses the
saved image geometry and camera order even if those source settings subsequently
change. Component implementation code must still be available and registered.
See [DESIGN-CompositeSensorProcessing.md](DESIGN-CompositeSensorProcessing.md)
for the broader design and future components.

### Temporal composites

Register a temporal module with `register_operation(..., temporal=True)`. It
implements `advance(*inputs, context=..., dt=...)`, returning `(output,
next_context)` without modifying the incoming context. `None` means the start
of a sequence. No temporal components are supplied by the framework.

Training calls `model.advance(inputs, context, dt=dt)` or
`model.advance_steps(inputs, context, dt=dt)` explicitly. These return the final
encoding or named intermediate outputs, respectively, together with a new
context dictionary keyed by temporal step name. Autograd is preserved; the
training recipe owns unrolling and context detachment. `encode()`, `forward()`,
and `forward_steps()` reject temporal graphs to prevent accidental memory loss.
Stateless graphs keep their original interfaces.

For inference, `sp.process(inputs, dt=dt)` retains runtime context in the SP
wrapper and returns NumPy as before. `sp.reset_context()` starts a new stream;
loading an encoder checkpoint also resets context. Memory is excluded from
model state dictionaries. For fixed-rate inference, configure `sample_interval`
in seconds in the composite exp/run; it is saved in the resolved configuration.
If omitted, every temporal inference call must supply `dt`. File, capture, and
multiview demonstration helpers accept the same optional keyword.

The VP and BC latent-data loaders reset temporal SPs at demonstration boundaries.
They accept `timestep_interval(demonstration, index)` as an optional keyword
callback returning elapsed observation time in seconds. Without it they use the
SP's configured interval, which must match the data. There is no inferred
timestamp convention for existing demonstrations. Temporal loading does not
skip failed frames and bypasses both reads and writes of legacy framewise
caches, leaving existing cache files untouched.

CNN/VAE controller wrappers forward context resets to their SPs. Their existing
model-specific propagation remains stateless; general temporal-composite
controller execution is not introduced here. Sequence trainers, temporal cache
formats, context serialization, and partial batch resets remain family-specific
or future work. See [DESIGN-SensorProcessingMemory.md](DESIGN-SensorProcessingMemory.md).

The size of the encoding vector is specified in the __experiments__ association with these models. The experiments are named sensorprocessing_Foo, and they are in the experiment_configs folder. The experiments also specify the data sets used to train the encoding. 

Train_Foo notebooks contain code to train the model Foo.

Verify_Foo notebooks contain code to verify the learned model Foo. This can be done visually or numerically.

## Models (as of Feb 2025)

* ConvVAE: a convolutional variational autoencoder.
* Conv-VAE-Neo: a BerryPicker-native convolutional variational autoencoder
  with experiment-configured image size and no external model checkout.
* ProprioTunedVGG19: a VGG19 model tuned and dimensionality reduced on proprioception training data.





## Multi-view sensor processing

Feature-fusion multi-view processors share the structure:
backbone(s) -> per-view features -> `multiview_fusion.MultiViewFusion` -> latent.
The Neo MultiView Concat model is intentionally separate: it builds one
full-width RGB observation and encodes it with a joint VAE.

| Module | Backbone | Trained | Runs / notebook |
|---|---|---|---|
| `sp_vit_multiview.py` | ViT (torchvision, pretrained) | fine-tuned (or frozen) | `sensorprocessing_propriotuned_Vit_multiview`, `Train_ProprioTuned_VIT_multiview.ipynb` |
| `sp_propriotuned_cnn_multiview.py` | VGG19 / ResNet50 (pretrained) | frozen backbone, trainable projector/fusion/head | `sensorprocessing_propriotuned_cnn_multiview`, `Train_ProprioTuned_CNN_multiview.ipynb` |
| `sp_conv_vae_multiview.py` | Conv-VAE encoder architecture | from scratch, supervised (no decoder/KL) | `sensorprocessing_conv_vae_multiview`, `Train_Conv_VAE_multiview.ipynb` |
| `sp_conv_vae_neo_multiview_fusion.py` | Conv-VAE-Neo mean encoder(s) | pretrained VAE encoder, supervised fusion/head | `sensorprocessing_conv_vae_neo_multiview_fusion`, `Train_Conv_VAE_Neo_MultiView_Fusion.ipynb` |
| `sp_conv_vae_neo_multiview_concat.py` | Joint full-width Conv-VAE-Neo | unsupervised reconstruction/KL | `sensorprocessing_conv_vae_neo_multiview_concat`, `Train_Conv_VAE_Neo_MultiView_Concat.ipynb` |

Shared pieces:

- `multiview_fusion.py`: the five fusion heads (`concat_proj`, `indiv_proj`, `attention`, `weighted_sum`, `gated`) with widths derived from the backbone feature width, so only the backbone changes between models.
- `multiview_backbones.py`: one shared backbone (default) or one per view; frozen backbones stay in eval mode; batched forward through a shared backbone.
- `helper_training_data.py`: `load_multiview_images_as_proprioception_training` (ordered camera lists, cached), `MultiViewDataset`, `collate_multiview`, `make_multiview_loaders`.
- `multiview_data.py`: lazy synchronized demonstration loading used by the
  two Conv-VAE-Neo multiview families.
- `training_harness`: `load_or_train` + `make_epoch_steps` (multi-view batches, gradient clipping) + optional early stopping.

Training-data entries are `[demonstration_run, demonstration_name, [camera, camera, ...]]`; the camera order is the view order everywhere (SP training, VP training, comparison, runtime). The processors store the trained order in `exp["cameras"]`. The Neo multiview processors reject a different order; legacy processors report a warning.
