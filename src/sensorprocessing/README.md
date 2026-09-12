# What is this

The ```sensorprocessing''' package contains code to process the robots' sensor input and output an __encoding vector z__. The sensor processing code is usually some learned encoding model. As of Feb 2025, this is a single camera vision input. The use of other type of sensory information is planned for the future. 

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
