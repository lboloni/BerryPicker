# VAE-GAN

`VAEGAN` combines the unchanged `ConvVAENeo` architecture with a discriminator
that returns real/fake logits and intermediate spatial features. It accepts
single-view RGB tensors in `[0, 1]`, including large and non-square images.
The discriminator exists only during representation training; deployment uses
the VAE latent mean through `VAEGANSensorProcessing`.

## Training

Select `sensorprocessing_vae_gan/sp_vae_gan_128_256px` or
`sp_vae_gan_256_256px` in `Train_VAE_GAN.ipynb`. Configure demonstration triples
as in Neo; even different cameras from the same demonstration cannot cross the
training/validation boundary. Memory use grows with resolution; reduce batch
size or channel widths for larger images.

Three Adam optimizers perform isolated updates:

- Encoder: weighted KL + feature reconstruction + optional pixel MSE.
- Generator (Neo decoder): weighted feature reconstruction + adversarial loss
  on reconstructed and prior-generated images + optional pixel MSE.
- Discriminator: real-image BCE plus the mean of the two fake-image BCE terms.

Loss reductions are means; the KL convention matches Neo. Generator adversarial
loss is non-saturating BCE-with-logits. Freezing discriminator parameters during
encoder/generator updates does not block gradients through its image input.
GroupNorm avoids running-statistic changes during frozen updates. The optional
pixel term defaults to zero. Feature loss is learned and should not be compared
across epochs as if its reference metric were fixed.

Set `initialization: conv_vae_neo`, `initialization_experiment` and
`initialization_run` to warm-start from a compatible Neo export. Architecture
settings and state dictionaries must match; missing/incompatible sources raise.
Source path and SHA-256 are recorded. Optional `discriminator_warmup_epochs`
trains only the discriminator before joint training. Defaults start randomly
with no warmup; these are starting hyperparameters, not validated performance
claims.

## Recovery and artifacts

Everything is placed in the exp/run data directory:

- `checkpoints/last.pt`: authoritative full training state (all models, three
  optimizers, next epoch/phase, RNG states, history and best VAE weights).
- `checkpoints/epoch_XXXXXX.pt`: retained intermediate full checkpoints;
  `keep_checkpoints` bounds their number and older intermediate files are removed.
- `vae_gan_vae.pth`: raw Neo-compatible weights from the lowest validation pixel
  MSE, evaluated with the deterministic latent mean.
- `metrics.jsonl` and `training_manifest.json`: history and last committed status.

Re-running `train(exp)` resumes automatically. `epochs` is the total target and
may be increased; substantive configuration changes require a new exp/run.
An interrupted epoch is repeated from the previous completed epoch. Files are
replaced atomically; on resume the last checkpoint repairs derived exports and
history. The best export never replaces the last training state. CPU/CUDA RNG
and the shuffled training loader generator are restored; CPU replay is tested,
but identical CUDA results also depend on hardware and deterministic kernels.
Use one training process per exp/run. No scheduler or early stopping is enabled.

The notebook callback plots loss history, fixed validation reconstructions and
fixed prior samples after each committed epoch, without changing training RNG.
Its final separate cell loads the best export; a commented alternative extracts
the VAE from an intermediate checkpoint. `Verify_VAE_GAN.ipynb` additionally
shows interpolation and checks runtime latent equality.

## Robot controllers

`robot_controller/rcco_sp_vae_gan_256` and
`roco_vae_gan_lstm_mdn_sample` reuse the existing `SP_VAE` component and graph.
Train the matching 256-dimensional sensor run first. Downstream controller
fine-tuning operates on the Neo encoder only; it does not continue GAN training.
Assess control performance separately from visual reconstruction quality.
