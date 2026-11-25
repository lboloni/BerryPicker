# Single-camera vector-quantized VAE

BerryPicker uses a grouped VQ-VAE to study how much robot position information survives discrete image encoding. Like ConvVAENeo, it accepts a 256 × 256 RGB image and returns a deterministic 128-dimensional floating-point vector through `encode(images)`. Visual proprioception and behavior cloning can consume this vector through the sensor-processing interface.

## Architecture and training

`VQVAE` and `ConvVAENeo` share the convolutional encoder and decoder in `ConvAutoencoderBase`. Neo has Gaussian latent heads. VQ-VAE projects the flattened feature map to 128 values and divides them into eight groups of 16. Each group selects its nearest entry by squared Euclidean distance from the same learned codebook of 256 embeddings. `quantize(images)` exposes the projected vectors, embeddings and eight code IDs. `encode(images)` concatenates the selected embeddings in group order. [Model implementation](vq_vae.py)

During training the decoder receives the quantized values. The straight-through estimator carries reconstruction gradients back to the projection. The default commitment weight is 0.25. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

```python
q_st = e + (q - e).detach()
reconstruction = model.decode(q_st.flatten(1))

reconstruction_loss = F.mse_loss(reconstruction, images)
codebook_loss = F.mse_loss(q, e.detach())
commitment_loss = F.mse_loss(e, q.detach())
loss = reconstruction_loss + codebook_loss + exp["commitment_weight"] * commitment_loss
```

Adam trains the embeddings and the rest of the model. The codebook loss moves the selected embeddings toward the encoder outputs. The commitment loss encourages the encoder outputs to stay close to their selected embeddings.

## BerryPicker integration

The encoder exp/run is `sensorprocessing_vq_vae/sp_vq_vae_128_256px`. Its training notebook uses the same demonstration image loader, epoch steps and checkpoint harness as Neo. The model state includes the codebook. The factory loads that state into `VQVAESensorProcessing`. A fixed-batch check reproduced the trained model’s encodings exactly through the factory. A separate check loaded an existing Neo checkpoint into the pre-refactor and current implementations and obtained identical encodings. The unit test checks nearest-code assignments and gradient paths. [Training notebook](Train_VQVAE.ipynb) · [Factory wrapper](sp_vq_vae.py) · [Unit test](../test/sensorprocessing/test_vq_vae.py)

The verification notebook writes pixel-averaged reconstruction MSE and a separate 256-bin code histogram for each of the eight groups over the validation demonstrations. A group that repeatedly chooses the same code contributes little variation to the representation. The histograms are recorded in `verification_metrics.json`. [Verification notebook](Verify_VQVAE.ipynb)

## Evaluation

The collection `comp_vq_vae_vs_neo_128` compares the same visual proprioception regressor trained on each frozen representation. The encoder recipes use matching preprocessing, demonstration splits and training budgets. The two regressor runs reserve the same two demonstrations for the final position-error comparison. This collection is configured. Its completed scores and a standalone validation code histogram were not found in the local results reviewed on 2026-11-25. [Comparison recipe](../../data/expruns/visual_proprioception_collections/comp_vq_vae_vs_neo_128.yaml)

A separate completed run, `VAE_AL5D_AllFrames`, evaluated frozen encoders with a ridge regressor on six normalized AL5D commanded-position fields. On 961 held-out frames from camera `dev2`, VQ-VAE gave pose RMSE **0.2078** and RGB reconstruction MSE **0.00891**. Neo gave **0.2053** and **0.00842**, respectively. Both improved on the training-mean pose baseline of **0.2387**. The report and score table are under `VAE_AL5D_AllFrames/results/vae_encoding_comparison/compare/` in the flows directory.
