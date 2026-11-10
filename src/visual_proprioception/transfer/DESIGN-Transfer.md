# Transfer of visual proprioception across robots

## Status and purpose

**Implemented; the smoke flow has run, the full flow has not** (2026-11-10). The unit tests pass. `Flow_Transfer_Smoke.ipynb` ran all its 40 stages on szenes (MPS) as `Transfer_smoke_00` and wrote the comparison with all 21 methods; its numbers mean nothing. `Flow_Transfer.ipynb` has not been executed. The first smoke run found that the backward pass of `torch.linalg.solve` is not implemented on MPS, so `ridge_fit` solves on the CPU (`solve_on_cpu`).

This directory holds the experiments on transferring visual proprioception from one robot to another, for the RAS 2027 paper (`paper-RAS2027-ProprioTransfer`). The paper's question is which components of a visual proprioception model trained on one robot can be reused on another.

The motivation is a result of the paper itself. A frozen ImageNet ViT with an untrained random projection does about as well on the G1 as the encoder trained on the G1, and better than the encoder trained on the Franka. The trained projection seems to discard what the target robot needs. The proposals that follow from this are in `paper-RAS2027-ProprioTransfer/design/TRANSFER-IMPROVEMENTS.md` (Part I). This directory implements them, together with the baselines a reviewer asks for (`criticism/REVIEW-2026-11-10.md` there).

## The protocol

For each **target robot** t, the regressor is fitted on K labeled frames of t's `vp_training` demonstrations and evaluated on its `vp_testing` demonstrations. `vp_testing` is never used by any encoder. K ranges over `k_shots`, where `None` means all frames.

Every regressor stage reports:
- the trivial baseline `mean` (predicting the mean of the K training targets);
- the per-joint RMSE on the normalized targets and in physical units (degrees; for the AL5D, its own units);
- results for each seed of `seeds`.

The packs are split by `group_chooser_sp_vp_standard`: `sp_training` (40%), `sp_validation`, `vp_training`, `vp_testing` (20%).

## The methods

Each method is named in the results as follows.

| Method | Design | Encoder |
|---|---|---|
| `native-<bb>` | baseline | trained on t itself (upper reference) |
| `single-<bb>-from-<s>` | baseline (the paper) | trained on one other robot s |
| `randproj-vgg19`, `randproj-vit_b_16` | baseline R1 | frozen backbone and a random projection, never trained |
| `multi-<bb>` | D1 | trained on all the other robots, one head each (leave-one-robot-out) |
| `kp-native`, `kp-from-<s>`, `kp-multi` | D2 | VGG19 up to `relu4_4`, 1×1 convolution to K/2 heatmaps, spatial softmax: a latent of image-space keypoints. `kp-multi` is generated only when D1 is also selected |
| `mask-from-<s>` | D3 | the robot masked against the median background of each demonstration; gray background |
| `maskswap-from-<s>` | D3 | the same, trained with random backgrounds (synthetic, and those of robots that are neither source nor target) and color jitter |
| `adabn-vit_b_16-from-<s>`, `adabn-vae-from-<s>` | D4a | the source encoder with its BatchNorm statistics re-estimated on unlabeled frames of t |
| `vae-native`, `vae-target`, `vae-from-<s>` | D4b | Conv-VAEs trained (without labels) on t's `sp_training`, on t's `vp_training`, and on s |
| `meta-ridge` | D5 | a projection meta-trained on the other robots so that a ridge head fitted on K frames predicts the joints; reported with `ridge_meta` (the meta-learned λ) and `mlp` |
| `meta-fomaml` | D5 | the same with first-order MAML (5 inner SGD steps over the projection and a linear head) |

`<bb>` is the backbone: `vgg19` (the encoder of `VGG19ProprioTunedRegression`) or `vit_b_16` (CLS token, then the projection of `sensorprocessing/vit_helper.py`, with BatchNorm).

## The regressors

The regressors are named in `regressors`:
- `mlp`: the 64-64 MLP of the paper, trained for a fixed number of steps (`regressor_steps`), so that every K gets the same budget;
- `ridge`: a closed-form ridge regression with `ridge_lambda`.

Both run on latents standardized with the statistics of the K training frames. `ridge_meta` runs on raw latents with the λ of the MetaRidge SP.

## The code

All the new code is in this directory. The existing code is unchanged, except for the list of flow notebooks in `src/test/test_exprun_notebooks.py`. The SP classes are not registered in `sensorprocessing/sp_factory.py`: `transfer_sp.create_transfer_sp` creates them and falls back to `sp_factory.create_sp` for every other class (the Conv-VAE, the random-projection CNN).

| File | Content |
|---|---|
| `transfer_data.py` | `load_frames` (the frames of `[pack, demo, camera]` items as uint8, with the normalized targets), the D3 masks and backgrounds, `kshot_rows`, `physical_errors` |
| `transfer_models.py` | `TransferEncoder` (one class for the single-source encoders, D1, D2 and D3), `SpatialSoftmax`, `ViTRandomProjection`, `MetaRidgeEncoder`, `ridge_fit`, `fomaml_task`, `adapt_batchnorm`, `RidgeRegressor` |
| `transfer_sp.py` | the SP classes `TransferEncoderSensorProcessing`, `ViTRandomProjectionSensorProcessing`, `MetaRidgeSensorProcessing`, `AdaBNSensorProcessing`; `create_transfer_sp`, `encode_frames` |
| `transfer_flow.py` | `TransferFlow`, the generators of both flows; `FAMILIES`, `DESIGNS`; `list_available_packs` |
| `Train_TransferEncoder.ipynb` | stage: trains a TransferEncoder; each step takes a batch of every robot of `robots` |
| `Train_MetaRidge.ipynb` | stage: meta-trains D5 on episodes over the source robots, from cached frozen features |
| `Train_TransferRegressor.ipynb` | stage: encodes the target, fits the regressors for every K, writes `results.csv` |
| `Compare_Transfer.ipynb` | stage: collects the `results.csv`, writes `summary.csv`, `summary_joints.csv` and `comparison_<target>.png/.pdf` |
| `Flow_Transfer.ipynb` | the full flow |
| `Flow_Transfer_Smoke.ipynb` | the cheap flow that checks the wiring |

Tests: `src/test/visual_proprioception/transfer/test_transfer.py`.

## The exp/run families

| Family | Notebook | Content |
|---|---|---|
| `transfer_datasets` | none | the packs of a machine, one run per machine (`szenes.yaml`) |
| `sensorprocessing_transfer_encoder` | `Train_TransferEncoder` | the TransferEncoders |
| `sensorprocessing_transfer_meta_ridge` | `Train_MetaRidge` | D5 |
| `sensorprocessing_transfer_adabn` | none (adapted at construction) | D4a |
| `sensorprocessing_transfer_randproj_vit` | none (training-free) | R1 for the ViT |
| `visual_proprioception_transfer` | `Train_TransferRegressor` | the regressor stages |
| `visual_proprioception_transfer_compare` | `Compare_Transfer` | the comparison |

The flows also generate runs in two existing families: `sensorprocessing_conv_vae` (trained with `sensorprocessing/Train_Conv_VAE.ipynb`) and `sensorprocessing_random_projection_cnn` (R1 for VGG19, pooled to 4×4). Each family has a checked-in example run. The example runs refer to demonstrations by the names `import_demopack` gives them (`sp_training_00000`, ...), so they work only where the pack has been imported. The flows generate their own runs.

Run names:
- SPs: `te_<backbone>[-mask|-maskswap]_<latent>_<sources>_s<seed>[_for-<target>]`, `vae_<latent>_<pack>-<group>_s<seed>`, `meta-<solver>_<latent>_<sources>_s<seed>`, `adabn-<scope>_<base>_on-<target>`, `rp_..._s<seed>`;
- regressor stages: `vp_<method>_to-<target>_s<seed>`;
- the comparison: `compare_<flow_name>`.

## The flows

`Flow_Transfer.ipynb` runs everything:
- all four robots as sources and targets, the backbones `vgg19` and `vit_b_16`, all the designs;
- latent 128, seeds 0–2, K ∈ {5, 10, 20, 50, 100, 200, all}, 100 epochs, 20000 meta steps.

That is about 200 SP trainings and 450 regressor stages, which needs a CUDA machine. Each SP is trained once and shared by every target that uses it. Each regressor stage encodes its target once and loops over the K values.

`Flow_Transfer_Smoke.ipynb` uses the same code with:
- the G1 as the only target, the Franka and the hand as sources;
- VGG19 only, a 32-dimensional latent, one seed, K ∈ {10, all};
- every 25th frame;
- one epoch for every encoder and Conv-VAE, 10 meta steps and 50 regressor steps.

All designs are included (D4a on the Conv-VAE only). Its numbers mean nothing; it checks that every stage runs and that the comparison is written. Run it first, on szenes:

```
cd src/visual_proprioception/transfer
papermill Flow_Transfer_Smoke.ipynb /tmp/Flow_Transfer_Smoke_out.ipynb
```

The results go to `<flows_path>/Transfer_smoke_00/results/visual_proprioception_transfer_compare/compare_Transfer_smoke_00/`.

## Datasets on other machines

The flows take their packs from `data/expruns/transfer_datasets/<datasets_run>.yaml`, which maps a short robot name to the pack directory, the camera and the robot exp/run. `szenes.yaml` lists the four packs available on szenes on 2026-11-10:

| Name | Pack | Camera | Robot |
|---|---|---|---|
| `al5d` | `random-both-cameras-video` | `dev2` | `robot_al5d/position_controller_00` |
| `panda` | `panda-dream-freeform` | `dev000` | `robot_franka/panda` |
| `g1` | `g1-left-arm-freeform` | `dev000` | `robot_g1/left_arm` |
| `hand` | `inspire-hand-freeform-6dof` | `dev000` | `robot_inspire_hand/hand_6dof` |

To find the datasets on another machine:
1. **Find the pack directory.** `~/.config/BerryPicker/mainsettings.yaml` names the machine's settings file (`configpath`). Its `demopacks_path` is the directory of the packs; in Python it is `Config()["demopacks_path"]`.
2. **List the packs.** Run `list_available_packs()` from `transfer_flow.py`; the flows print it too. It prints, for each pack, the number of demonstrations, the robot named in `<pack>/<pack>.yaml` (`robot_exp`/`robot_run`) and its cameras. Older AL5D packs do not name their robot; it is `robot_al5d/position_controller_00`.
3. **Get the packs that are missing.**
   - The packs of the other robots are made by `src/demonstration/import_robi_packs.py` from Robi's material (see `PLAN-MultiRobot.md`, Part 2). Each pack also has a zip next to it in the `demopacks_path` of szenes; copying the zip and unpacking it in the other machine's `demopacks_path` is enough.
   - The DREAM frames of the Franka are already inside `panda-dream-freeform`; the DREAM sequence itself (`external_proprioception.sequences` in the settings) is not needed.
4. **Write the datasets run** `data/expruns/transfer_datasets/<machine>.yaml` with the same fields, and set `datasets_run = "<machine>"` in the flow. Alternatively, keep `szenes` and put an override in the machine's `experiment_system_dependent_dir` (`transfer_datasets/szenes_sysdep.yaml`), which the exp/run framework merges over the checked-in run.
5. **Check the packs.** Every pack needs 10 demonstrations (or another multiple of 5) for the standard split.

## Known limitations

- **The Conv-VAE training does not take a seed.** `Train_Conv_VAE.ipynb` ignores the `seed` field, so the Conv-VAE runs of different seeds differ only by the randomness of training.
- **D3 is not applied to the Conv-VAE**, whose training reads its frames through its own path.
- **The masks need a fixed camera within a demonstration.** Check them visually on a few frames of each pack before trusting D3, in particular for the real AL5D and the DREAM Franka.
- **The G1 pack has only four moving joints, and the hand's fingers are coupled.** See `PLAN-MultiRobot.md`.
- **Four robots are a small task distribution for D5.** A conclusive test needs many simulated robots (follow-up F1 of the paper's design note).
