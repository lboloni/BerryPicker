# Accessing external robot pose estimators from BerryPicker: RoboPEPP and HPE

## Status and intent

**A proposal, not implemented** (2026-11-03; revised the same day, after porting Robi's multi-robot pipeline, see
[PLAN-MultiRobot.md](PLAN-MultiRobot.md)).

**The objective is access.** Two published robot pose estimation methods, RoboPEPP and HPE, should be usable from
BerryPicker as one more way of estimating the position of a robot from its images:
- run through exp/runs and flows, on BerryPicker demonstrations;
- producing positions in the representation of the robot (the position class of its robot exp/run, see
  `robot/position_factory.py`).

Everything that uses our visual proprioception models (encoder + MLP regressor) can then use them too: Verify, Compare,
the flows, and later the controllers.

A comparison with our models is one use of this access, not its design driver. All reviewers of the RA-L submission
asked for one (task 8 of `paper-RAS2027-ProprioTransfer/criticism/TOP-TEN.md`; the choice of these two methods is
argued in `paper-RAS2027-ProprioTransfer/related_work/BASELINES.md`). What such a comparison needs on top of the access
is in its own section below.

The design has three parts:
- **Part A:** access on the DREAM real Franka Panda, which both methods support out of the box.
- **Part B:** what is needed to use them on our other robots (AL5D, G1 arm, Inspire hand).
- **Part C (later, optional):** their pretrained encoders as BerryPicker sensor processing.

## The two methods

| | RoboPEPP (Goswami et al., CVPR 2025) | HPE / HoRoPose (Ban et al., ECCV 2024) |
|---|---|---|
| Code | [raktimgg/RoboPEPP](https://github.com/raktimgg/RoboPEPP) | [Oliverbansk/Holistic-Robot-Pose-Estimation](https://github.com/Oliverbansk/Holistic-Robot-Pose-Estimation) |
| Output | Joint angles (joint head), then keypoints by forward kinematics and camera pose by PnP | Joint angles, root rotation and root depth in one forward pass |
| Robots in the repository | Panda only (`configs/panda.yaml`) | Panda, Kuka, Baxter (one config folder per robot) |
| Training data | DREAM synthetic Panda; I-JEPA-style embedding predictive pre-training; optional self-supervised sim-to-real on the unlabeled real sequence | DREAM synthetic; depthnet, then full network; optional self-supervised stage on the unlabeled real sequence |
| Needs at inference | RGB image, camera intrinsics, bounding box (GroundingDINO), URDF | RGB image, camera intrinsics, bounding box, URDF |
| Extra for self-supervision | — | Segmentation masks (the repository ships Panda segmentation weights per camera) and pytorch3d rendering *(check)* |
| Weights | Google Drive: pre-trained encoder and fine-tuned models; `robopepp_ssl_realsense.pt` named in the README | Google Drive model zoo; which checkpoint is which is not listed *(check)* |
| Environment | Python 3.10, CUDA 12.2, pytorch3d, accelerate, wandb (can be disabled), Grounded-SAM-2 | Python 3.9, PyTorch 1.13.1+cu117, pytorch3d 0.7.4, Ubuntu 20.04 |
| Evaluation | `python test.py --config <cfg>`, prints ADD AUC, PCK and joint error for `test_seq` | `python scripts/test.py -e <exp> --dataset <name>` |
| License | Not stated | Not stated |

Neither README describes training on a robot that it does not already support.

## The interface: an external estimator is an exp/run

An external estimator is used like any other BerryPicker component, through an exp/run:
- Family `data/expruns/external_proprioception/`, with one run per method × robot × checkpoint, e.g.
  `robopepp_panda_published.yaml`, `hpe_panda_published.yaml`, `hpe_panda_matched.yaml`.
- Fields:
  - `method` and `checkpoint`;
  - `robot_exp`/`robot_run`, the robot whose positions it predicts (e.g. `robot_franka/panda`);
  - `testing_data`, the same `[pack, demonstration, camera]` triples as our runs;
  - `input-to-notebook: [visual_proprioception/Run_External_Proprioception.ipynb]`.
- **Result:** `predictions.npz` in the run's `data_dir`, with the per-frame joint angles of the method and the same
  dictionary as `predict_positions()` (`targets`, `predictions`, `filtered`, `lengths`), normalized by the robot run.
- **The one change to the existing code:** `predict_positions()` in `visproprio_helper.py` loads `predictions.npz` when
  the run's experiment is `external_proprioception`, instead of running an SP and a regressor. Verify, Compare and the
  collections then treat an external estimator like one of our runs.

Nothing in the interface is specific to a comparison. A flow can run an external estimator on any demonstration whose
robot it supports, and use its positions.

## Architecture in BerryPicker

The methods cannot be imported as Python modules into the BerryPicker venv. They need their own Python, PyTorch and
pytorch3d versions, and CUDA on Linux. They are therefore run as **external tools connected through files**:

```text
BerryPicker demonstration(s)
   -> export: DREAM-format frames (images + per-frame JSON)   [BerryPicker venv]
   -> run: RoboPEPP or HPE in its own conda env                [subprocess]
   -> predictions file: per-frame joint angles (.npz)
   -> import: positions of the robot run, same dict as predict_positions()   [BerryPicker venv]
   -> Verify, Compare, flows: like any visual proprioception run
```

The method repositories are external checkouts, like `ExpRunFlow`. Their code is not vendored into BerryPicker. Their
paths and conda environment names are machine-specific. They go into the system-dependent override in the machine
settings (`Lotzi-ActivityReports/Projects/2-Continuous/CODE-BerryPicker/settings`), not into the repository.

### New components

| Component | Role |
|---|---|
| `src/visual_proprioception/external_export.py` | Writes the frames of a run's `testing_data` (and, for fine-tuning, `training_data`) in DREAM format: `NNNNNN.rgb.jpg` and `NNNNNN.json` with joint angles, keypoints and camera data, plus `_camera_settings.json` with the intrinsics. For demonstrations taken from a DREAM sequence (`source_sequence` and `source_frame_offset` in `_metadata.yaml`), it links the original frames `source_frame_offset + i` of that sequence, at their original resolution and with the sequence's own `_camera_settings.json`, instead of re-encoding our resized copies. |
| `src/visual_proprioception/external_import.py` | Reads the predictions file and turns each frame's joint angles into a position of the robot run, `create_position(exp_robot, angles).to_normalized_vector(exp_robot)`, so it uses exactly the joints, order and limits of our own models. Returns the same dict as `predict_positions()`. |
| `src/visual_proprioception/Run_External_Proprioception.ipynb` | Stage notebook of an external run: export, run the external command through `subprocess`, import, save `predictions.npz` in `exp["data_dir"]`, `exp.done()`. |
| `data/expruns/external_proprioception/` | `_defaults_external_proprioception.yaml` and the runs described above. |
| `predict_positions()` in `visproprio_helper.py` | Loads `predictions.npz` for runs of `external_proprioception` (see above). |
| Driver scripts | Small scripts (`berrypicker_predict.py`) that load a checkpoint, run on an exported folder and write the `.npz`. They are kept in BerryPicker under `src/visual_proprioception/external_drivers/` and copied or symlinked into the checkouts, because the evaluation scripts of both repositories print metrics but do not save per-frame predictions *(check)*. |
| Tests in `src/test/visual_proprioception/` | Round trip: export a small demonstration and read it back; import a hand-written `.npz` and check it against `create_position` with known limits; check that a demonstration with `source_frame_offset` links the right original frames. No GPU, no external code. |

## The robots and their data

Most of what this design needed to know about our data is now known. Robi's multi-robot pipeline has been found and
ported ([PLAN-MultiRobot.md](PLAN-MultiRobot.md)).

**Robot representation.** Each robot has its own package and exp/run family, with the joints, their order and their
limits in the run:
- `robot_franka/panda`: the 7 Panda joints.
- `robot_g1/left_arm`: the 4 joints of the G1 left arm that move in our data.
- `robot_inspire_hand/hand_6dof`: the 6 actuated joints of the hand.

An external estimator predicts joint angles, so it maps onto these runs directly.

**Do not use the `_robi` runs here.** The `_robi` runs reproduce Robi's experiments and do not cover the same joints.
`panda_robi` has joints 1, 2, 3, 4, 5 and 7: Robi's runs never learned joint 6. Our models trained on it therefore
cannot be placed next to a method that predicts joints 1–6. For any joint-by-joint use, our models must be trained on
`robot_franka/panda`.

**The Panda data: one DREAM sequence.**
- Robi's Panda pack (`panda-dream-freeform` on szenes) is the DREAM real **RealSense** sequence
  (`panda-3cam_realsense`, 5944 frames).
- He dropped its first 730 frames and cut the rest into consecutive demonstrations of 500 frames.
- Frame *i* of demonstration *k* (in sorted order) is DREAM frame **730 + 500·k + i**. This was checked against the
  joint angles of all 5000 frames.
- `import_robi_packs.py` records the sequence and the offset in each demonstration's `_metadata.yaml`
  (`source_sequence`, `source_frame_offset`).
- Frames 0–729 and 5730–5943 of the sequence are not used.
- The pack's images are the full 640×480 frames resized to 256×256: the aspect ratio is distorted, but there is no
  crop.
- The original frames, their JSON files and the intrinsics (fx ≈ 615.5, fy ≈ 615.2, cx ≈ 328.3, cy ≈ 251.8, 640×480)
  are in Robi's package, `G1_or_Franka_Panda_convertors/raw_datasets_examples/panda-3cam_realsense/`. They should be
  installed next to the packs.
- The other DREAM real sequences (AK, XK, ORB) are not used by our data and are not in the package.

**The G1 arm and the hand.**
- Only 4 of the 7 G1 arm joints move in our data; the wrist joints are constant.
- Robi's package does not contain the original Isaac Sim frames or the camera intrinsics of the G1 pack; there is only
  one raw sample run. Part B needs them from Robi.

**Still needed: a Linux machine with CUDA** for both repositories. Neither runs on macOS.

## Part A: access on the DREAM real Franka Panda

1. Install both repositories in their own conda environments. Download the published weights. Install the DREAM
   RealSense sequence next to the packs (from Robi's package).
2. **Reproduce the published numbers first**, with each repository's own evaluation script:
   - RoboPEPP: joint error on DREAM synthetic Photo and DR (3.2°, 3.8°) and ADD AUC on RealSense (80.5);
   - HPE: 8.83° L1 joint error on AK after self-supervision.

   If a number is not reproduced, stop and investigate before going on: the driver scripts would only reproduce the
   error.
3. Write the driver scripts, `external_export.py`, `external_import.py` and `Run_External_Proprioception.ipynb`, and
   the runs `robopepp_panda_published.yaml` and `hpe_panda_published.yaml` with `robot_franka/panda`.
4. **Check the access end to end** on `panda-dream-freeform`: the joint angles imported from the predictions file must
   reproduce the repository's own evaluation numbers on the same frames.
5. Optionally, a fine-tuned checkpoint per method (the matched-label regime, below) as one more run.

**Effort:** moderate, mostly installation and data plumbing. It does not depend on Part B.

## Using the access for a comparison

A comparison with our models is fair only if all the following hold.

1. **Same frames, held out from our training.**
   - Robi's published numbers are on `vp_validation`: for the Panda, demonstrations 9 and 4 (DREAM frames 5230–5729 and
     2730–3229). Our sensor processing was trained on these frames, so our numbers there are optimistic.
   - A comparison uses the clean protocol of the cross-robot flow (`robot_variant = "full"`, sorted order,
     `eval_group = "vp_testing"`). Its test demonstrations are 0 and 6: DREAM frames 730–1229 and 3730–4229.
   - The sequence is continuous, so each test chunk borders training chunks in time. This is the same for all
     methods; state it.
2. **Same joints and units.**
   - Report joint angles in degrees (mean absolute error and RMSE per joint), and the normalized RMSE that
     BerryPicker's Compare notebooks plot.
   - The conversion between the two uses the limits of the robot run.
   - For the Panda: joints 1–6, as the DREAM-based papers report, and all 7 joints separately. Our models must be
     trained on `robot_franka/panda`, not `panda_robi`.
   - For the G1: only the 4 moving joints.
3. **Same information, or stated differences.**
   - The external methods use camera intrinsics, a URDF, a bounding box and about 100k synthetic images. Our models use
     none of these, but use labeled real images.
   - In the published regime, the self-supervised checkpoints (e.g. RoboPEPP's `robopepp_ssl_realsense.pt`) were
     adapted, without labels, on this very RealSense sequence, including our test frames. The paper must state both.
4. **Two training regimes for the external methods:**
   - **Published regime:** the published weights (synthetic training, plus self-supervision on the unlabeled target
     sequence where available). This is the regime of their papers.
   - **Matched-label regime:** their network fine-tuned with the joint loss on the same labeled demonstrations our
     regressor sees (the 4 `vp_training` demonstrations, and the few-shot sizes 10–200). This isolates the architecture
     from the data, and avoids the self-supervision on the test frames.
5. Add the external runs and ours to one `visual_proprioception_collections` run (e.g. `comp_external_panda.yaml`).

## Part B: using them on our other robots

The methods can be used on our own robots, but only after we supply what they assume and we do not: a kinematic
model, a calibrated camera, keypoint labels and a large synthetic training set. For each robot:

| Need | AL5D (real) | G1 arm (Isaac Sim) | Inspire hand (Isaac Sim) |
|---|---|---|---|
| **URDF and meshes** | Not in BerryPicker; find a community AL5D URDF or build one from the dimensions in `src/robot/al5d` *(check)* | Unitree publishes the G1 URDF and meshes | In the Unitree G1-with-Inspire description *(check)* |
| **Joint-angle labels** | Our labels are height, distance, heading, wrist angle, wrist rotation, gripper. They must be converted to servo angles with the AL5D inverse kinematics (`src/robot/al5d/helper.py` has the angle/pulse conversion) | In our packs (`robot_g1`); only 4 joints move | In our packs (`robot_inspire_hand`); actuator values, the 12 joints follow from the coupling of the 6 actuators |
| **Original frames** | Our recordings | Not in Robi's package; needed from Robi (the packs hold 256×256 resized copies) | Same |
| **Camera intrinsics** | Calibrate each camera (checkerboard, or ArUco with `sensorprocessing/aruco`) | Exact, from Isaac Sim; needed from Robi | Exact, from Isaac Sim; needed from Robi |
| **Camera-to-base extrinsics** (for keypoints and HPE's root pose) | Calibrate: an ArUco marker at the base, or hand–eye calibration | Exact, from Isaac Sim | Exact, from Isaac Sim |
| **2D keypoints per frame** | Forward kinematics + extrinsics + intrinsics, projected; computed by `external_export.py` | Same, or exported by Isaac Sim | Same |
| **Bounding boxes** | GroundingDINO, as RoboPEPP uses at test time | From the projected keypoints or the sim | Same |
| **Masks** (HPE self-supervision) | SAM 2 (from Grounded-SAM-2) | Isaac Sim segmentation | Isaac Sim segmentation |
| **Synthetic training set** (~100k domain-randomized images, as in DREAM) | Needs a simulated AL5D: e.g. a MuJoCo scene from the URDF, reusing the MuJoCo camera code of `src/robot/wxai` | Isaac Sim Replicator with domain randomization | Isaac Sim Replicator |
| **Code changes in the method** | New robot config: joint count, keypoint list, URDF path, joint limits. HPE has a per-robot config pattern; RoboPEPP has only the Panda and its joint head and keypoint set must be generalized | Same | The hand is a kinematic tree with coupled joints, and both methods assume a single chain *(check)*. Likely a substantial change |

**A cheaper path, the matched-label regime only.**
- Skip the synthetic set and the self-supervision. Start from the published Panda weights and fine-tune on our labeled
  demonstrations with the joint loss alone.
- This needs only the joint-angle labels and a changed joint head. RoboPEPP's joint head does not depend on keypoints or
  PnP; HPE's joint branch probably does not either *(check)*.
- Its limitation: the methods lose the advantage of their synthetic pre-training on the new robot.

**Recommended order:**
1. **G1 arm.** Isaac Sim gives everything exactly (URDF, intrinsics, extrinsics, keypoints, masks, synthetic data), once
   Robi supplies the original frames and the camera parameters.
2. **AL5D, matched-label regime only.** The camera calibration and the joint-angle conversion are moderate work. A
   simulated AL5D for synthetic pre-training is a project of its own.
3. **Inspire hand:** not attempted. The kinematic tree and coupled joints fall outside both methods.

## Part C (later, optional): their encoders as BerryPicker sensor processing

The RoboPEPP encoder is pre-trained on robot images. As a frozen encoder with our MLP regressor, it would enter the
transfer experiments like our own encoders, on any robot, without the method's joint head, keypoints or PnP.

It cannot run in the BerryPicker venv, so it would also work through files:
- a driver writes the encoder's latent for every frame of a set of demonstrations;
- a sensor processing class `sp_external.py` serves these cached latents through the usual `process()` interface.

This covers training and evaluation on recorded demonstrations (all of our flows), but not live use on a robot. Decide
after Part A, once the encoder runs.

## Effort and likelihood of success

Estimated 2026-11-03 for one person working with Claude Code, with a Linux CUDA machine available from the start. The
time is in working days and includes debugging, but not waiting for answers from co-authors. Both numbers are
judgments, not measurements.

| Work package | Working days | Likelihood of success | Main risk |
|---|---|---|---|
| Prerequisites: the data pipeline, the joint-vector position, the DREAM frame mapping | done (2026-11-03) | — | — |
| A: install, data, weights, reproduce the published numbers | 2–4 | ~80% | pytorch3d builds; HPE model zoo checkpoints unlabeled; RoboPEPP SSL weights only for RealSense |
| A: export/import, drivers, notebook, exp/runs, tests, `predict_positions` | 3–5 | ~90% once the reproduction works | Silent errors in the reimplemented inference loop |
| A: published regime on our frames, end-to-end check | ~1 | ~90% | — (our Panda data is the RealSense sequence, for which RoboPEPP has an SSL checkpoint) |
| A: matched-label regime (fine-tune on our demonstrations, few-shot sizes) | 3–5 | ~65% | Training code assumes synthetic DREAM labels; fine-tuning on a few hundred frames may be unstable |
| **Part A total** | **9–15 (2–3 weeks)** | **~75% for the published regime, ~60% for both** | |
| B: G1 arm, matched-label regime | 4–7 after Part A | ~60% | Generalizing RoboPEPP's joint head and HPE's config to the G1 arm; the original frames and intrinsics from Robi |
| B: AL5D, matched-label regime | 5–8 after Part A | ~50% | Camera calibration and the conversion of our labels to servo angles |
| B: G1 arm, published regime (Isaac Sim synthetic set, full training) | 20–30 | ~35% | 100k-image generation, RoboPEPP's embedding pre-training at DREAM scale, many days of GPU time |
| B: Inspire hand | not planned | <15% | Kinematic tree and coupled joints, outside both methods |
| C: RoboPEPP encoder as sensor processing | 2–4 after Part A | ~70% | The encoder's input pipeline (crop, resolution) outside the method's own code |

**For the paper** (about 12 weeks to 2027-01-31, including the holidays; tasks 6, 7 and 9 compete for the same GPU and
the same person):
- Part A in both regimes, plus the G1 arm in the matched-label regime: about **4 weeks** in total.
- Add the AL5D if Part A finishes by mid-December.
- Leave out the G1 published regime and the hand; the paper states them as limitations of the methods.

**Fallback if Part A fails.** If the reproduction fails for one method, use the other only, and cite the published
numbers of the failed one with the caveat that they are on other frames. If both fail, the paper reports the published
numbers from BASELINES.md and argues the comparison qualitatively.

## Risks

- **Silent errors in the drivers.** The repositories do not save per-frame predictions, so the driver scripts must
  reimplement their inference loops. This is the most likely source of silent errors; Part A steps 2 and 4 guard
  against it.
- **pytorch3d builds** are fragile on new CUDA versions. Pin the versions given in each README.
- **No license** is stated for either repository. Research use and reporting results are fine, but do not redistribute
  their code or weights.
- **Joint order and limits.** A mismatch between a method's joint order and the robot run corrupts every number. The
  import goes through `create_position`, so the order and limits are those of the robot run. The joint names of the
  method's output must still be mapped to the run's names; test this on frames with known joint angles (the DREAM JSON
  files give them).
- **Resized frames.** Our Panda pack's images are distorted 256×256 copies. Feeding them to an external method would
  need rescaled intrinsics and would hurt it unfairly. The export therefore links the original frames.
