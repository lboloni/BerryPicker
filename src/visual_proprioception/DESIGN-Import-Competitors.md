# Accessing external robot pose estimators from BerryPicker: RoboPEPP and HPE

## Status and intent

**The BerryPicker side is implemented; the methods themselves have not been run yet** (2026-11-03). The interface,
the export, the import, the stage notebook, the flow and the drivers of both methods are in place, and the whole
path runs on szenes with the ground-truth reference method `dream_ground_truth`. RoboPEPP and HPE need a Linux
machine with CUDA (see "State of the implementation" at the end). The design was revised on 2026-11-03, after porting
Robi's multi-robot pipeline ([PLAN-MultiRobot.md](PLAN-MultiRobot.md)).

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
- Family `data/expruns/external_proprioception/`, with one run per method × robot × checkpoint:
  `robopepp_panda_published.yaml`, `hpe_panda_published.yaml`, and the reference `dream_ground_truth_panda.yaml`.
- Fields:
  - `method` (the driver `external_drivers/<method>_predict.py`) and `checkpoint` (a file or folder in the method's
    checkpoint directory);
  - `robot_exp`/`robot_run`, the robot whose positions it predicts: `robot_franka/panda_6dof` for RoboPEPP (joints
    1–6, the only ones it predicts), `robot_franka/panda` for HPE (joints 1–7);
  - `validation_data`, the `[pack, demonstration, camera]` triples it runs on. It is the field our runs are evaluated
    on, so Verify and Compare need no change. The checked-in runs do not set it (the family default is `[]`): they
    are templates, see below;
  - `driver_args`, further arguments of the driver;
  - `input-to-notebook: [visual_proprioception/Run_External_Proprioception.ipynb,
    visual_proprioception/Verify_VisualProprioception.ipynb]`.
- **The checked-in runs only run in the flow.** Which demonstrations a method runs on depends on the imported pack
  and its split, so `Flow_External_Proprioception.ipynb` writes, for every checked-in run, the run
  `<run>_<eval_group>` (e.g. `robopepp_panda_published_vp_testing`) with `validation_data` filled in, and queues that
  one. `Run_External_Proprioception.ipynb` run directly on a checked-in run, such as its default
  `robopepp_panda_published`, stops at the assert of `source_sequence()`, because an empty `validation_data` has no
  source sequence. To run the notebook by hand, point it at the workspace of a flow (`expruns_path`, `results_path`)
  and at a run the flow generated.
- **Result:** `angles_validation_data.npz` in the run's `data_dir`: the per-frame joint angles of the method (radians),
  their joint names and the original frame files.
- **The one change to the existing code:** `predict_positions()` in `visproprio_helper.py` reads this file for runs of
  `external_proprioception`, instead of running an SP and a regressor. It maps the angles by name onto the joints of
  the robot run it is given, and applies the run's position filter as for our models. Verify, Compare and the
  collections then treat an external estimator like one of our runs.
- **Mapping by joint name** lets one comparison hold methods that predict different joints. A collection on
  `robot_franka/panda_6dof` compares RoboPEPP (joints 1–6) and HPE (joints 1–7, plus the finger) on joints 1–6.
- Predicted positions outside the limits of the robot run normalize outside [0, 1], as the outputs of our regressors
  can (`to_normalized_vector(exp, enforce_limits=False)`). The targets are checked against the limits as usual.

Nothing in the interface is specific to a comparison. A flow can run an external estimator on any demonstration whose
robot and source sequence it supports, and use its positions.

## Architecture in BerryPicker

The methods cannot be imported as Python modules into the BerryPicker venv. They need their own Python, PyTorch and
pytorch3d versions, and CUDA on Linux. They are therefore run as **external tools connected through files**:

```text
BerryPicker demonstration(s)
   -> export: DREAM-format folder, links to the original frames   [BerryPicker venv]
   -> run: the driver of the method, with the Python of its env     [subprocess]
   -> angles file: per-frame joint angles (.npz)
   -> predict_positions(): positions of the robot run               [BerryPicker venv]
   -> Verify, Compare, flows: like any visual proprioception run
```

The method repositories are external checkouts, like `ExpRunFlow`; their code is not vendored into BerryPicker. Their
paths, the Python of their environments, their checkpoint directories and the machine-specific driver arguments are in
the machine settings (`Lotzi-ActivityReports/Projects/2-Continuous/CODE-BerryPicker/settings/settings-<machine>.yaml`),
under `external_proprioception`, next to the directories of the source sequences. `src/install/settings-sample.yaml`
has a commented template:

```yaml
external_proprioception:
  sequences:
    "DREAM panda-3cam_realsense": <the DREAM-format directory of the sequence>
  dream_ground_truth:        # every method of a run needs a section, also the reference
    code_dir: ''
    python: <the Python of the BerryPicker venv>
    checkpoint_dir: ''
  robopepp:
    code_dir: <the RoboPEPP checkout>
    python: <the Python of its conda env>
    checkpoint_dir: <its checkpoints>
    driver_args:
      jepa-path: <jepa_joints-ep200.pth.tar>
      bbox-dir: <the GroundingDINO boxes of the sequence>
  hpe:
    code_dir: <the HPE checkout>
    python: <the Python of its conda env>
    checkpoint_dir: <the model zoo: one experiment folder per model>
```

### Components

| Component | Role |
|---|---|
| `src/visual_proprioception/external_proprioception.py` | `export_dream`: writes the frames of a run's `validation_data` as a DREAM-format folder, named after the sequence (both methods recognize the real DREAM sequences by the folder name). The demonstrations must come from a source sequence (`source_sequence`, `source_frame_offset` in `_metadata.yaml`). The files `NNNNNN.rgb.jpg` and `NNNNNN.json` are numbered in the order of the demonstrations and are symbolic links to the original frames, at their original resolution, with the sequence's `_camera_settings.json`. A driver finds the original frame by resolving the link, e.g. for RoboPEPP's per-frame bounding boxes. `run_external`: runs the driver with the Python of the method's environment. `load_external_predictions`: reads the angles back as positions of a robot run (see above). |
| `src/visual_proprioception/external_drivers/robopepp_predict.py` | Loads a RoboPEPP checkpoint and predicts joints 1–6 for every exported frame. It reimplements the preprocessing of RoboPEPP's test dataset, because `datasets/dream_ssl.py` cannot be imported without GroundingDINO. The steps are: the precomputed GroundingDINO box, extended as for the RealSense sequence; crop; resize the longer side to 224; edge-pad to 224×224; ImageNet normalization. The joints come from the encoder, the predictor and the joint head, denormalized with the joint statistics of `test.py`; the keypoint heads and the PnP are not needed. |
| `src/visual_proprioception/external_drivers/hpe_predict.py` | Loads an HPE model (an experiment folder of its model zoo) and predicts joints 1–7 and the finger, with HPE's own `make_cfg`, `DreamDataset` and model, and the depth scale of `scripts/test.py`. As in HPE's evaluation, the image crop comes from the ground-truth keypoints in the frames' JSON files. |
| `src/visual_proprioception/external_drivers/dream_ground_truth_predict.py` | The reference method: the ground-truth joint angles of the frames' JSON files. It needs no model and no GPU, so it checks the whole path on any machine, and its error must be zero. |
| `src/visual_proprioception/Run_External_Proprioception.ipynb` | Stage notebook of an external run: export, run the driver, print the RMSE per position field (normalized units), `exp.done()`. |
| `src/visual_proprioception/Flow_External_Proprioception.ipynb` | Imports a pack and runs, verifies and compares the given external runs on one group of its demonstrations (by default `vp_testing`), on the joints of the robot run `compare_robot`. |
| `data/expruns/external_proprioception/` | The family (see above). |
| `data/expruns/robot_franka/panda_6dof.yaml` | Panda joints 1–6. |
| `predict_positions()` in `visproprio_helper.py` | Reads the angles of external runs (see above). |
| `src/test/visual_proprioception/test_external_proprioception.py` | The export links the right original frames in the order of the demonstrations; the angles are mapped by name onto the robot run (an extra joint, another order, values outside the limits); `predict_positions()` reads an external run; the driver command. No GPU, no external code. |

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
3. *(Done on the BerryPicker side, 2026-11-03.)* The drivers, `external_proprioception.py`,
   `Run_External_Proprioception.ipynb`, `Flow_External_Proprioception.ipynb`, and the runs
   `robopepp_panda_published.yaml` (`robot_franka/panda_6dof`) and `hpe_panda_published.yaml` (`robot_franka/panda`).
   - RoboPEPP additionally needs the GroundingDINO boxes of the sequence: `bbox_grounding_dino/create_bbox.py`, once.
   - HPE needs its `data/deps/panda-description` and `models/hrnet_w32-36af842e_roc.pth` in its checkout.
4. **Check the access end to end** on `panda-dream-freeform`, with `Flow_External_Proprioception.ipynb`
   (`external_runs = ["dream_ground_truth_panda", "robopepp_panda_published", "hpe_panda_published"]`). Run the
   repository's own evaluation on the same frames: export them, then point its `test.py` at the exported folder. The
   joint errors in degrees of the two must agree; this checks the drivers' reimplemented preprocessing.
5. Optionally, a fine-tuned checkpoint per method (the matched-label regime, below) as one more run.

**Effort:** moderate, mostly installation and data plumbing. It does not depend on Part B.

## Using the access for a comparison

A comparison with our models is fair only if all the following hold.

1. **Same frames, held out from our training.**
   - Robi's published numbers are on `vp_validation`: for the Panda, demonstrations 9 and 4 (DREAM frames 5230–5729 and
     2730–3229). Our sensor processing was trained on these frames, so our numbers there are optimistic.
   - A comparison uses the clean protocol of the cross-robot flow (`robot_variant = "full"`, sorted order,
     `eval_group = "vp_testing"`). In the sorted order, its test demonstrations are the last two, 8 and 9: DREAM
     frames 4730–5729. (In Robi's directory order, the test demonstrations would be 0 and 6, DREAM 730–1229 and
     3730–4229, but his runs never used them.) The exported links of a run show which frames were used.
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
5. Add the external runs and ours to one `visual_proprioception_collections` run (the flow generates it as `external_<pack>_<eval_group>`; no such run is checked in).

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
| **2D keypoints per frame** | Forward kinematics + extrinsics + intrinsics, projected; computed by a new `external_export.py` (proposed; the existing export is `export_dream` in `external_proprioception.py`) | Same, or exported by Isaac Sim | Same |
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

## State of the implementation (2026-11-03)

Done and tested on szenes:
- The interface, the export, the import, `predict_positions()`, the stage notebook and the flow.
- The unit tests.
- The whole path with `dream_ground_truth`, on the held-out demonstrations of `panda-dream-freeform`.
- The DREAM RealSense sequence is installed at
  `~/Documents/Develop/Data/BerryPicker-data-external/DREAM/panda-3cam_realsense`, from Robi's package.
- The RoboPEPP and HPE checkouts are cloned next to BerryPicker (`~/Documents/Develop/Github/RoboPEPP`,
  `~/Documents/Develop/Github/Holistic-Robot-Pose-Estimation`).

Written but not run (they need CUDA):
- `robopepp_predict.py` and `hpe_predict.py`, written against the code of the two repositories (RoboPEPP c4a43d9, HPE
  77cd316).

Next, on a Linux CUDA machine:
1. Install both environments, the weights, RoboPEPP's I-JEPA encoder, and HPE's URDF and HRNet weights.
2. Make RoboPEPP's GroundingDINO boxes.
3. Add the `external_proprioception` section to that machine's settings.
4. Run Part A steps 2 and 4.

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
