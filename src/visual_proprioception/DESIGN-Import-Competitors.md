# Importing competitor methods: RoboPEPP and HPE

## Status and intent

**A proposal, not implemented** (2026-11-03). This document plans how two published robot pose estimation methods,
RoboPEPP and HPE, are run from BerryPicker so that their results can be compared with our visual proprioception
models (encoder + MLP regressor). The comparison was requested by all reviewers of the RA-L submission. It is task 8 of
`paper-RAS2027-ProprioTransfer/criticism/TOP-TEN.md`. The choice of these two methods is argued in
`paper-RAS2027-ProprioTransfer/related_work/BASELINES.md`.

The design has two parts:
- **Part A:** run both methods on the DREAM real Franka Panda, which they support out of the box.
- **Part B:** what is needed to run them on our own captured data (AL5D, G1 arm, Inspire hand).

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

## What "comparable" means

The comparison is fair only if all the following hold.

1. **Same frames.** Every method is evaluated on exactly the frames of our test split, not on the full DREAM sequences.
2. **Same units.** The predictions of all methods are reported in two forms:
   - joint angles in degrees: mean absolute error and RMSE per joint;
   - the normalized RMSE used in the paper, which BerryPicker's Compare notebooks plot.

   The conversion uses the joint limits of the robot. DREAM-based papers report joints 1–6 of the Panda, so we report
   joints 1–6 and all moving joints separately.
3. **Same information, or stated differences.** The competitors use camera intrinsics, a URDF, a bounding box and about
   100k synthetic images. Our models use none of these, but use labeled real images. The paper must state this.
4. **Two training regimes for the competitors:**
   - **Published regime:** the published weights (synthetic training, plus self-supervision on the unlabeled target
     sequence where available). This is the regime of their papers.
   - **Matched-label regime:** their network fine-tuned with the joint loss on the same labeled sessions our regressor
     sees (the 4 regressor sessions, and the few-shot sizes 10–200). This isolates the architecture from the data.

## Architecture in BerryPicker

The competitors cannot be imported as Python modules into the BerryPicker venv. They need their own Python, PyTorch
and pytorch3d versions, and CUDA on Linux. They are therefore run as **external tools connected through files**:

```text
BerryPicker demonstration(s)
   -> export: DREAM-format frames (images + per-frame JSON)   [BerryPicker venv]
   -> run: RoboPEPP or HPE in its own conda env                [subprocess]
   -> predictions file: per-frame joint angles (.npz)
   -> import: normalized vectors, same dict as predict_positions()   [BerryPicker venv]
   -> Compare_VisualProprioception.ipynb, next to our models
```

The competitor repositories are external checkouts, like `ExpRunFlow`. Their code is not vendored into BerryPicker.
Their paths and conda environment names are machine-specific. They go into the system-dependent override in the
machine settings (`Lotzi-ActivityReports/Projects/2-Continuous/CODE-BerryPicker/settings`), not into the repository.

### New components

| Component | Role |
|---|---|
| `src/demonstration/import_dream.py` (+ design note) | Converts DREAM real Panda sequences into BerryPicker demonstrations: one demonstration per session, the image and the 7 joint angles per frame. Copy-and-convert, like `import_aloha.py`. Needed only if the existing DREAM import cannot be found (see Prerequisites). |
| `src/visual_proprioception/competitor_export.py` | Writes the frames of the `testing_data` (and, for the matched-label regime, `training_data`) of a run in DREAM format: `NNNNNN.rgb.jpg` and `NNNNNN.json` with joint angles, keypoints and camera data, plus `_camera_settings.json` with the intrinsics. For DREAM-sourced demonstrations it links the original DREAM frames instead of re-encoding them. |
| `src/visual_proprioception/competitor_import.py` | Reads the predictions file, converts joint angles to normalized vectors with the robot's joint limits, and returns the same dict as `predict_positions()` (`targets`, `predictions`, `filtered`, `lengths`). |
| `src/visual_proprioception/Run_Competitor.ipynb` | Stage notebook of a competitor run: export, run the external command through `subprocess`, import, save `predictions.npz` in `exp["data_dir"]`, `exp.done()`. |
| `data/expruns/competitor_proprioception/` | `_defaults_competitor_proprioception.yaml` and one run per method × robot × regime, e.g. `robopepp_panda_published.yaml`, `hpe_panda_published.yaml`, `hpe_panda_matched.yaml`. Fields: `method`, `checkpoint`, `regime`, `testing_data` (the same triples as our runs), `robot_exp`/`robot_run`, `input-to-notebook: [visual_proprioception/Run_Competitor.ipynb]`. Repository paths and conda env names come from the system-dependent override. |
| `predict_positions()` in `visproprio_helper.py` | Smallest change: if the run's experiment is `competitor_proprioception`, load `predictions.npz` instead of running an SP and a regressor. The Compare notebooks then plot competitors like any other run. |
| Wrapper scripts in the competitor checkouts | Small driver scripts (`berrypicker_predict.py`) that load a checkpoint, run on an exported folder and write the `.npz`. They are kept in BerryPicker under `src/visual_proprioception/competitor_drivers/` and copied or symlinked into the checkouts, because the evaluation scripts of both repositories print metrics but do not save per-frame predictions *(check)*. |
| Tests in `src/test/visual_proprioception/` | Round trip: export a small demonstration and read it back; import a hand-written `.npz` and check the normalization against known limits. No GPU, no competitor code. |

## Prerequisites (open questions)

1. **Where is the DREAM, G1 and Inspire data pipeline?** BerryPicker `main` and its branches contain no DREAM, Franka,
   G1 or Inspire code. Also, `Demonstration.get_action()` always returns an AL5D `RobotPosition`, so the paper's Franka,
   G1 and hand runs must have used other code (probably Robert's or Sahara's). It must be found and merged, or
   `import_dream.py` written.
2. **A joint-vector robot position.** Franka, G1 and the hand need a position class with joint names and joint limits
   that provides `to_normalized_vector()` and `from_normalized_vector()`, as `src/robot/wxai/position.py` may already do
   for the WidowX AI *(check)*. The normalization must be the same one used for the paper's results, or the RMSE values
   are not comparable.
3. **How DREAM was divided into "sessions".** This decides which DREAM frames form our test split
   (`criticism/CRITICISM.md` point 13).
4. **A Linux machine with CUDA** for both repositories. Neither runs on macOS.

## Part A: DREAM real Franka Panda

1. Resolve prerequisites 1–3.
2. Install both repositories in their own conda environments. Download the DREAM real sequences and the published
   weights.
3. **Reproduce the published numbers first:**
   - RoboPEPP: joint error on DREAM synthetic Photo and DR (3.2°, 3.8°) and ADD AUC on RealSense (80.5);
   - HPE: 8.83° L1 joint error on AK after self-supervision.

   If a number is not reproduced, stop and investigate before comparing.
4. Check which self-supervised checkpoints exist for the AK, XK, RS and ORB sequences of our split
   (BASELINES.md, open checks).
5. Write the driver scripts and `Run_Competitor.ipynb`; produce `predictions.npz` for our test frames in the published
   regime.
6. Matched-label regime: fine-tune each network with the joint loss only on our 4 regressor sessions, and at the
   few-shot sizes.
7. Add both to a `visual_proprioception_collections` run (e.g. `comp_competitors_panda.yaml`) and to the paper's table
   in degrees and normalized RMSE.

**Effort:** moderate, mostly installation and data plumbing. It does not depend on Part B.

## Part B: using the competitors on our own captured data

The methods can be used on our own robots, but only after we supply what they assume and we do not: a kinematic model,
a calibrated camera, keypoint labels and a large synthetic training set. For each robot:

| Need | AL5D (real) | G1 arm (Isaac Sim) | Inspire hand (Isaac Sim) |
|---|---|---|---|
| **URDF and meshes** | Not in BerryPicker; find a community AL5D URDF or build one from the dimensions in `src/robot/al5d` *(check)* | Unitree publishes the G1 URDF and meshes | In the Unitree G1-with-Inspire description *(check)* |
| **Joint-angle labels** | Our labels are height, distance, heading, wrist angle, wrist rotation, gripper. They must be converted to servo angles with the AL5D inverse kinematics (`src/robot/al5d/helper.py` has the angle/pulse conversion) | Recorded by the simulator | Actuator commands; the 12 joints follow from the coupling of the 6 actuators |
| **Camera intrinsics** | Calibrate each camera (checkerboard, or ArUco with `sensorprocessing/aruco`) | Exact, from Isaac Sim | Exact, from Isaac Sim |
| **Camera-to-base extrinsics** (for keypoints and HPE's root pose) | Calibrate: an ArUco marker at the base, or hand–eye calibration | Exact, from Isaac Sim | Exact, from Isaac Sim |
| **2D keypoints per frame** | Forward kinematics + extrinsics + intrinsics, projected; computed by `competitor_export.py` | Same, or exported by Isaac Sim | Same |
| **Bounding boxes** | GroundingDINO, as RoboPEPP uses at test time | From the projected keypoints or the sim | Same |
| **Masks** (HPE self-supervision) | SAM 2 (from Grounded-SAM-2) | Isaac Sim segmentation | Isaac Sim segmentation |
| **Synthetic training set** (~100k domain-randomized images, as in DREAM) | Needs a simulated AL5D: e.g. a MuJoCo scene from the URDF, reusing the MuJoCo camera code of `src/robot/wxai` | Isaac Sim Replicator with domain randomization | Isaac Sim Replicator |
| **Code changes in the competitor** | New robot config: joint count, keypoint list, URDF path, joint limits. HPE has a per-robot config pattern; RoboPEPP has only the Panda and its joint head and keypoint set must be generalized | Same | The hand is a kinematic tree with coupled joints, and both methods assume a single chain *(check)*. Likely a substantial change |

In addition, `competitor_export.py` must write these labels in DREAM format for our demonstrations, and the
evaluation must use the same test frames as our models.

**A cheaper path, the matched-label regime only.** Skip the synthetic set and the self-supervision. Start from the
published Panda weights and fine-tune on our labeled sessions with the joint loss alone. This needs only the joint-angle
labels and a changed joint head. RoboPEPP's joint head does not depend on keypoints or PnP; HPE's joint branch probably
does not either *(check)*. The comparison is then "their pretrained robot-pose encoder + their head" against "our
encoders + our regressor" on identical labels.
- This is also the comparison that matters for transfer: the RoboPEPP encoder is pre-trained on robot images, so it can
  be wrapped as one more sensor-processing model (`sp_robopepp.py`, frozen, with our MLP regressor) and enter the
  transfer experiments like our encoders.
- Its limitation: the competitors lose the advantage of their synthetic pre-training on the new robot, which reviewers
  may call unfair. The paper must say so.

### Recommended order for our own data

1. **G1 arm.** Isaac Sim gives everything exactly (URDF, intrinsics, extrinsics, keypoints, masks, synthetic data), so
   it is the cheapest robot for the full published regime.
2. **AL5D, matched-label regime only.** The camera calibration and the joint-angle conversion are moderate work. A
   simulated AL5D for synthetic pre-training is a project of its own and is probably not feasible before 2027-01-31.
3. **Inspire hand:** not attempted. The kinematic tree and coupled joints fall outside both methods. The paper states
   this as a limitation of the competitors.

**Effort of Part B:** high. The G1 in the published regime is about as much work as Part A again. The matched-label
regime on the G1 and the AL5D is moderate.

## Effort and likelihood of success

Estimated 2026-11-03 for one person working with Claude Code, with a Linux CUDA machine available from the start.
"Success" means numbers on our test frames that we can defend in the paper. The time is in working days and includes
debugging, but not waiting for answers from co-authors. Both numbers are judgments, not measurements. The largest
uncertainty is prerequisite 1.

| Work package | Working days | Likelihood of success | Main risk |
|---|---|---|---|
| Prerequisites 1–3 (find the DREAM pipeline, joint-vector position, session split) | 1–3 (more if the old pipeline is lost) | High, ~90% | The old normalization cannot be recovered, and the paper's Franka results must be recomputed |
| A: install, data, weights, reproduce the published numbers | 2–4 | ~80% | pytorch3d builds; HPE model zoo checkpoints unlabeled; RoboPEPP SSL weights only for RealSense |
| A: export/import, drivers, notebook, exp/runs, tests, Compare integration | 3–5 | ~90% once the reproduction works | Silent errors in the reimplemented inference loop |
| A: published regime on our test frames | ~1 | ~85% | Our split mixes sequences without SSL checkpoints, so some numbers use the synthetic-only weights |
| A: matched-label regime (fine-tune on our sessions, few-shot sizes) | 3–5 | ~65% | Training code assumes synthetic DREAM labels; fine-tuning on a few hundred frames may be unstable |
| **Part A total** | **10–18 (2–4 weeks)** | **~75% for the published regime, ~60% for both** | |
| B: G1 arm, matched-label regime | 4–7 after Part A | ~60% | Generalizing RoboPEPP's joint head and HPE's config to the G1 arm |
| B: AL5D, matched-label regime | 5–8 after Part A | ~50% | Camera calibration and the conversion of our labels to servo angles |
| B: G1 arm, published regime (Isaac Sim synthetic set, full training) | 20–30 | ~35% | 100k-image generation, RoboPEPP's embedding pre-training at DREAM scale, many days of GPU time |
| B: Inspire hand | not planned | <15% | Kinematic tree and coupled joints, outside both methods |

**Against the deadline.** There are about 12 weeks to 2027-01-31, including the holidays, and the experiments of tasks
6, 7 and 9 compete for the same GPU and the same person.
- **Recommended scope:** Part A in both regimes, plus the G1 arm in the matched-label regime, about **4–5 weeks** in
  total. This answers the reviewers on DREAM (where they asked for it) and adds one of our robots.
- **Add the AL5D** if Part A finishes by mid-December.
- **Leave out the G1 published regime and the hand.** The paper states them as limitations of the competitors.

**Fallback if Part A fails.** If the reproduction in step 3 fails for one method, report the other only, and cite the
published numbers of the failed one with the caveat that they are on other frames. If both fail, the paper reports the
published numbers from BASELINES.md and argues the comparison qualitatively. That is weaker, but it still answers the
"no comparison" criticism in part.

## Risks

- The repositories do not save per-frame predictions, so driver scripts must reimplement their inference loops. This is
  the most likely source of silent errors; the reproduction in Part A step 3 guards against it.
- pytorch3d builds are fragile on new CUDA versions. Pin the versions given in each README.
- No license is stated for either repository. Research use and reporting results are fine, but do not redistribute
  their code or weights.
- A mismatch in joint limits or joint order between the competitors and our normalization corrupts every number. Test
  it on frames with known joint angles.
