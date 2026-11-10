# Multi-robot visual proprioception: porting Robi's extension

## Status and intent

**A plan, mostly implemented: parts 1–3 are on `main`; the full reproduction and the clean protocol remain** (see "Implementation status" at the end) (2026-11-03; revised the same day: one package and family per robot, as for the AL5D and the WidowX). Robi (Robert) extended an older BerryPicker (commit `fb03d17`, 2026-01-09) so that the visual proprioception pipeline trains on demonstrations of other robots: the Unitree G1 left arm, the Franka Panda (DREAM) and the Inspire hand. He also ran cross-robot experiments: the sensor processing (SP) is trained on one robot, and the regressor on another. His material is in `~/Downloads/Robi-SourceCode` (unpacked in `unpacked/`).

This plan has three parts:
1. port the extension cleanly, in the style of the current code;
2. install his converted demonstration packs next to the AL5D packs on this machine;
3. write a new flow that reproduces his cross-robot results.

It resolves prerequisites 1 and 2 of [DESIGN-Import-Competitors.md](DESIGN-Import-Competitors.md): where the G1, DREAM and Inspire pipeline is, and which joint-vector normalization produced the paper's numbers.

## What Robi did (as found in the package)

**Data.** Converter scripts (`G1_or_Franka_Panda_convertors/`) write the joints of the other robot into the **AL5D field names** of `rc-position-target`.

| AL5D field | G1 (`convert_yaml_g1.py`) | Panda (`convert_jsons_to_yaml.py`) |
|---|---|---|
| height | left_elbow_joint | panda_joint3 |
| distance | left_shoulder_pitch_joint | panda_joint1 |
| heading | left_shoulder_yaw_joint | panda_joint2 |
| wrist_angle | left_wrist_pitch_joint | panda_joint4 |
| wrist_rotation | left_wrist_roll_joint | panda_joint5 |
| gripper | left_shoulder_roll_joint | panda_joint7 |
| 7th_number | left_wrist_yaw_joint | panda_joint6 |

The Inspire hand pack uses its own joint names: `L_index_joint`, `L_middle_joint`, `L_ring_joint`, `L_pinky_joint`, `L_thumb_proximal_joint`, `L_thumb_joint`.

**Code.** All the changes are uncommitted edits on `fb03d17`:
- `RobotPosition.FIELDS` in `robot/al5d_position_controller.py` is switched by commenting alternatives in and out (6 AL5D fields, 6 or 12 hand joints).
- The vectors are sized `len(FIELDS)`.
- `output_size` is set by hand: 7 for the arms, 6 for the hand, 13 for the hand with radius.
- The limits are pasted into `robot_al5d/position_controller_00.yaml`, one commented block per robot.

**Consequences that the reproduction must keep, or fix explicitly:**
- **Only 6 joints were learned for the arms.** The 7th entry (`7th_number`) was never filled, so it is a constant 0. Panda joint 6 and the G1 wrist yaw were dropped.
- **The G1 data has only 4 moving joints.** `wrist_angle` and `wrist_rotation` are constant (±0.001), with limits set by hand to ±0.1.
- **The limits are the pack's `dataset_info.yaml` min/max, rounded outward.**
  - Panda MIN: height -1.15, distance -1.2, heading -1.05, wrist_angle -2.9, wrist_rotation -1.05, gripper -0.95.
  - Panda MAX: 0.6, 1.15, 0.25, -1.7, 0.65, 1.9.
  - G1 MIN: height -0.25, distance -0.75, heading -0.3, wrist_angle -0.1, wrist_rotation -0.1, gripper -0.2.
  - G1 MAX: 0.65, 0.1, 0.1, 0.1, 0.1, 0.1.
  - Hand: `~/Downloads/Robi-SourceCode/position_controller_00_Inspire_hand.yaml`. MIN is 0, except thumb_proximal at 0.049; MAX is 1.209, 1.199, 1.006, 1.093, 0.7025, 0.8735 (in the order of `JOINTS` in the run: index, middle, pinky, ring, thumb_proximal, thumb). The data slightly exceeds this MAX (L_index reaches 1.246). His code did not check the limits; ours raises.
- **Splits:** `group_chooser_sp_vp_standard` on 10 demos of 500 frames, camera `dev000` only. His `import_demopack` did not sort the demos, so the split followed the directory order of his Linux machine. The order is recovered from his configs (`sensorprocessing_propriotuned_cnn/vgg19_128.yaml`, `visual_proprioception/vp_ptun_vgg19_128.yaml`) and confirmed against `GT_data.csv` of all three robots (the ground truth is demo `2025_10_26__03_27_03` followed by `2025_10_24__20_17_15`, matching to 1e-7):
  - `2025_10_26__03_27_03`, `2025_10_24__20_17_15`, `2025_10_25__02_58_05`, `2025_10_25__18_59_04` (SP training; the first two also VP validation);
  - `2025_10_24__19_30_56`, `2025_10_25__21_34_30` (SP validation and VP training), `2025_10_24__02_18_43`, `2025_10_24__01_51_54` (VP training);
  - `2025_10_24__01_34_46`, `2025_10_25__05_32_59` (testing, never used; their order is unknown and does not matter).

  The same demo names are used in every pack, so one order reproduces the split for all robots. The current `import_demopack` sorts the demos; the reproduction passes the order explicitly (`group_chooser_fixed_order` in `demopack.py`).
  - The reported numbers are on VP validation, which is a subset of SP training. In-domain numbers are therefore optimistic.
  - The test demos were never used.
- **Meaning of a cross-robot result** `X_regress_Y`: the SP is trained on X, the regressor is trained on Y's `vp_training` and evaluated on Y's `vp_validation`. No code states this. It was inferred from the ground truth in `GT_data.csv`, and the runs were probably made by re-running X's flow directory with pack Y and `exist-ok`.
- **The metric is per-joint RMSE on the normalized targets**, written to `msecomparison_values.txt` (the files are appended on every rerun).

**Target numbers.** The `all_cnns_7DOF` results use 300 SP and 1000 VP epochs. The values are mean RMSE over the 6 learned joints; for G1 targets, two of these joints are constant.

| SP robot → VP robot | VAE128 | VAE256 | VGG128 | VGG256 | RN128 | RN256 |
|---|---|---|---|---|---|---|
| Panda → Panda | .128 | .159 | .105 | .109 | .167 | .191 |
| G1 → G1 | .049 | .039 | .016 | .016 | .061 | .055 |
| Panda → G1 | .043 | .045 | .079 | .067 | .114 | .112 |
| G1 → Panda | .177 | .173 | .191 | .180 | .166 | .183 |
| Hand → Hand | .064 | .051 | .012 | .014 | .135 | .116 |

The ViT results (`~/Downloads/Robi-SourceCode/vits`) used a different normalization and are out of scope.

## Part 1: clean port

The principle: **the robot is data, not code.** Each robot is a robot exp/run that names its joints and their limits, in a package and family of its own, like the AL5D and the WidowX. The pack says which robot recorded it. The output size follows from the robot. Nothing is switched by commenting code in and out.

### 1.1 Robots as packages, in the pattern of the AL5D and the WidowX

Each robot that BerryPicker uses already has its own package (`robot/al5d`, `robot/widowx`, `robot/wxai`), its own exp/run family (`robot_al5d`, `robot_widowx`, `robot_wxai`) with its own `robot_name` and `controller_type: position_controller`, and its own position class (`RobotPosition`, `WidowXPose`, `WXAIPose`). Related robots share code by subclassing: `WXAIPose(WidowXPose)` changes only `ROBOT_NAME`. The new robots follow the same pattern, so that a robot that is later controlled (the Franka) gets its `PositionController` and `SimulatedPositionController` in the same package, next to its position class, as for the WidowX AI.

- **A shared base class** `JointPosition` in `src/robot/joint_position.py`, for robots whose positions are vectors of named joints. Modeled on `WidowXPose`:
  - `ROBOT_NAME` is a class attribute, checked against `exp["robot_name"]` in the constructor.
  - The joints come from the exp (`exp["JOINTS"]`, an ordered list), not from a class attribute, so that a robot can have reduced runs (e.g. only the moving joints).
  - Limits are `POS_MIN`/`POS_MAX`/`POS_DEFAULT` dicts keyed by joint name, the same keys as the AL5D.
  - It has the interface that the loaders and notebooks use from `RobotPosition`: `to_normalized_vector(exp)`, `from_normalized_vector(exp, v)`, `from_vector`, `limit`, `empirical_distance`, `__getitem__` and `__str__`. The classmethods construct `cls`, so subclasses return their own type.
  - Normalization is linear from `[POS_MIN, POS_MAX]` to `[0, 1]` (the formula Robi used). It raises when a value is outside the limits, as the AL5D class does, unless the run sets `enforce_limits: false` (only for reproducing runs that did not check the limits).
- **One package per robot**, each with a thin subclass:

  | Package | Position class | Family | `robot_name` |
  |---|---|---|---|
  | `src/robot/franka/` | `FrankaPosition(JointPosition)` | `robot_franka` | `franka` |
  | `src/robot/g1/` | `G1ArmPosition(JointPosition)` | `robot_g1` | `g1_arm` |
  | `src/robot/inspire_hand/` | `InspireHandPosition(JointPosition)` | `robot_inspire_hand` | `inspire_hand` |

  A Franka controller is not part of this plan. When it comes, a joint-space controller uses `FrankaPosition`; a Cartesian one would get a pose class like `WidowXPose`.

### 1.2 Robot exp/runs

Each family has `_defaults_robot_<robot>.yaml` (`robot_name`, `controller_type: position_controller`, `action_key: joint-position`, `enforce_limits: true`) and these runs:
- `robot_franka/panda.yaml`: `panda_joint1..7`; `robot_franka/panda_robi.yaml`: Robi's 6 joints (the table above, in AL5D field order, without `7th_number`), with his limits.
- `robot_g1/left_arm.yaml`: the 4 moving joints of the left arm; `robot_g1/left_arm_robi.yaml`: Robi's 6 joints and limits.
- `robot_inspire_hand/hand_6dof.yaml`: the 6 actuated joints; `robot_inspire_hand/hand_6dof_robi.yaml`: his limits, which do not contain all of the data, with `enforce_limits: false` (as in his code, such values normalize slightly outside `[0, 1]`).

The full runs use limits from the data (`dataset_info` min/max rounded outward). Joints that do not move (the G1 wrist joints) are left out of `JOINTS` and listed in a comment, so that a degenerate limit never reaches the normalization.

### 1.3 Robot selection: one factory, one action key

- Add `src/robot/position_factory.py`: `create_position(exp_robot, values)` dispatches on `exp_robot["robot_name"]`: `al5d` gives `RobotPosition`, `franka` `FrankaPosition`, `g1_arm` `G1ArmPosition`, `inspire_hand` `InspireHandPosition`. It also gives `position_fields(exp_robot)` and `robot_output_size(exp_robot)`.
- The WidowX and the WidowX AI are not in the factory yet: their demonstrations record commands (`widowx-command`, `wxai-command`: a pose plus gripper fields), which would need a small adapter. Not needed for Robi's results.
- `Demonstration.get_action(i, type, exp)`: when `exp` is given, return `create_position(exp, self.actions[i][type])`; otherwise return the raw entry. This removes the AL5D import from `demonstration.py`.
- The loaders read the action key from the robot instead of hard-coding `"rc-position-target"`: `action_key(exp_robot)`, i.e. `exp_robot.get("action_key", "rc-position-target")`. This follows the precedent of `robot_controller/training_data.py` (`action_type`); the AL5D runs need no change. The places changed:
  - `visual_proprioception/visproprio_helper.py:159, 280`
  - `sensorprocessing/helper_training_data.py:156, 360`
  - `sensorprocessing/multiview_data.py:187`
  - `behavior_cloning/bc_trainingdata.py:96` is left alone; behavior cloning stays AL5D-only for now.

### 1.4 Output size from the robot

- The flow generators set `output_size = len(position fields of exp_robot)`. Add a small helper `robot_output_size(exp_robot)` in the factory module.
- The YAML defaults keep `output_size: 6` (AL5D), so existing runs are unchanged.
- The sensor processing classes that do `exp.get("output_size", 6)` are left alone. The generated runs always set the field.

### 1.5 Robot of a run

- The VP and propriotuned SP runs already carry `robot_exp`/`robot_run`. The generators set them from the pack instead of inheriting the AL5D default.
- The pack YAML (`<pack>/<pack>.yaml`, copied by `import_demopack` to `expruns/demonstration/<pack>.yaml`) gets `robot_exp`/`robot_run`. A flow reads the robot from `Config().get_experiment("demonstration", pack)`.
- A run has exactly one robot. Mixed-robot training sets (several output widths in one regressor) are out of scope.

### 1.6 Notebooks

- `Train_VisualProprioception.ipynb` and `Train_ProprioTuned_CNN.ipynb` copy the family of `exp["robot_exp"]`, not a hard-coded `"robot_al5d"`.
- `Verify_VisualProprioception.ipynb` and `Compare_VisualProprioception.ipynb`:
  - The titles come from the position fields of `exp_robot` instead of the hard-coded AL5D list.
  - The subplot grid is sized from the field count.
  - The stray `POS_MAX` cell in Verify is removed.
  - Compare uses the one robot of the collection (`exp_robot`, from its `robot_exp`/`robot_run`) for all the compared runs. (As implemented, it does not check that the runs themselves have that robot.)
  - Compare also writes the per-joint RMSE values to a CSV that is overwritten, not appended.

### 1.7 Tests (`src/test/robot/`, `src/test/demonstration/`)

- `JointPosition`: normalized round trip; a value out of limits raises.
- The factory dispatches AL5D and joints runs.
- `get_action` on a tiny demonstration with `joint-position` actions returns a `JointPosition`.
- The existing AL5D tests and the notebook-hygiene tests (`test_exprun_notebooks.py`) pass unchanged.

## Part 2: demonstration packs next to the AL5D packs

Robi's packs cannot be used as they are:
- The joint names are AL5D names.
- `_metadata.yaml` lists 120 cameras, but only `dev000` exists.
- `_annotation.yaml` is a stale AL5D copy (3661 entries for 500 steps).
- The pack YAML says `name: freeform`.

So they are converted once, by a script in the style of `import_aloha.py`: `src/demonstration/import_robi_packs.py`, with a short design note. It copies and converts; it never touches the source.

| Source (`~/Downloads/Robi-SourceCode/unpacked/…`) | Target pack in `BerryPicker-Demopacks/` | Robot run |
|---|---|---|
| `Unitree_G1/freeform_smooth` | `g1-left-arm-freeform` | `robot_g1/left_arm` |
| `Franka_Panda/freeform_dream_ds` | `panda-dream-freeform` | `robot_franka/panda` |
| `all_cnns_6DOF_Inspire_Hand_and_Demopack/BerryPicker-Demopacks/freeform_Hand_6DOF` | `inspire-hand-freeform-6dof` | `robot_inspire_hand/hand_6dof` |

For each demonstration, the script:
- maps the AL5D-named fields back to the native joint names (the table above), into the action key `joint-position`;
- writes `_metadata.yaml` with `cameras: [dev000]`, `maxsteps: 500` and `stored_as_images: True`;
- writes a fresh `_annotation.yaml`;
- for the Panda, records in `_metadata.yaml` the DREAM sequence the frames come from and the index of the demonstration's first frame in it (`source_sequence: DREAM panda-3cam_realsense`, `source_frame_offset: 730 + 500·k`; see [DESIGN-Import-Competitors.md](DESIGN-Import-Competitors.md));
- keeps the JPEG frames as they are, because re-encoding to video would change the inputs of the reproduction.

For each pack, the script:
- writes `<pack>.yaml` with the camera (256×256, 10 fps) and `robot_exp`/`robot_run`;
- keeps `dataset_info.yaml`, regenerated from the converted actions, which fixes the stale hand file;
- writes a `README.md` with the provenance;
- writes a zip next to the pack, as for the existing packs.

The `demonstration/<pack>.yaml` files describe the cameras, like `random-both-cameras-video.yaml`. (As implemented, they are not checked in for the three new packs: they are in the pack, and `import_demopack` copies them into the workspace.)

Not imported:
- `freeform_dummy_7`: an AL5D pack with a dummy 7th field (`22222`), and its `_action.yaml` does not parse.
- The raw G1 and Panda examples: they are already covered by the converted packs.

## Part 3: the cross-robot flow

Add `src/visual_proprioception/Flow_VisualProprioception_CrossRobot.ipynb`, a new-style flow (`setup_flow`, generated exp/runs, `flow_entry`, `run_flow`, `display_flow_report`). It keeps the generators of `Flow_VisualProprioception.ipynb`, but takes the robot from the pack.

**Parameters:**
- `flow_name`;
- `packs` (default: the G1, Panda and hand packs above);
- `pairs`, a list of `(sp_pack, vp_pack)`. The default is the four arm pairs plus hand→hand;
- `sp_types` (`conv_vae`, `vgg19`, `resnet50`), `latent_sizes` ([128, 256]), `epochs_sp` (300), `epochs_vp` (1000);
- `camera` (`dev000`);
- `robot_variant` (`robi` for the reproduction limits and joints, `full` for all moving joints);
- `demo_order` (Robi's directory order for the reproduction, or `None` for the sorted order);
- `eval_group` (`vp_validation` reproduces Robi's protocol; `vp_testing` is the clean held-out evaluation);
- `creation_style`.

**Queue:**
1. `import_demopack(pack, group_chooser_sp_vp_standard)` for every pack.
2. **SP trainings, once per pack that appears as an `sp_pack`**, for each SP type and latent size. The propriotuned CNNs get the pack's robot and its `output_size`. The run names carry the pack, e.g. `sp_vgg19_128_g1`.
3. **One VP regressor per pair × SP**: `sp_experiment`/`sp_run` point to the SP of `sp_pack`, `training_data` comes from `vp_pack`'s `vp_training`, `validation_data` from `vp_pack`'s `eval_group`, and `robot_exp`/`robot_run` and `output_size` come from `vp_pack`'s robot. The run names carry both packs, `vp_<sp_type>_<latent>_<sp_pack>_to_<vp_pack>`, e.g. `vp_vgg19_128_g1_to_panda`. The regressor sees only the latent of the SP, so a different output width on the two sides is no problem.
4. **One comparison per pair** in `visual_proprioception_collections` (`crossrobot_<sp>_to_<vp>`), comparing all SP types of that pair. This mirrors Robi's four result directories plus the hand one.
5. **A final summary stage**: a small notebook `Summarize_CrossRobot.ipynb` (or a cell of the flow) writes one table in the shape of the target table above, with Robi's numbers next to ours.

Unlike Robi's runs, no flow directory is reused: each SP is trained once and referenced by every pair that needs it, inside one workspace.

Document the flow in `data/expruns/DESIGN-Flows.md` (Current flows), and add a checked-in collection exprun if the notebook-hygiene test requires one.

## Order of work and verification

1. **Part 1, AL5D unchanged.** Run the tests, then the minimal AL5D flow, which must still pass:
   ```
   papermill Flow_VisualProprioception.ipynb … -p epochs_sp 1 -p epochs_vp 1 (VGG only)
   ```
   This is the smoke test that already passed on szenes on 2026-11-03.
2. **Part 2.** Run the conversion script, then:
   - load each new pack with `Demonstration` and check `get_action(i, "joint-position", exp_robot).to_normalized_vector()` on all frames;
   - check that `dataset_info` matches Robi's ranges;
   - spot-check the values against his `GT_data.csv`. For example, the first G1 elbow value 0.368 normalizes to 0.6867 with the `robi` limits.
3. **Part 3, smoke.** Run the cross-robot flow with 1 epoch, VGG19-128, and pairs G1→Panda and hand→hand, to check the wiring, not the numbers.
4. **Part 3, reproduction.** Run with `robot_variant=robi`, `eval_group=vp_validation`, 300/1000 epochs and all SP types, on a CUDA machine (about 6 SPs × 3 packs plus 30 regressors; too slow for szenes). Accept the reproduction if the ranking of SP types per pair matches and the mean RMSE values are within about ±0.02 of the target table. Differences in random seed and library version make an exact match unlikely.
5. **Then the clean protocol.** Run `robot_variant=full` and `eval_group=vp_testing`. These are the numbers to put in the paper, with Robi's protocol reported as the earlier result.

## Implementation status (2026-11-03; committed on `main`: `a745de0`, `02c48e6`, `611be3a`, `df8dbf8`)

Done:
- **Part 1.**
  - `robot/joint_position.py` (`JointPosition`), and the packages `robot/franka`, `robot/g1` and `robot/inspire_hand`.
  - `robot/position_factory.py`; `Demonstration.get_action` dispatches through it.
  - The action key now comes from the robot in the visual proprioception and sensor processing loaders.
  - The families `robot_franka`, `robot_g1` and `robot_inspire_hand`, each with a full run and a `_robi` run.
  - The notebooks: Train VP, Train ProprioTuned CNN, Verify, Compare.
  - Tests in `src/test/robot/joint_position/`.
- **Part 2.**
  - `src/demonstration/import_robi_packs.py`.
  - The packs `g1-left-arm-freeform`, `panda-dream-freeform` and `inspire-hand-freeform-6dof` (with zips) in `BerryPicker-Demopacks` on szenes.
  - With the `_robi` runs and Robi's demonstration order, the normalized targets of all three packs equal his `GT_data.csv` (G1 and Panda checked to 2e-3; the hand matches to 6e-8).
- **Part 3.**
  - `Flow_VisualProprioception_CrossRobot.ipynb`, and `group_chooser_fixed_order` in `demopack.py`.
  - Smoke test passed on szenes: 1 epoch, Conv-VAE and VGG19 at 128, pairs G1→Panda and hand→hand, 14 stages, no errors.
- **Line endings.** The repository was converted to LF (`.gitattributes`).

Found on the way:
- `Train_Conv_VAE.ipynb` produced no `model.pth` when `epochs` was not a multiple of `save_period`. It copied the trainer's checkpoint of the last epoch, which the trainer writes only every `save_period` epochs. It then only printed a message, so the failure surfaced in the next stage. **Fixed:** the notebook saves the model at the end of the training itself, in the trainer's checkpoint format, and no longer depends on the checkpoints.

Not done:
- The full reproduction (step 4) and the clean protocol (step 5). They need a CUDA machine: 300/1000 epochs, 3 SP types × 2 latent sizes × 3 packs, and 30 regressors.

## Open questions

- Did the paper (ProprioTransfer) use Robi's numbers? If so, the leakage in the in-domain protocol and the dropped 7th joint must be addressed there (see `paper-RAS2027-ProprioTransfer/criticism`).
- The G1 data is "smooth" Isaac Sim motion with only 4 moving joints. Is a richer G1 pack available?
- The hand radius variant (12 joints + `rad_est_mm`, pack `freeform_touch_hand_real_tennis_only`) is not in the package. Is it needed?
