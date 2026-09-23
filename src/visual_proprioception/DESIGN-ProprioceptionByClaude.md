# Visual proprioception by Claude: a by-eye baseline

## Purpose and scope

Visual proprioception in BerryPicker is normally done by a trained model: an
image encoder plus a regressor that maps camera frames to the robot's
`rc-position-target`. This document records a different experiment. Claude, a
multimodal language model, looked at the camera frames of one demonstration and
estimated the AL5D pose at every timestep. There was no model training, and
Claude did not see that demonstration's actions.

The result is a baseline that needs no training data from the target
demonstration. It also shows which pose values can be read from these two camera
views and which cannot. The per-value error uses the same normalized RMSE as
[Compare_VisualProprioception.ipynb](Compare_VisualProprioception.ipynb), so the
numbers can sit directly beside learned models.

This was a single run on a single demonstration. It is a measurement record, not
a component of the codebase. Nothing in `src/` was changed to produce it.

## Data

- **Demopack:** `BerryPicker-Demopacks/random-both-cameras-video`, 10
  demonstrations of the AL5D moving randomly, recorded on 2025-03-08.
- **Target demonstration:** `2025_03_08__14_19_12`. It has 446 timesteps and
  two cameras, `dev2` and `dev3`, each 256×256 at 10 fps, stored as video
  (`video_dev2.mp4`, `video_dev3.mp4`).
- **Reference demonstration:** `2025_03_08__14_15_53`, with 387 timesteps. Its
  frames and its `_action.yaml` were used only for calibration.

**Blindness rule:** the target's `_action.yaml` was not opened, parsed or
grepped until the guesses were written to disk. The only files read from the
target directory before then were the two videos and `_metadata.yaml`.

## Output format

The guess is written to `_action_guessed.yaml` in the target directory. It uses
the same structure as `_action.yaml`: one list entry per timestep, with
`rc-position-target` (the six pose values), `rc-angle-target` (servos `'0'`–`'4'`),
`rc-pulse-target` (servos `'0'`–`'5'`) and `time`.

Only the six pose values are guessed. Angles and pulses are computed from them
using the controller's own code, the same path `PositionController.move` takes:

- angle 0 = `90 + heading`
- angles 1–3 = `PositionController.ik_shoulder_elbow_wrist(pos)`
  ([position_controller.py](../robot/al5d/position_controller.py))
- angle 4 = `wrist_rotation`
- pulses 0–4 = `RobotHelper.servo_angle_to_pulse` with the
  `robot_al5d/angle_controller_00` and `robot_al5d/pulse_controller_00`
  configurations ([helper.py](../robot/al5d/helper.py))
- pulse 5 (the gripper) = `1000 + 15 * (100 - gripper)`

Before use, this conversion was checked against the reference demonstration.
Recomputing angles and pulses from its recorded `rc-position-target` reproduced
all 387 recorded `rc-angle-target` and `rc-pulse-target` entries exactly, with a
maximum difference of 0. Any error in the guessed file therefore comes from the
pose estimate, not from the conversion.

## Approach

### 1. Learn the visual appearance of each pose value

From the reference demonstration, frames were chosen at the minimum and maximum
of each of the six pose values, plus the home pose at t=0 and a few samples in
between. Each frame was rendered as a `dev2 | dev3` pair labelled with its known
pose. Studying these pairs gave the following cues.

- **dev3** looks almost straight down on the table. Heading 0 points the arm
  toward the top of the image, positive heading turns it clockwise (about +80°
  points right), and negative heading turns it counter-clockwise (−56° points
  up-left).
- **dev2** is a closer, more oblique view, rotated about 90° relative to dev3.
  Heading 0 points left, +80° points up, and −56° points down.
- **distance** is the horizontal reach, seen in dev3 as the pixel distance from
  the round black base to the gripper. Very roughly, 45 px ≈ 4.5, 65 px ≈ 6 and
  105 px ≈ 9.
- **wrist_angle** near −2 points the gripper straight outward, so it looks
  long. Near −88 it points down and looks short from above.
- **gripper** opening shows only as the separation of the two white fingers,
  which is a few pixels at this resolution.
- **height** and **wrist_rotation** have no clear cue in either view.

### 2. A calibrated heading scale for dev3

dev3 is not exactly overhead, so the relation between heading and the arm's
angle in the image is not linear. The base pivot was located on a pixel grid at
about (126, 132) in dev3. The heading-to-image-angle mapping was then fitted
piecewise-linearly through three reference poses. Image angles are measured
clockwise from image-up:

| heading | image angle |
|---|---|
| −56° | −63° |
| 0° | −3° |
| +81° | +118° |

The fit was checked by drawing the ray for each known heading on 25 reference
frames, one every 16 steps. The rays lined up with the arm in all of them. The
scale is drawn as tick marks on a ring around the base, every 15° of heading,
labelled every 30°. A first version used full-length rays, but these hid the
arm and were dropped.

### 3. Contact sheets of the target

The target was sampled every 5 steps (0.5 s), giving 90 keyframes. Each keyframe
was rendered as plain dev2 beside dev3 with the heading scale. Ten pairs went on
each sheet, for nine sheets in total. Claude viewed every sheet.

Some keyframes are corrupted by decoding artefacts, which show as displaced
blocks of the image: dev3 at t = 150, 255, 335 and 390, and dev2 at t = 165 and
310. At those keyframes the other camera and the neighbouring
keyframes were used instead.

### 4. Keyframe estimation by eye

For each keyframe, Claude wrote down the six pose values. The approach per value:

- **heading**: read from the dev3 tick scale and cross-checked with the arm
  direction in dev2. Where the two disagreed (t ≈ 155–180, when the arm is short
  and low), the dev2 reading was preferred.
- **distance**: from the base-to-gripper pixel distance in dev3, using the
  pixel scale above.
- **height**: from the apparent size and position of the gripper in dev2. A
  gripper that looks larger and higher in the image was assumed to be higher.
- **wrist_angle**: from how long the gripper looks in dev3 and how it is angled
  in dev2. The default was about −40.
- **wrist_rotation** and **gripper**: not readable, so they were held near
  typical values (80 and 45). The first two keyframes used the reference home
  pose (75.5 and 95), because the first target frames look identical to it.
- All values were kept within the ranges seen in the reference demonstration.

### 5. Interpolation and completion

`make_guessed_actions.py`:

1. linearly interpolates the 90 keyframes over t = 0…445;
2. clips each value to a safe range;
3. computes angles and pulses as described in the Output format section; and
4. writes the YAML with `yaml.dump(..., indent=4)`, the same style as
   `Demonstration.save_metadata`.

If a pose failed IK or went outside the angle limits, the script would move the
wrist angle toward −45 until it was accepted. No step needed this.

The guessed file was checked for 446 entries, `time` = 0…445, the same key
structure as the reference `_action.yaml`, and finite values throughout. Every
entry also loads as a `RobotPosition` and normalizes through
`to_normalized_vector`.

## Result

The plots and error figures were made only after the guessed file was written,
by `plot_guessed_vs_truth.py`. It follows the 2×3 layout of
`Compare_VisualProprioception.ipynb`, with ground truth as a thick black line.

| value | RMSE (raw) | RMSE (normalized) | MAE (raw) | correlation |
|---|---|---|---|---|
| height | 1.44 | 0.359 | 1.27 | −0.13 |
| distance | 0.76 | 0.109 | 0.65 | 0.87 |
| heading | 8.7° | 0.049 | 6.7° | 0.96 |
| wrist_angle | 23.5° | 0.262 | 17.2° | 0.03 |
| wrist_rotation | 9.3° | 0.154 | 7.7° | −0.06 |
| gripper | 31.3 | 0.313 | 26.7 | 0.08 |

**Heading and distance were recovered.** The guesses follow the timing and size
of almost every swing and reach in the ground truth. Heading error per
25-step window is 3–7° for most of the demonstration. It is concentrated in
three places:

- **t ≈ 325–349 (RMSE 19.5°):** a short swing to about +33° was missed, because
  the arm is folded close to the base there and its direction is hard to read.
- **t ≈ 150–199 (RMSE about 15°):** the guesses overshoot the negative extreme,
  reaching −65 against a true minimum of −56.7.
- **t ≈ 75–99 (RMSE 11°):** the guesses lag a swing to the right.

**Height was not recovered**, even though it was actively estimated. During the
low phases (t ≈ 100–150 and 320–350) the true height falls to 1.2–1.4, while
the guesses stayed near 3. Apparent size in dev2 turned out to be a poor
cue for height.

**Wrist angle, wrist rotation and gripper carry no information.** Their
correlation with the truth is about zero. For wrist rotation and gripper this is
expected, since they were held constant. The ground truth for all three is a
sequence of random piecewise-linear moves between holds, and none of it is
visible at 256×256.

**The home-pose assumption was slightly wrong.** The first target frames look
identical to the reference home pose, but the true wrist angle at t=0 is −46.5
rather than −43.5.

## What this suggests for learned visual proprioception

- Heading and distance are the values that these two views show clearly. A
  learned model that is worse than this baseline on them is underusing the
  image.
- Height, wrist angle, wrist rotation and gripper opening are hard to see at
  256×256 from these viewpoints. Better results on them probably need a
  different signal, for example higher resolution, a crop around the gripper,
  or a camera that sees the arm from the side. More training alone may not be
  enough.
- The ground truth is `rc-position-target`, the commanded target, not the
  measured pose. How far the pictured arm lags behind the command was not
  measured here. Any such lag counts as error for this baseline and for learned
  models alike.

## Limitations

- One demonstration and one annotator. There is no estimate of how much the
  results would vary between runs or between demonstrations.
- Keyframes were every 0.5 s with linear interpolation. Motions shorter than a
  keyframe interval can be missed, and holds get rounded off.
- The heading scale was fitted to three reference poses. It is accurate to
  within a few degrees near the fitted points and may be less accurate between
  them.
- The calibration used one reference demonstration, whose actions had been
  read before the experiment started.

## Reproduction

All working files are in `BerryPicker-Demopacks/random-both-cameras-video-visual-guess/`,
next to the demopack. That folder's `README.md` lists them. The scripts take
that folder as their first argument and are run from `BerryPicker/src` with
`PYTHONPATH=.`.

1. `make_calib.py`, `grid_view.py` and `check_overlay.py` build the reference
   images in `calib/`.
2. `make_sheets.py <folder> 5 0 446 s` builds the contact sheets in `sheets/`.
3. The by-eye step has no script. Its output is `keyframes_19_12.csv`.
4. `make_guessed_actions.py` writes `_action_guessed.yaml` into the target
   demonstration.
5. `plot_guessed_vs_truth.py` produces `guessed_vs_truth{,_normalized}.{pdf,jpg}`
   and prints the error table.
