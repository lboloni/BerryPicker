# Plan: WidowX AI support in BerryPicker

## Status and intent

**Implemented on 2026-10-07, on a Mac without the arm**, with the deviations
listed in "Implementation" below. Everything that does not need the arm or
Linux is implemented and tested against a kinematic fake driver and the
MuJoCo simulation. What remains to be checked or fixed on a Linux machine
and on a machine with the arm is listed in "Problems to fix". The phases
below are the original plan, kept as the record of the design.

This plan describes how to add the Trossen
Robotics **WidowX AI** (`wxai`) as a third robot embodiment, next to the AL5D
and the Interbotix WidowX 250s, with the same functionality the WidowX has
today:

- a native task-space position controller for the **real arm**;
- a strict headless **simulated controller** without dependencies;
- a **MuJoCo simulation** with rendered cameras, playing the role that
  Gazebo plays for the WidowX;
- demonstration collection with the Xbox end-effector controller, AutoMove,
  and, new for this robot, a **WidowX AI leader arm**;
- expruns, a verification notebook, tests and install instructions.

The guiding rule is to mirror `src/robot/widowx` file by file, so that a
reader who knows the WidowX code can read the WidowX AI code. The code is
new; the existing WidowX code is changed only where it has to become robot
independent (Phase 4).

## Why a new package and not a WidowX variant

The two robots share a name, not an API.

| | WidowX 250s (`robot/widowx`) | WidowX AI (`robot/wxai`) |
|---|---|---|
| Hardware | Dynamixel servos through a U2D2 (USB) | Trossen actuators, controller box on Ethernet (default `192.168.1.2`) |
| Library | `interbotix_xs_modules` on a ROS 2 node (`WidowXRuntime`) | `trossen_arm` (`pip install trossen-arm`), no ROS |
| Object | `bot.arm`, `bot.gripper`, `bot.core` | one `trossen_arm.TrossenArmDriver` |
| Cartesian move | `set_ee_pose_components(x, y, z, roll, pitch, yaw, custom_guess, moving_time, blocking)`, which returns `reachable` | `set_cartesian_positions(goal, interpolation_space, ...)` |
| Joint move | `set_joint_positions(joints, moving_time, blocking)` | `set_arm_positions(goal, goal_time, blocking)` |
| Modes | position (fixed) | `position`, `velocity`, `external_effort`, `idle` per joint |
| Gripper | `grasp()`/`release()` with PWM pressure | the 7th joint; position in meters (0–0.04 m) or effort in N (up to 100 N) |
| Home/sleep | `go_to_home_pose()`, `go_to_sleep_pose()` | no helpers; home `[0, π/2, π/2, 0, 0, 0]`, sleep all zeros |
| Teleoperation | Xbox, or a backdriven WidowX | Xbox, or a WidowX AI leader arm in `external_effort` mode |
| Simulation | headless fake, Interbotix SDK + Gazebo Classic (ROS 2 Humble only) | headless fake, MuJoCo (`trossen_arm_mujoco`, models `wxai_base.xml`, `wxai_follower.xml`) |

Consequences:

- No `WidowXRuntime`. One driver per arm, owned by its controller.
- No Interbotix IK. Cartesian moves are delegated to the driver; reachability
  must come from somewhere else (see Phase 0).
- The gripper is not a binary grasp/release but a joint with a width or an
  effort.
- The real robot can run on macOS (`trossen-arm` supports macOS 14–15), so no
  distrobox container is needed, unlike the WidowX Gazebo setup.

## Phase 0: verify the API before writing code

The `trossen_arm` API is documented as "under heavy development" and
"subject to frequent changes". Pin a version in the install, and resolve the
following questions against the API reference
(`docs.trossenrobotics.com/api/library_root.html`) and the real arm, with a
short throwaway script. The answers go into the "Decisions" section of this
file before Phase 1 starts.

1. **Cartesian pose format.** Are elements 3–5 of a Cartesian position roll,
   pitch, yaw or an angle-axis vector? In which frame (base, flange or tool)?
2. **Reachability.** Does `set_cartesian_positions` report an unreachable
   target (return value or exception), and is there an IK call without
   motion (the analogue of `execute=False`)?
3. **Timing.** The exact signatures of `set_arm_positions`,
   `set_cartesian_positions` and the gripper calls: `goal_time`, `blocking`,
   and what a non-blocking call followed by a new call does (the WidowX
   participant sends a new non-blocking target every 0.1 s tick).
4. **State.** The fields of `get_robot_output()` (joint positions, velocities,
   efforts, Cartesian pose) and their units.
5. **Shutdown.** The correct sequence at the end: move to sleep, set modes to
   `idle`, and the call that releases the connection (`cleanup()` or similar).
6. **Errors.** What `configure(..., clear_error)` does, and how driver errors
   surface in Python (they must propagate, not be caught).
7. **Joint limits.** The six arm joint ranges and the gripper range from the
   driver, to compare with the published specifications.
8. **Python version.** Whether `trossen-arm` and `mujoco` have wheels for the
   Python version of `berrypickervenv` (3.14). `trossen_arm_mujoco` is
   tested with MuJoCo 3.2.3 and Python 3.10+. If the wheels are missing, the
   choice is between a second venv for the WidowX AI and building from
   source; this is a decision for the user.
9. **Simulation model frame.** Whether the end-effector frame of
   `wxai_follower.xml` matches the tool frame of the real driver, so that the
   same pose means the same thing in simulation and on the robot.

## Phase 1: pose, command, motion helpers and the strict simulator

New files, each the analogue of the WidowX file of the same name:

```text
src/robot/wxai/
  __init__.py                        exports, as in robot/widowx/__init__.py
  position.py                        WXAIPose, WXAICommand
  move.py                            move_towards, move_pose_by, move_pose_by_clamped, move_pose_towards
  simulated_position_controller.py   SimulatedPositionController
```

**`WXAIPose`.** The same structure as `WidowXPose`: fields
`x, y, z, roll, pitch, yaw` in meters and radians, `POSE_DEFAULT`,
`POSE_MIN`, `POSE_MAX` from the exprun, `validate`, `limit`, `to_vector`,
`to_normalized_vector`, `from_vector`, `from_normalized_vector`,
`empirical_distance`, `as_dict`. It checks `exp["robot_name"] == "wxai"`. If
Phase 0 shows that the driver uses angle-axis, the conversion happens in the
controller; the pose stays roll/pitch/yaw so that the Xbox and AutoMove code
and the recorded data have the same shape for both arms.

**`WXAICommand`.** A pose plus a gripper command. Because the WidowX AI
gripper is a joint, the command carries `gripper_action` in
`("hold", "position", "effort")` and a `gripper_value` (meters for
`position`, newtons for `effort`). The recorded action is
`{"wxai-command": command.as_dict()}`.

**`move.py`.** A copy of `robot/widowx/move.py` with `WXAIPose`. The functions
are pure and independent of the robot except for the pose class. To avoid
the copy, `move.py` could take the pose class as a parameter, but that is a
change to working WidowX code. Copy first; merge later only if both stay
identical.

**`SimulatedPositionController`.** A copy of the WidowX one: no physics, the
pose jumps to the target, `can_reach` only validates the limits, and
`get_state` returns the pose and the gripper command. It has no
dependencies, so it is the default in tests and in collectors on machines
without the robot.

Tests (`src/test/robot/wxai/`): `test_wxai_position.py` and
`test_wxai_simulated_controller.py`, mirroring the WidowX tests.

## Phase 2: the real robot controller

```text
src/robot/wxai/
  position_controller.py   PositionController, on trossen_arm.TrossenArmDriver
```

The public interface is exactly that of `robot.widowx.PositionController`, so
that the participants and the remote-control code work unchanged:
`start_robot`, `stop_robot`, `get_position`, `get_target`, `get_state`,
`can_reach`, `move(command, moving_time, blocking)`, `command_gripper`,
`move_joint_positions`, `go_home`, `go_sleep`. `move_cartesian`, which the
WidowX uses for relative Cartesian trajectories, is left out unless a caller
needs it.

- **Construction.** `__init__(exp, driver=None)`. Without a driver it creates
  `TrossenArmDriver()` and calls `configure(Model.wxai_v0, end_effector, ip,
  clear_error)` in `start_robot`. The `driver` argument exists for the tests
  (fake driver), as `bot` does for the WidowX.
- **`start_robot`.** Configure, set the arm to `position` mode, set the
  gripper mode, then the `startup_pose` (`hold`, `home`, `sleep`, `default`)
  as for the WidowX.
- **`move`.** Validate the pose, convert it to the driver's Cartesian format,
  call `set_cartesian_positions` with the interpolation space from the exprun
  and `moving_time` as goal time, then apply the gripper command. An
  unreachable target raises `ValueError`, as on the WidowX.
- **`can_reach`.** Uses the answer to Phase 0, question 2. If the driver has
  no motion-free IK check, use a damped-least-squares IK on the MuJoCo model
  of Phase 3 (the method of the `trossen_arm_mujoco` controller). That makes
  MuJoCo a dependency of `can_reach` on the real robot. AutoMove needs
  `can_reach` to sample reachable poses.
- **`get_state`.** Timestamp, pose, the seven joint positions (six arm joints
  plus the gripper), joint efforts, and the gripper command. Recording the
  efforts is new; it is cheap and needed later for force-aware learning.
- **`go_home`, `go_sleep`.** `set_arm_positions` with the home and sleep
  joint vectors, taken from the exprun (not hard-coded).
- **`stop_robot`.** `shutdown_pose`, then `idle` modes and the release of the
  connection (Phase 0, question 5).

Tests: `src/test/robot/wxai/wxai_test_support.py` with a `FakeDriver` that
records the calls (as `FakeArm`/`FakeGripper` do for the WidowX), and
`test_wxai_controller.py`.

**Expruns** (`data/expruns/robot_wxai/`):

```yaml
# _defaults_robot_wxai.yaml
input-to-notebook: []

# position_controller_wxai_00.yaml
# Native task-space controller for a WidowX AI. Distances are meters and angles radians.
robot_name: wxai
controller_type: position_controller
model: wxai_v0
end_effector: wxai_v0_follower
ip_address: 192.168.1.2       # overridden per machine in the _sysdep overlay
clear_error: false
interpolation_space: cartesian
moving_time: 0.1
gripper_effort: 20.0
startup_pose: hold
shutdown_pose: sleep
HOME_JOINTS: [0.0, 1.5708, 1.5708, 0.0, 0.0, 0.0]
SLEEP_JOINTS: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
POSE_DEFAULT: {x: ..., y: ..., z: ..., roll: 0.0, pitch: 0.0, yaw: 0.0}
POSE_MIN: {...}
POSE_MAX: {...}
```

The pose values are measured on the arm in Phase 0. The IP address is a
machine fact and belongs in
`<experiment_system_dependent_dir>/robot_wxai/position_controller_wxai_00_sysdep.yaml`.

**Notebook.** `src/demonstration/Verify_Robot_WXAI.ipynb`, modeled on
`Verify_Robot_WidowX.ipynb`: start, read the state, go home, move through a
few poses, open and close the gripper, sleep, stop. It follows the stage
notebook contract (parameters cell, `exp.done()` at the end) and is listed in
the `input-to-notebook` of the run.

## Phase 3: MuJoCo simulation with cameras

```text
src/robot/wxai/
  simulation/
    __init__.py
    mujoco_runtime.py                one MuJoCo model and data, shared by the robot and the cameras
    mujoco_position_controller.py    MujocoPositionController
    scene_wxai_berrypicker.xml       the scene: the wxai_follower model, a table, the fixed cameras
```

**Model.** Use `wxai_follower.xml` from `trossen_arm_mujoco`, cloned next to
BerryPicker and installed editable, as Conv-VAE-PyTorch and ExpRunFlow are.
The BerryPicker scene includes it and adds a table and the cameras at the
positions of the real fixed cameras. It does not copy the robot model.

**`MujocoRuntime`.** The analogue of `WidowXRuntime`: it owns the
`MjModel`/`MjData`, steps the physics, and lets both the robot controller
and the camera participant register on the same simulation. It steps the
physics in `update`, driven by the demonstration tick, not in a background
thread, so a recording is deterministic.

**`MujocoPositionController`.** The same interface as the real controller.
Cartesian targets go through a damped-least-squares differential IK, the
method of the `trossen_arm_mujoco` `Controller` (`ik_scale`, `ik_damping`),
to joint targets for the position actuators. `can_reach` runs the same IK
without moving. `moving_time` becomes a linear interpolation of the joint
targets over that time. The gripper joint takes the position or effort
command. The IK lives in one function, `ik.py`, which the real controller can
reuse for `can_reach` (Phase 2).

**Cameras.** A `MujocoCameraParticipant` renders the named cameras of the
scene with `mujoco.Renderer` at the resolution of the real cameras, headless
by default (`MUJOCO_GL=egl` on Linux, the default backend on macOS). It
records images under the same names as the real camera participant, so that
the demonstrations are interchangeable with real ones.

**Expruns.**
`data/expruns/robot_wxai/position_controller_mujoco_wxai_00.yaml` (the same
fields, plus `scene`, `physics_timestep`, `ik_scale`, `ik_damping`) and
`data/expruns/controllers/mujoco_cameras.yaml` (camera names and
resolution).

Tests: `test_wxai_mujoco_controller.py` (start, move to a reachable pose and
measure the pose error, reject an unreachable pose, gripper) and a camera
test that renders one frame headless and checks its shape.

## Phase 4: demonstration integration

**Participants** (`src/demonstration/demonstration_participant.py`):

- `WXAIParticipant(name, spec, exp, kind)` with `kind` in `hardware`,
  `simulated`, `mujoco`. It is the `WidowXParticipant` with the WidowX AI
  controller classes and the `wxai-command` action key.
- `MujocoCameraParticipant`.
- `WXAILeaderParticipant` (Phase 5).

New factory entries in `create_participants`: `wxai_hardware`,
`wxai_simulated`, `wxai_mujoco`, `mujoco_cameras`, `wxai_leader`.

**Machine bindings** (`data/expruns/machine/current_sysdep.template.yaml`):
`wxai_robot` (`wxai_hardware`, `resources: [wxai_robot]`), `simulated_wxai`,
`mujoco_wxai` (`resources: [mujoco_wxai]`), `mujoco_cameras`
(`resources: [mujoco_cameras]`; the robot and the cameras share one
`MujocoRuntime`, so they must not claim the same resource), `wxai_leader`
(`resources: [wxai_leader_robot]`). All `available: false` in the template.

**Making the Xbox and AutoMove controllers robot independent.**
`WidowXXboxEndEffectorController` and `WidowXAutoMoveController` import
`WidowXPose`, `WidowXCommand` and the WidowX `move` functions. They only need
a pose class, a command class and the move helpers. The smallest change is
that each position controller exposes them as class attributes, and the two
controllers take them from `robot_controller`:

```python
class PositionController:          # in both robot/widowx and robot/wxai
    POSE = WXAIPose
    COMMAND = WXAICommand
```

```python
self.Pose = type(robot_controller).POSE
```

The gripper part differs (grasp/release with pressure vs. width or effort).
The Xbox controller maps its two gripper buttons to `grasp`/`release` for the
WidowX and to `effort` with `-gripper_effort`/`+gripper_effort` for the
WidowX AI; this mapping is a small method of the command class
(`COMMAND.grasp(pose, exp)`, `COMMAND.release(pose, exp)`). This is the only
change to existing WidowX code, and the existing WidowX tests must keep
passing unchanged. The leader participants that bind to a target robot
(`WidowXXboxEndEffectorLeaderParticipant`, `WidowXAutoMoveLeaderParticipant`)
accept a `WXAIParticipant` as target as well.

**Collector recipes** (`data/expruns/demonstration_collector/`), one per
WidowX recipe that makes sense for the WidowX AI:

- `xbox_simulated_wxai_cameras.yaml`
- `xbox_mujoco_wxai_mujoco_cameras.yaml`
- `xbox_wxai_cameras.yaml`
- `automove_simulated_wxai_cameras.yaml`
- `automove_mujoco_wxai_mujoco_cameras.yaml`
- `automove_wxai_cameras.yaml`
- `wxai_leader_wxai_cameras.yaml` (Phase 5)

plus `data/expruns/automove/automove_wxai_random_pose_00.yaml` with a
WidowX AI sampling workspace.

Tests: extend `test_widowx_participant.py` with a `test_wxai_participant.py`,
and run the existing remote-control tests for both robots.

## Phase 5: leader–follower teleoperation

The WidowX AI is sold with a leader arm. This is how the Aloha recordings
were made, and it gives better demonstrations than the Xbox.

- `WXAILeaderParticipant`: owns a second `TrossenArmDriver` configured with
  `wxai_v0_leader` at its own IP. In `start` it puts the leader in
  `external_effort` mode with zero effort (gravity compensated, freely
  movable). In `update` it reads the leader joint positions and emits them
  as the command of the target robot.
- The follower then needs a **joint-space** command: `WXAICommand` gets an
  optional `joints` field (6 arm joints plus the gripper), and the
  controllers execute it with `set_all_positions` (real) or the joint
  targets (MuJoCo). The recorded action is the leader joint vector, the
  telemetry the follower joint vector, exactly the split that the Aloha
  importer uses (`pair_1_leader`, `pair_1_follower`).
- Force feedback (the follower efforts sent back to the leader, gain about
  0.1 in the Trossen demo) is optional, behind an exprun field
  `force_feedback_gain`, 0 by default.
- In MuJoCo, the leader can be the real leader arm driving the simulated
  follower. This makes MuJoCo demonstrations with a human operator possible
  without the follower arm.

Exprun: `data/expruns/controllers/wxai_leader_00.yaml` (`ip_address`,
`end_effector: wxai_v0_leader`, `force_feedback_gain`).

## Phase 6: install and documentation

- `src/robot/wxai/INSTALL-WXAI.md`: network setup of the controller box (PC on
  `192.168.1.x`, the arm on `192.168.1.2`, the leader on `192.168.1.3`),
  `pip install trossen-arm==<pinned>`, the MuJoCo install, the headless
  rendering setting, and how to run `Verify_Robot_WXAI.ipynb`.
- `src/install/berry_install.sh`: clone `trossen_arm_mujoco` next to
  BerryPicker; `pip install trossen-arm==<pinned> mujoco` and
  `pip install -e "$CHECKOUTS/trossen_arm_mujoco"`.
- `src/install/settings-sample.yaml`: nothing new; the IP addresses go into
  the `_sysdep` overlays.
- A design document, `src/robot/wxai/DESIGN-WXAI.md`, written from this plan
  once Phases 0–4 are done, replacing this file, and referenced from
  `CLAUDE.md`.

## Later, not in this plan

- An Aloha importer that produces `wxai-command` actions instead of the raw
  `aloha-pair-1-leader` vectors (`DESIGN-Aloha-WidowX-AI-SimpleImport.md`),
  once the joint order and units are confirmed in Phase 0.
- Training RCCO controllers on WidowX AI demonstrations, which needs the
  normalized `WXAIPose` vector as the controller output (it exists after
  Phase 1).
- Cross-embodiment experiments between the AL5D, the WidowX and the WidowX AI
  (Phase 5 of the robot-controller plan).

## Order and dependencies

```text
Phase 0 (API answers) -> Phase 1 -> Phase 2 (real) ---------\
                                 \-> Phase 3 (MuJoCo) -------> Phase 4 -> Phase 5 -> Phase 6
```

Phases 1 and 3 can be done without the real arm. Phase 2 can be written
against the fake driver, but is finished only after a run of
`Verify_Robot_WXAI.ipynb` on the arm.

## Verification

- Unit tests per phase, run from the repository root, each test file in its
  own process as the rest of the suite:
  `PYTHONPATH=src python -m unittest src.test.robot.wxai.test_wxai_position` etc.
- The existing WidowX and remote-control tests pass unchanged after Phase 4.
- `test_exprun_notebooks.py` passes with the new expruns and notebook.
- Real arm: `Verify_Robot_WXAI.ipynb` completes and ends with `exp.done()`.
- MuJoCo: `Collect_Demonstration.ipynb` with
  `automove_mujoco_wxai_mujoco_cameras` records a demonstration whose images
  and `wxai-command` actions can be read back with
  `Verify_Demonstration.ipynb`.
- Leader–follower: a short demonstration with `wxai_leader_wxai_cameras` in
  which the follower tracks the leader, and the recorded leader and follower
  joint vectors differ by less than a tolerance measured in Phase 0.

## Implementation

### Files

```text
src/robot/wxai/
  __init__.py                         exports PositionController, SimulatedPositionController, WXAIPose, WXAICommand
  position.py                         WXAIPose, WXAICommand (subclasses of the WidowX ones), pose_to_driver, pose_from_driver
  move.py                             re-exports the WidowX motion helpers, which are now generic
  kinematics.py                       WXAIKinematics: forward kinematics and damped least squares IK of ee_site
  position_controller.py              PositionController on a driver: trossen_arm, fake or mujoco
  fake_driver.py                      FakeTrossenArmDriver: the driver API on the kinematics, instantaneous
  leader_controller.py                LeaderController: a hand-moved leader arm
  simulated_position_controller.py    the strict headless simulator
  simulation/
    scene_wxai_berrypicker.xml        table, a target ball, two fixed cameras, the robot_mount frame
    mujoco_runtime.py                 MujocoRuntime: the scene with the robot attached, shared by robot and cameras
    mujoco_driver.py                  MujocoTrossenArmDriver: the driver API on the MuJoCo position actuators
  INSTALL-WXAI.md
src/demonstration/
  demonstration_participant.py        WXAIParticipant, WXAILeaderParticipant, MujocoCameraParticipant, factories
  Verify_Robot_WXAI.ipynb             the stage notebook of every robot_wxai run
  Show_WXAI_Simulation.ipynb          watch the MuJoCo arm move through waypoints, live and as a GIF
src/test/robot/wxai/                  pose, kinematics, controller (fake), MuJoCo, leader, simulator
src/test/demonstration/test_wxai_participant.py
data/expruns/robot_wxai/              _defaults, position_controller_wxai_00 (real), _fake_wxai_00, _mujoco_wxai_00
data/expruns/controllers/             mujoco_cameras, wxai_leader_00, wxai_leader_fake_00
data/expruns/automove/automove_wxai_random_pose_00.yaml
data/expruns/demonstration_collector/ {automove,xbox}_{wxai,simulated_wxai,mujoco_wxai_mujoco}_cameras,
                                      automove_fake_wxai_cameras, wxai_leader_wxai_cameras,
                                      {,fake_}wxai_leader_mujoco_wxai_mujoco_cameras
data/expruns/machine/current_sysdep.template.yaml   the WidowX AI bindings (all unavailable)
```

The Mac profile in `Lotzi-BerryPicker-Settings/settings/experiment-config/szenes/machine/current_sysdep.yaml`
has the fake, simulated and MuJoCo bindings available.

### Deviations from the plan

- **MuJoCo is a driver, not a controller.** Instead of a separate
  `MujocoPositionController`, the MuJoCo simulation implements the subset of
  the `TrossenArmDriver` API that the controller uses, as does the fake
  driver. There is one `PositionController`, and the `driver` field of the
  robot exprun (`trossen_arm`, `fake`, `mujoco`) selects the arm. The
  physics advances in `PositionController.update(dt)`, which
  `WXAIParticipant` calls once per tick, and in blocking commands.
- **The gripper actions are those of the WidowX.** `WXAICommand` keeps
  `hold`/`grasp`/`release` and `gripper_pressure` in [0, 1]; grasp and
  release are an external effort of `gripper_pressure * gripper_max_effort`
  (negative closes). This keeps the Xbox and AutoMove controllers unchanged.
  The joint command of the leader is the optional `joints` field (six arm
  joints in rad and the gripper in m).
- **Subclasses instead of copies.** `WXAIPose` and `WXAICommand` subclass
  `WidowXPose` and `WidowXCommand`; `WXAIParticipant` subclasses
  `WidowXParticipant`; the WidowX AI simulator subclasses the WidowX one.
  The changes to the WidowX code are small: `WidowXPose.ROBOT_NAME`,
  `__copy__` keeping the class, `move_pose_towards` returning the class of
  its input, `POSE`/`COMMAND` class attributes on the WidowX controllers,
  the Xbox and AutoMove controllers and `WidowXParticipant` taking their
  pose and command classes from the robot controller, and
  `WidowXParticipant.ACTION` for the action key. The WidowX tests pass
  unchanged.
- **`can_reach` uses the MuJoCo IK for every driver**, because the driver
  has no motion-free reachability check. Before the start it solves from
  the sleep joints, because the real driver has no joints before
  `configure` (AutoMove samples its waypoints at bind time).
- **The leader stops** with the exit key of a camera preview, or after
  `max_timesteps` ticks, since the MuJoCo cameras have no preview window.
- **The Xbox and AutoMove leader bindings are shared** with the WidowX
  (`widowx_xbox_end_effector`, `widowx_automove`); the AutoMove type names
  (`random_widowx_pose`, `widowx_pose_velocity`) are reused.

### Verified on the Mac

- `trossen-arm` 1.11.0 and `mujoco` 3.15.0 install into the Python 3.14
  venv on macOS (arm64).
- The unit tests: 28 in `src/test/robot/wxai`, 5 in
  `test_wxai_participant.py`; all other test files give the same results as
  before the change.
- `Verify_Robot_WXAI.ipynb` runs with papermill on the fake and the MuJoCo
  run: home, ±2 cm test moves with errors below 1 mm, gripper open and
  close, sleep.
- Headless collections with `DemonstrationRecorder`:
  `automove_mujoco_wxai_mujoco_cameras` (60 ticks in 6.4 s, three rendered
  cameras, `wxai-command` actions, observed pose within a few mm of the
  target) and `fake_wxai_leader_mujoco_wxai_mujoco_cameras` (50 ticks, the
  leader joints in the action, the follower joints in the telemetry).

## Decisions

The answers to Phase 0, from the `trossen_arm` 1.11.0 docstrings and the
`trossen_arm_mujoco` models; the ones marked *unverified* need the arm.

1. **Cartesian pose format.** `[x, y, z]` plus the **angle-axis** vector of
   the rotation, "of the end effector frame measured in the base frame".
   `WXAIPose` stays roll/pitch/yaw (extrinsic xyz, as for the WidowX), and
   `pose_to_driver`/`pose_from_driver` convert.
2. **Reachability.** There is no motion-free IK call.
   `set_cartesian_positions` has `num_trajectory_check_samples` (default
   0) to check the feasibility of the trajectory; how an infeasible goal is
   reported is *unverified*. `can_reach` uses the MuJoCo IK.
3. **Timing.** `set_cartesian_positions(goal, interpolation_space,
   goal_time=2.0, blocking=True, ...)`, `set_arm_positions(goal, goal_time,
   blocking)`, `set_all_positions` (six arm joints in rad and the gripper in
   m), `set_gripper_external_effort(effort_N, goal_time, blocking)`. The
   interpolation depends on goal_time: quintic above 0.2 s, linear above
   0.001 s, immediate below. A 0.1 s tick with `moving_time: 0.1` uses the
   linear interpolation. What a new non-blocking goal does to a running one
   is *unverified*.
4. **State.** `get_all_positions`, `get_all_efforts` (Nm, N for the
   gripper), `get_all_external_efforts`, `get_cartesian_positions`,
   `get_gripper_position` (m), and `get_robot_output()` with `joint`
   (`all`, `arm`, `gripper`) and `cartesian` parts.
5. **Shutdown.** `set_all_modes(Mode.idle)` and `cleanup()`;
   `cleanup(reboot_controller=False)` is the documented release. Whether
   idle is safe in the sleep pose is *unverified*.
6. **Errors.** `configure(model, end_effector, serv_ip, clear_error,
   timeout=20.0)`; `clear_error()` re-configures with `clear_error=True`;
   `get_error_information()` returns the error text. Driver errors propagate
   (`trossen_arm.RuntimeError`, `trossen_arm.LogicError`).
7. **Joint limits.** From the MuJoCo model: joint_0 ±3.054, joint_1
   [0, π], joint_2 [0, 2.356], joint_3 and joint_4 ±π/2, joint_5 ±π, the
   gripper carriages [0, 0.044] m. The specification says joint_0 ±π and a
   0.04 m finger displacement; `get_joint_limits()` of the arm is
   *unverified*.
8. **Python version.** Wheels exist for Python 3.14 on macOS arm64. Linux
   is *unverified*.
9. **Tool frame.** `wxai_v0_follower` has `t_flange_tool = [0.156062, 0, 0,
   0, 0, 0]`, the MuJoCo `ee_site` is 0.156 m in front of the flange, so the
   frames agree to 0.06 mm; that the base frame of the driver is the
   `base_link` of the model is *unverified*.

## Problems to fix

### On a machine with the arm

1. **Frames.** Check that `get_cartesian_positions()` equals
   `WXAIKinematics.forward()` of `get_arm_positions()` in a few poses
   (position and angle-axis). If not, the MuJoCo IK of `can_reach` and the
   driver disagree; fix the base or tool frame in `kinematics.py`.
2. **Unreachable Cartesian goals.** Find out how `set_cartesian_positions`
   reports a goal it cannot reach, and whether `num_trajectory_check_samples`
   should be set. The fake and MuJoCo drivers raise `RuntimeError`.
3. **Streaming non-blocking goals.** The participants send a new
   non-blocking Cartesian goal every 0.1 s with goal_time 0.1 s. Check that
   the motion is smooth; tune `moving_time` and `interpolation_space`
   (`cartesian` or `joint`).
4. **Gripper.** `gripper_max_effort: 40.0` N and `gripper_time` are guesses;
   check that a positive external effort opens (as in Trossen's
   `simple_move.py`), the grip force, and the cost of switching the gripper
   between external effort and position mode for joint commands.
5. **Joint limits, home and sleep.** Compare `get_joint_limits()` with the
   model; check `HOME_JOINTS` and `SLEEP_JOINTS` (from the Trossen demos).
6. **Shutdown.** Check that `set_all_modes(idle)` after the sleep pose does
   not drop the arm; otherwise keep position mode until `cleanup()`.
7. **Pose envelope and AutoMove workspace.** `POSE_DEFAULT`, `POSE_MIN`,
   `POSE_MAX` and the AutoMove sampling box of
   `automove_wxai_random_pose_00` were chosen on the MuJoCo model; measure
   them on the real setup. Nothing checks collisions.
8. **End-effector revision.** `trossen_arm` has dated end effectors
   (`wxai_v0_follower_20250509`, `_20260626`); choose the one of the arm.
9. **Latency of `get_state`.** It makes five driver calls per tick
   (Cartesian pose, positions, efforts, external efforts, gripper); use one
   `get_robot_output()` if this is too slow over Ethernet.
10. **Leader arm.** Check that zero external effort makes the leader
    gravity compensated and freely movable; the sign and size of
    `force_feedback_gain`; and whether 10 Hz joint streaming to the follower
    is smooth enough (the Trossen demo runs a tight loop) or needs a
    separate, faster loop.
11. **IP addresses** go into the `_sysdep` overlays of the machine
    (`robot_wxai/position_controller_wxai_00_sysdep.yaml`,
    `controllers/wxai_leader_00_sysdep.yaml`).
12. Run `Verify_Robot_WXAI.ipynb` with `position_controller_wxai_00`, then
    `xbox_wxai_cameras`, `automove_wxai_cameras` and
    `wxai_leader_wxai_cameras`.

### On a Linux machine

1. **Headless rendering.** The MuJoCo cameras are tested only on macOS. On
   Linux without a display set `MUJOCO_GL=egl` or `osmesa`, and run
   `src/test/robot/wxai/test_wxai_mujoco.py` and
   `src/test/demonstration/test_wxai_participant.py`.
2. **Wheels.** Check that `trossen-arm==1.11.0` and `mujoco` install for the
   Python version of the Linux venv.
3. **Install script.** `berry_install.sh` now clones `trossen_arm_mujoco`
   and installs `trossen-arm`, `mujoco` and `trossen_arm_mujoco
   --no-deps`; it is not tested on a fresh machine.
4. **Real cameras with the simulated arm.** `automove_simulated_wxai_cameras`
   and `xbox_simulated_wxai_cameras` use `fixed_cameras`; not run on the Mac.

### Found in simulation, independent of the machine

1. **IK from a single start.** From the home joints it solves about half of
   random reachable targets; this is enough for small steps (Xbox,
   AutoMove interpolation), and AutoMove resamples rejected waypoints. Random
   restarts in `WXAIKinematics.inverse` would help large jumps.
2. **Starting folded.** From the sleep pose, moving forward at the same
   height and pitch is not reachable, so an Xbox collection with
   `startup_pose: hold` in the sleep pose rejects the first forward moves.
   Consider `startup_pose: home` for the Xbox recipes.
3. **MuJoCo tracking.** The Trossen PD gains leave the home pose about 1 cm
   low and a few mm of tracking error; `waypoint_reached_distance` of the
   WidowX AI AutoMove run is 0.003 instead of 0.001.
4. **MuJoCo gripper.** Position controlled: a positive effort opens it to
   0.044 m, a negative one closes it; the effort value is ignored. The
   external efforts of the MuJoCo and the fake drivers are zeros.
5. **One MuJoCo runtime per scene and process.** `stop_robot` releases it,
   so a stopped MuJoCo controller cannot be started again; create a new one.
6. **Camera placement.** The fixed cameras of `scene_wxai_berrypicker.xml`
   are placeholders that show the whole workspace; place them like the
   real cameras.
7. **The fake driver** moves instantaneously, and the fake leader stays in
   the sleep pose unless a test moves it.
8. **Leader actions.** In a leader collection, the `pose` of the recorded
   `wxai-command` is the follower's last Cartesian target, not the leader
   pose; training code must use `joints`.
9. **The Aloha importer** still writes `aloha-pair-1-leader` vectors;
   converting them to `wxai-command` joint actions is left for later.

## Sources

- Trossen Arm documentation: <https://docs.trossenrobotics.com/trossen_arm/v1.9/index.html>
- WidowX AI specifications: <https://docs.trossenrobotics.com/trossen_arm/v1.9/specifications/wxai.html>
- Software setup: <https://docs.trossenrobotics.com/trossen_arm/v1.9/getting_started/software_setup.html>
- Demo scripts: <https://docs.trossenrobotics.com/trossen_arm/v1.9/getting_started/demo_scripts.html>
- `simple_move.py`, `teleoperation.py`, `cartesian_position.py`: <https://github.com/TrossenRobotics/trossen_arm/tree/main/demos/python>
- Trossen Arm MuJoCo: <https://docs.trossenrobotics.com/trossen_arm/v1.9/tutorials/trossen_arm_mujoco.html>
