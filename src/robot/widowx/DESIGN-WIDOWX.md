# DESIGN: the WidowX 250s robot

**Status (2026-11-27):** §5, §6, §9 and §10 implemented; tested in Gazebo on glassy; start and stop, with the soft stop of §10, tested on the real robot (§7, step 7); the automove episode on the real robot is open.

**How to stop the robot**: through the ready pose to the rest joints, then a soft torque off; see the decision at the end of §10.6.

This document describes the driver of the Interbotix **WidowX 250s** (`robot/widowx`, ROS 2 Humble, Interbotix SDK, Gazebo Classic), and the changes planned for it. The installation is in `INSTALL-WIDOWX.md`. The WidowX AI (`robot/wxai`, Trossen, MuJoCo) is a different robot with its own driver.

## 1. Components

| File | Role |
|---|---|
| `position.py` | `WidowXPose` (x, y, z in meters, roll, pitch, yaw in radians, checked against `POSE_MIN`/`POSE_MAX` of the exp/run) and `WidowXCommand` (a pose and a gripper action `hold`/`grasp`/`release` with an optional pressure). |
| `move.py` | Pure helpers for pose targets: `move_pose_by`, `move_pose_by_clamped`, `move_pose_towards` (velocity-limited step towards a target). |
| `runtime.py` | `WidowXRuntime`: one Interbotix ROS node and executor shared by all WidowX arms of the process; `robot_startup` when the first controller starts, `robot_shutdown` when the last one stops. |
| `position_controller.py` | `PositionController`: the task-space controller. Wraps `InterbotixManipulatorXS`; the IK and the motion are done by Interbotix. |
| `simulated_position_controller.py` | `SimulatedPositionController`: the same interface without ROS. No kinematics: every pose within the limits is reachable, and a move sets the pose. |
| `simulation/run_gazebo.py` | Launches Gazebo with the WidowX and a fixed camera (`fixed_camera.sdf`), the Interbotix SDK simulator (`xs_sdk_sim`, robot name `wx250s_sdk`), and the bridge. |
| `simulation/interbotix_gazebo_bridge.py` | Mirrors the joint states of the SDK simulator into the Gazebo trajectory controllers. |

The demonstration participants are created in `demonstration/demonstration_participant.py` from the bindings of `machine/current` (factories `widowx_hardware` for the real robot and Gazebo, `widowx_simulated` for the headless simulation, `widowx_automove_leader` for the automove).

## 2. Configuration

The exp/runs are in `data/expruns/robot_widowx/`:

- `position_controller_wx250s_00`: the real robot (`interbotix_robot_name: wx250s`, the `xs_sdk` driver).
- `position_controller_gazebo_wx250s_00`: Gazebo (`interbotix_robot_name: wx250s_sdk`, the SDK simulator).

Both stream commands at 10 Hz: `moving_time: 0.1`, `accel_time: 0.05`. `startup_pose: hold` leaves the arm where it is at the start, `shutdown_pose: sleep` is meant to park it at the end (but see §6.1). The task-space limits `POSE_MIN`/`POSE_MAX` are only a coarse box; whether a pose can actually be reached is decided by the IK (§4).

## 3. Motion

`PositionController.move(command)` calls `bot.arm.set_ee_pose_components(...)` with the current measured joints as the IK seed. Interbotix solves the IK numerically (`modern_robotics.IKinSpace`), checks the solution (§4), publishes it with the trajectory time `moving_time`, and then the gripper action is applied. A rejected pose raises `ValueError`.

The streaming controllers (the automove `remote_control/widowx_automove_controller.py`, the gamepad `remote_control/widowx_xbox_end_effector_controller.py`) send on every tick a small step towards their target, computed with the configured velocities and `move_pose_towards` / `move_pose_by_clamped`. With `moving_time: 0.1` each step is a 0.1 s trajectory.

## 4. Reachability

`PositionController.can_reach(pose)` asks whether the arm can go to a pose. It calls `set_ee_pose_components(..., custom_guess=<current joints>, execute=False)` and returns its success flag. It is used in two ways:

- **Waypoint sampling**: the automove samples random poses in its workspace and keeps the reachable ones (`generate_waypoints`), before any motion.
- **Per-tick step check**: the automove checks every intermediate step (`next_command`), the gamepad controller every candidate step.

The automove is shared with the WidowX AI (`robot/wxai`), whose `can_reach` is a pure IK check on the MuJoCo model, with no speed condition.

### 4.1 Interbotix's checks

`InterbotixArmXSInterface._check_joint_limits` accepts an IK solution only if

1. every joint is within its position limits, **and**
2. every joint can get there from the last commanded joints (`joint_commands`) within `self.moving_time` without exceeding its velocity limit: `|goal - current| / moving_time <= joint_velocity_limit`.

Condition 2 belongs to a motion that is about to be executed. For `can_reach` with `execute=False` it turns the question "can the arm be at this pose" into "can the arm get there within one `moving_time`".

## 5. Known issue: waypoint sampling fails with the streaming `moving_time`

**Observed** (2026-11-27, glassy/lboloni, Gazebo, `automove_gazebo_widowx_gazebo_cameras`): the automove rejected all 10000 sampled waypoints, `RuntimeError: Could not sample 10 reachable WidowX waypoints in 10000 attempts`. The arm was in the sleep pose `[0, -1.8, 1.55, 0, 0.8, 0]`. The log shows IK solutions that are valid joint configurations, for instance `[-0.06, -0.31, 1.06, -0.07, -0.93, 0.05]` for x=0.31, z=0.22, pitch=-0.18, still marked unreachable.

**Cause**: condition 2 of §4.1 with `moving_time: 0.1`. From the sleep pose, any waypoint of the workspace is more than 0.1 s of motion away at the joint velocity limits, so every sample fails. The real robot starts from the sleep pose too and would fail in the same way. The headless simulation did not show it, because `SimulatedPositionController.can_reach` has no kinematics.

### 5.1 Fix

`can_reach` checks only the joint position limits (condition 1). The speed is still checked by `move`, which runs the same Interbotix check with the real `moving_time` before publishing a step. This gives `can_reach` the same meaning as in the WidowX AI driver: the pose has an IK solution within the joint limits, found from the current joints.

The smallest change: during the check, set the arm's `moving_time` to infinity, which makes every speed in condition 2 zero, and restore it afterwards. `moving_time` is a public attribute of `InterbotixArmXSInterface` (set by `set_trajectory_time`); setting it directly, unlike `set_trajectory_time`, does not write the motor profile registers, so the check stays free of side effects.

```python
    def can_reach(self, pose):
        """IK check from the current joints, against the joint position limits only.
        Interbotix would also reject solutions that cannot be reached within one
        moving_time; that check is left to move, so it is switched off here."""
        if not isinstance(pose, WidowXPose):
            raise TypeError("WidowX reachability requires a WidowXPose")
        pose.validate(self.exp)
        joints = self.bot.arm.get_joint_positions()
        moving_time = self.bot.arm.moving_time
        self.bot.arm.moving_time = float("inf")
        started = perf_counter()
        try:
            solution, reachable = self.bot.arm.set_ee_pose_components(
                **self._pose_kwargs(pose), custom_guess=joints, execute=False,
            )
        finally:
            self.bot.arm.moving_time = moving_time
        logger.info(...)  # unchanged
        return bool(reachable)
```

**Consequence for the per-tick checks**: a step that has an IK solution but would need a jump (for instance to another IK branch near a singularity) is now accepted by `can_reach`, and then rejected by `move` with `ValueError`, which ends the collection. With 0.1 s steps from the current joints this should be rare; previously the gamepad controller silently dropped such a step. If it happens in practice, the gamepad controller can catch it, but that is not part of this fix.


## 6. Start and stop through the ready pose

### 6.1 Problem

The sleep pose (joints `[0, -1.8, 1.55, 0, 0.8, 0]`, gripper at x=0.12, z=0.08) is folded down on the rest, far outside the workspace of the demonstrations. Starting from it is a general problem, not only for the waypoint sampling of §5:

- the first streamed segment interpolates in task space from the sleep pose to the first target; the intermediate poses pass close to the base and may have no IK solution;
- every move from it is far more than one `moving_time` (0.1 s) of motion, so Interbotix rejects it (§4.1);
- the existing transitions have the same flaw: `go_sleep` / `go_home` call `go_to_sleep_pose` / `go_to_home_pose` with `moving_time=None`. Interbotix checks the speed with the arm's current `moving_time` (0.1 s after streaming) **before** it applies a given `moving_time`, so with the streaming configuration `shutdown_pose: sleep` is rejected from any pose in the workspace, and `startup_pose: home` from the sleep pose. The motion test of 2026-11-27 worked only because it called `set_trajectory_time(moving_time=3.0)` first.

Interbotix's home pose (all joints 0, gripper at x=0.46, z=0.36) is not a good place to start either: it is outside the workspace box of the random-movement runs (x <= 0.38, z <= 0.32), with the wrist in the singular configuration where `forearm_roll` and `wrist_rotate` are aligned.

### 6.2 Design

A **ready pose**, in the middle of the workspace, between the sleep pose and the demonstrations:

- **Ready pose = `POSE_DEFAULT`** (x=0.30, y=0, z=0.20, level gripper). This is already the pose where the headless `SimulatedPositionController` starts, so the simulated, the Gazebo and the real demonstrations all start from the same pose.
- **`READY_JOINTS` in the robot exp/run**: the joint configuration of the ready pose, `[0.0, -0.40, 1.09, 0.0, -0.70, 0.0]` (IK of `POSE_DEFAULT` with `modern_robotics` and `mr_descriptions.wx250s`, the same solution from the zero and from the sleep seed). Stored explicitly, not solved at runtime, so that the path is fixed and can be checked. The WidowX AI exp/runs have `SLEEP_JOINTS` in the same way.
- **Transitions are joint-space moves**, `set_joint_positions` with a long trajectory time (`transition_time`, 3 s), not task-space streams. A joint-space move has no IK and no intermediate pose to reach. Before the move, `set_trajectory_time(transition_time)`, so that the speed check of §4.1 uses it; afterwards back to the streaming `moving_time` and `accel_time`.
- **Stabilize**: after the blocking move, wait until every measured joint is within `settle_tolerance` (0.05 rad) of the goal; raise if it does not happen within `settle_timeout` (2 s).

**Start** (`startup_pose: ready`):

1. The arm must be in the sleep pose (measured joints within `sleep_tolerance`, 0.2 rad, of the sleep joints of Interbotix), otherwise raise with a message to park it. This keeps the joint-space path sleep → ready the one that has been checked. The tolerance is looser than `settle_tolerance`, because with the torque off the real arm rests on its cradle about 0.1 rad below the sleep joints (2026-11-27: shoulder -1.91 instead of -1.80, elbow 1.63 instead of 1.55).
2. Set the trajectory time to `transition_time` (`set_trajectory_time`, which writes the profile of the motors), then torque on for all motors (`robot_torque_enable("group", "all", True)`), then `capture_joint_positions()` (as Interbotix requires after the torque was off). When the torque comes on, the motors go to their last goal (the sleep joints commanded at the last stop) from where the arm rests on the cradle; with the transition time this takes 3 s instead of a 0.1 s jerk.
3. Transition to `READY_JOINTS`, stabilize.

**Stop** (`shutdown_pose: sleep`):

1. Transition to `READY_JOINTS`, stabilize. From the last pose of a demonstration (inside the workspace) this is a short move.
2. Transition to the sleep joints, stabilize.
3. Torque off for all motors (`robot_torque_enable("group", "all", False)`): the arm rests on its cradle. The `xs_sdk` driver keeps running; the next start switches the torque on again (step 2 of Start).

The transitions happen in `WidowXParticipant.start` / `stop`, which the recorder calls before the first and after the last tick (`DemonstrationRecorder.run`), so they are not part of the recorded demonstration: every demonstration starts at the ready pose.

**Waypoint sampling before the start**: the automove samples its waypoints in `bind`, before `start_robot`, with the arm still in the sleep pose. Before the start, `can_reach` seeds the IK with `READY_JOINTS` instead of the measured joints, since the arm will move from there (the WidowX AI `can_reach` uses its `SLEEP_JOINTS` before the start in the same way). The fix of §5 is still needed: before the start Interbotix's `joint_commands` are the sleep joints, so the speed check would still reject every sample.

### 6.3 Code sketch (`position_controller.py`)

```python
    def _transition(self, joints):
        """Joint-space move with the transition time, then wait until the joints settle."""
        self.bot.arm.set_trajectory_time(
            moving_time=self.exp["transition_time"], accel_time=self.exp["transition_time"] / 4)
        if not self.bot.arm.set_joint_positions(joints, blocking=True):
            raise ValueError(f"Interbotix rejected the WidowX transition to {joints}")
        deadline = monotonic() + self.exp["settle_timeout"]
        while not self._at_joints(joints):
            if monotonic() > deadline:
                raise RuntimeError(f"WidowX did not settle at {joints}")
            sleep(0.05)
        self.bot.arm.set_trajectory_time(
            moving_time=self.exp["moving_time"], accel_time=self.exp["accel_time"])

    def _at_joints(self, joints):
        return max(abs(m - j) for m, j in zip(self.bot.arm.get_joint_positions(), joints)) \
            <= self.exp["settle_tolerance"]
```

In `start_robot`, a new branch:

```python
            elif startup_pose == "ready":
                if not self._at_joints(self.bot.arm.group_info.joint_sleep_positions,
                                       self.exp["sleep_tolerance"]):
                    raise RuntimeError("WidowX must start from the sleep pose; park it first")
                self.bot.arm.set_trajectory_time(
                    moving_time=self.exp["transition_time"],
                    accel_time=self.exp["transition_time"] / 4,
                )
                self.bot.core.robot_torque_enable("group", "all", True)
                self.bot.arm.capture_joint_positions()
                self._transition(self.exp["READY_JOINTS"])
```

In `stop_robot`, `sleep` becomes:

```python
            elif shutdown_pose == "sleep":
                self._transition(self.exp["READY_JOINTS"])
                self._transition(self.bot.arm.group_info.joint_sleep_positions)
                self.bot.core.robot_torque_enable("group", "all", False)
```

and `home` uses `_transition` with the zero joints, so that it is no longer rejected (§6.1). In `can_reach`, the seed is `self.bot.arm.get_joint_positions() if self.started else self.exp["READY_JOINTS"]`. The existing options `hold` and `default` stay as they are; `default` streams to `POSE_DEFAULT` with `moving_time` and has the flaw of §6.1, it is superseded by `ready`.

New fields of both `robot_widowx` exp/runs (`position_controller_wx250s_00`, `position_controller_gazebo_wx250s_00`):

```yaml
# "ready": from the sleep pose, torque on, joint-space move to READY_JOINTS (= POSE_DEFAULT).
# "sleep": through READY_JOINTS to the sleep pose, then torque off.
startup_pose: ready
shutdown_pose: sleep
READY_JOINTS: [0.0, -0.40, 1.09, 0.0, -0.70, 0.0]
transition_time: 3.0
settle_tolerance: 0.05
settle_timeout: 2.0
sleep_tolerance: 0.2
```

`SimulatedPositionController` needs no change: it starts at `POSE_DEFAULT`, which is the ready pose, and ignores the new fields.

## 7. Implementation plan (§5 and §6)

Steps 1-6 done on 2026-11-27 (glassy/lboloni). Step 7: start and stop done on the real robot (two cycles, the second from the torqued-off arm on the cradle, see §8); the automove episode with half velocities is open.

1. **Code** (`position_controller.py`): `can_reach` as in §5.1, with the seed of §6.2; `_transition`, `_at_joints`; the `ready` branch of `start_robot`; `sleep` and `home` of `stop_robot` and `go_sleep` / `go_home` through `_transition`.
2. **Exp/runs**: the new fields of §6.3 in both `robot_widowx` runs.
3. **Test support** (`test/robot/widowx/widowx_test_support.py`): the test exp gets the new fields; `FakeArm` gets `moving_time` (recorded in `pose_calls` at each call), `set_trajectory_time`, `set_joint_positions` (sets `joints`), `capture_joint_positions`, and `group_info.joint_sleep_positions`; `FakeCore` gets `robot_torque_enable` (records the calls).
4. **Unit tests** (`test/robot/widowx/test_widowx_controller.py`):
   - `can_reach` runs with an infinite `moving_time`, which is restored afterwards; before the start it seeds with `READY_JOINTS`, after the start with the measured joints;
   - start with `ready` from the sleep joints: torque on, then the joints are `READY_JOINTS`, and the trajectory time is back to the streaming one;
   - start with `ready` from another pose raises;
   - stop with `sleep`: `READY_JOINTS`, then the sleep joints, then torque off, in this order;
   - a transition that does not settle raises.
5. **Run the tests**: `pytest src/test/robot/widowx src/test/remote_control/test_widowx_automove_controller.py src/test/demonstration/test_widowx_participant.py`, in the regular venv and in `berrypickervenv-ros` inside the container.
6. **Gazebo** (glassy/lboloni): Gazebo spawns the arm in the sleep pose. Collect one episode with `automove_gazebo_widowx_gazebo_cameras`, seed 73017; watch sleep → ready → waypoints → ready → sleep in the Gazebo window. Check in `collection.log`: the waypoints are sampled (`reachable=True` among the `IK check` lines), the per-tick IK durations against the 0.1 s tick, the first recorded pose is `POSE_DEFAULT`.
7. **Real robot** (glassy/lboloni, `xs_sdk`, hand on the power switch): first only start and stop (a run with `waypoint_count` 1 or the controller alone): sleep → ready → sleep → torque off, then start again from the torqued-off sleep pose, checking that the arm does not jump when the torque comes on. Then one automove episode with half velocities.
8. **Document**: mark §5 and §6 as implemented here; update `PLAN-WidowXRandomMovement.md` (Steps 4 and 5).

## 8. Open questions

- **IK seed for sampling**: with the `READY_JOINTS` seed the sampling should converge much better than from the sleep joints (where many samples ended at joint values in the hundreds of radians). If not, `can_reach` can retry with Interbotix's `initial_guesses`. First Gazebo run (2026-11-27): 10 waypoints in 10 attempts.
- **Start and stop through the ready pose, first Gazebo run (2026-11-27, glassy/lboloni)**: confirmed by watching and by a joint-state log: sleep → `READY_JOINTS` in about 3 s, the waypoints, then back to `READY_JOINTS` and sleep.
- **Wrist singularity with a fixed yaw (fixed by option 1, see §9)**: the random-pose automove (`automove_widowx_random_pose_00`) fixes `yaw: 0` (and `roll: 0`), so the gripper always points along +x, whatever the waist heading. For `y != 0` the wrist must twist back against the waist: `forearm_roll` and `wrist_rotate` move in opposite directions (up to ±1.05 rad at a waist of 0.73 rad in the log). With a pitch near 0, `wrist_angle` is near 0 and these two axes are aligned: this is the singular configuration, in which the numerical IK can jump between solutions. On 2026-11-27, at tick 81, the per-tick `can_reach` accepted a step (after §5.1 it no longer checks the speed) whose solution moved `forearm_roll` from -0.74 to -0.11 and `wrist_rotate` from 0.73 to 0.11, about 6 rad/s; `move` rejected it (`Interbotix could not reach WidowX target`, x=0.29, y=0.00, z=0.28, pitch=0.11) and the collection ended. Before §5.1 the same step would have been rejected by `can_reach`, and the automove would have raised `intermediate pose is unreachable` instead. Possible fixes, to be decided:
  1. **Yaw follows the heading** (recommended): sample the yaw as `atan2(y, x)` (optionally plus a small random offset), so that the gripper points away from the base, as on the 5-DOF AL5D. `forearm_roll` and `wrist_rotate` then stay near 0 and the arm stays away from the singularity. Needs a new pose option in the automove (for instance `yaw: {heading: [-0.2, 0.2]}`) and in `move_pose_towards` the yaw interpolated along with it.
  2. **Keep the pitch away from 0**: for instance pitch in [-0.6, -0.2]. Avoids the alignment of the two axes but narrows the workspace, and the large wrist twist remains.
  3. **Fixed `forearm_roll`**: solve the IK with `forearm_roll` locked at 0, i.e. treat the arm as 5-DOF, so that the yaw cannot be chosen freely. Larger change, in the IK itself.
- **Gazebo camera rate (open, worked around on glassy)**: `simulation/fixed_camera.sdf` sets `<update_rate>10</update_rate>` at 512x512. On glassy (RTX 5090, Gazebo real-time factor 1.00, GPU at 22%) `ros2 topic hz` shows about 8 Hz, with gaps of up to 0.4 s between frames. `RosImageCameraController` (`controllers/gazebo_cameras`, `max_frame_age: 0.25`) then raises `Stale ROS camera frames` and ends the collection (2026-11-27, at tick 52, a frame 0.31 s old). Workaround: `controllers/gazebo_cameras_sysdep.yaml` on glassy with `max_frame_age: 0.5`, which accepts repeated frames in the recording. To be done later: find where the frames are lost (the rendering in `gzserver`, the GUI client, or the delivery to the collector) and fix it there, for instance with a higher `update_rate` (20) in the sdf, so that a 10 Hz recording gets a fresh frame on every tick; then remove the workaround.
- **Torque on at the start**: the Dynamixel motors keep their last goal position (the sleep joints) while the torque is off, but the real arm sags onto its cradle, about 0.1 rad below them. Switching the torque on therefore moves the arm back up to the last goal, with the profile set at that moment; hence the transition time is set before (§6.2, Start, step 2). Confirmed on the real robot (2026-11-27, glassy/lboloni): two start/stop cycles, the second from the torqued-off arm on the cradle (shoulder -1.91). Each start took 3.5 s to `READY_JOINTS` (measured within 0.03 rad; the gripper at x=0.295, z=0.190, about 1 cm below `POSE_DEFAULT` from gravity), each stop 6.9 s. In a joint log at 10 Hz every move was smooth, at most 0.72 rad/s, with no jerk when the torque came on.
- **Trajectory time logged**: the Interbotix motion test of 2026-11-27 called `set_trajectory_time(moving_time=3.0)`, while the startup log showed the default 2.0 s; the log is written at construction, before the call. Not an issue, noted to avoid confusion.

## 9. Yaw that follows the heading (fix of the wrist singularity, §8)

### 9.1 Design

A new kind of pose field for the random-pose automove (`remote_control/widowx_automove_controller.py`), allowed only for `yaw`:

```yaml
pose:
  yaw: {heading: [0.0, 0.0]}
```

The yaw is the heading of the gripper position seen from the base, plus an offset: `yaw = atan2(y, x) + offset`, with the offset drawn uniformly from the range, wrapped to [-π, π]. With offset 0 the gripper points straight away from the base, as on the 5-DOF AL5D, so `forearm_roll` and `wrist_rotate` stay near 0 and the arm stays away from the singularity where they are aligned. The offset range is kept in the configuration, but for the WidowX 250s it must be 0 (§9.3).

**Sampling** (`_sample_pose`): the fields are sampled in the order of `WidowXPose.FIELDS`, so `x` and `y` are known when `yaw` is computed.

**Approach** (`next_command`): the fields of the step are moved independently towards the waypoint (`move_pose_towards`), each with its own velocity. Interpolating the yaw in the same way would let it arrive earlier or later than the position, and the gripper would turn away from the heading in between, which brings back the wrist twist. With `heading`, the yaw of every step is therefore recomputed from the position of the step: the offset (current yaw minus the current heading) is moved towards the offset of the waypoint with the yaw velocity (`move_towards`, angular), and the yaw of the step is the heading of the step plus that offset. The yaw then changes by the heading change of the step (at most the xy velocity over the distance from the base, 0.08 / 0.22 ≈ 0.4 rad/s in the workspace of the random-movement runs) plus the offset change (at most the yaw velocity).

**Limits check** (`_read_pose_fields`): the minimum and maximum poses checked against `POSE_MIN`/`POSE_MAX` use the offset range for the yaw, since the heading depends on the sampled position; the yaw of every sampled pose is checked anyway when the pose is created.

**Exp/run**: `automove_widowx_random_pose_00` changes from `yaw: {fixed: 0.0}` to `yaw: {heading: [0.0, 0.0]}`. This changes the run on every machine; with a fixed yaw it runs into the singularity on any machine with the real IK (Gazebo or the real robot). The headless simulation is not affected, apart from the different yaw values.

### 9.2 Implementation and tests

1. `widowx_automove_controller.py`: `heading` in `_read_pose_fields` (only for `yaw`), in `_sample_pose`, and the yaw recomputation in `next_command`.
2. `data/expruns/automove/automove_widowx_random_pose_00.yaml`: `yaw: {heading: [0.0, 0.0]}` (first `[-0.2, 0.2]`, see §9.3).
3. Tests (`test/remote_control/test_widowx_automove_controller.py`): sampled waypoints have `yaw - atan2(y, x)` within the offset range; every step of an approach has `yaw - atan2(y, x)` moving monotonically from the start offset to the waypoint offset; `heading` for another field is rejected.
4. Gazebo (glassy/lboloni): a full episode with `automove_gazebo_widowx_gazebo_cameras`, seed 73017; in the joint log `forearm_roll` and `wrist_rotate` stay within about ±0.3 rad.

### 9.3 Why the offset must be 0

The first Gazebo run with `heading: [-0.2, 0.2]` (2026-11-27, glassy/lboloni) failed at tick 51 in the same way as with the fixed yaw: a step at x=0.27, y=-0.10, z=0.29, pitch=0.00, with a yaw offset of only -0.09 rad, needed `forearm_roll` to go from 1.33 to 0.42 and `wrist_rotate` from -1.31 to -0.42 in one tick.

The reason is the geometry of the wrist. Relative to the forearm, the gripper has to bend vertically (by the difference between the gripper pitch and the forearm pitch) and sideways (by the yaw offset from the heading). The wrist does this by turning `forearm_roll` to the direction of the bend, `atan2(sideways, vertical)`, bending by `wrist_angle`, and turning back with `wrist_rotate` to keep the roll 0. With a sideways bend that is not 0, whenever the vertical bend passes through 0 during a move (which happens for many pairs of waypoints with the random pitch), `forearm_roll` has to swing through ±π/2 within a short distance, and the numerical IK jumps. With the fixed yaw of §8 the sideways bend was the whole heading, so this happened even more often.

With offset 0 the bend is purely vertical, `forearm_roll` and `wrist_rotate` stay at 0, and passing through the straight wrist is harmless. So the WidowX 250s random movements keep the gripper pointing away from the base, like the 5-DOF AL5D. If a yaw offset is wanted later, the pitch range has to keep the vertical bend away from 0, or the IK has to handle the wrist flip (§8, option 3).

### 9.4 Result

Gazebo, 2026-11-27, glassy/lboloni, `automove_gazebo_widowx_gazebo_cameras`, seed 73017 (`data/demonstration/freeform/2026_11_27__16_07_17`): the full episode, 10 waypoints, 136 steps, 136 frames. The first recorded pose is the ready pose (x=0.30, y=0, z=0.20). `forearm_roll` stayed within [-0.02, 0.03] rad and `wrist_rotate` within [-0.03, 0.02]; the largest joint step per tick was 0.15 rad (`wrist_angle`). No tick overran (largest work 16 ms of the 0.1 s tick). The joint log showed sleep → ready → waypoints → ready → sleep. The largest camera frame age was 0.31 s, which passed only with the glassy workaround of §8.

## 10. A soft stop: no drop when the torque goes off

### 10.1 Problem

At the end of a stop (§6.2) the arm holds the nominal sleep joints of Interbotix (shoulder -1.80, elbow 1.55), and then the torque goes off. The real arm rests on its cradle about 0.1 rad lower (shoulder -1.91, elbow 1.63), so it drops onto the cradle, with an audible knock (2026-11-27, glassy/lboloni). Two remedies, both implemented, each can be switched on or off:

### 10.2 A: park where the arm rests

`start_robot` with `startup_pose: ready` records the measured joints of the parked arm (`rest_joints`) before it switches the torque on: this is where the arm rests on its cradle. `go_sleep` goes through `READY_JOINTS` to `rest_joints` instead of the nominal sleep joints; before such a start (other startup poses) it uses the nominal sleep joints. The arm is then on the cradle when the torque goes off, and has nothing to fall. A start from up to `sleep_tolerance` away from the nominal sleep joints returns there at the stop, which is also a place where the arm was resting.

**Joint limits**: the cradle is beyond the joint limits of Interbotix: the shoulder rests at -1.911 (lower limit -1.885), the elbow at 1.649 (upper limit 1.606). Interbotix rejects goals outside the limits (the first real-robot try of A failed with `Interbotix rejected the WidowX transition`, and left the arm at the ready pose), so the rest joints are clamped into `group_info.joint_lower_limits` / `joint_upper_limits`. The arm then stops about 0.03-0.04 rad above the cradle instead of 0.11 rad.

Remaining risk: the clamping, and the motors stopping within `settle_tolerance` of the goal, leave a small drop; B removes it.

### 10.3 B: lower the motor output before the torque goes off

The Dynamixel X-series register `Goal_PWM` (0-885) limits the output voltage of the motor in every operating mode, also in the position mode used by Interbotix, and can be written with the torque on. After the arm has reached its rest joints, the stop sets `Goal_PWM` to `pwm_ramp_start` and lowers it linearly to 0 over `torque_off_ramp_time` (2 s, 20 steps), so that the arm settles onto the cradle under gravity, and only then switches the torque off. `pwm_ramp_start` is 90, about 10 % of `pwm_max`: holding the arm needs only a few percent (§10.6), so a ramp from `pwm_max` would spend nearly all its time where nothing happens (refinement of 2026-11-27; the first version ramped from `pwm_max`). With the torque off, `Goal_PWM` is set back to `pwm_max`, and it is set to `pwm_max` again at the start before the torque comes on, so that an interrupted stop does not leave a weak arm. `torque_off_ramp_time: 0` switches B off.

The writes go through `bot.core.robot_set_motor_registers("single", name, "Goal_PWM", value)`, one motor at a time: a group write (`"group", "arm"`) reaches only the joints of the group, not the shadow motors (`InterbotixDriverXS::set_motor_registers`). The motors are listed in the exp/run as `pwm_ramp_motors`: the motors that carry the arm, `shoulder`, `shoulder_shadow`, `elbow`, `elbow_shadow`, `wrist_angle` (the first version had all six arm joints and the two shadows; fewer motors make every step of the ramp shorter). The gripper is not in the list: it runs in PWM mode, where `Goal_PWM` is its command.

Not used: `Goal_Current` limits the current only in the current-based position mode, and switching the operating mode needs the torque off, which is the drop to avoid.

To check on the real robot: that `Goal_PWM` indeed limits the output in position mode (the arm sinks during the ramp), and that the writes reach the shadow motors (otherwise the shoulder and the elbow keep holding through their shadows).

### 10.4 Configuration

```yaml
# soft stop (section 10): the stop parks at the joints where the arm rested at the
# start; then lowers Goal_PWM over torque_off_ramp_time (0: off) before torque off
torque_off_ramp_time: 2.0
pwm_max: 885
pwm_ramp_start: 90
pwm_ramp_motors: [shoulder, shoulder_shadow, elbow, elbow_shadow, wrist_angle]
```

A has no switch: it applies whenever the start was `ready`.

### 10.5 Plan

1. `position_controller.py`: `rest_joints` in `start_robot`; `go_sleep` to `rest_joints`; `_set_pwm(value)` over `pwm_ramp_motors`; the ramp and the restore in `stop_robot`; the restore before the torque on in `start_robot`.
2. Both `robot_widowx` exp/runs: the fields of §10.4.
3. Tests: the fakes record the register writes; the stop goes to the rest joints, writes a decreasing `Goal_PWM` ending at 0 before the torque off, and `pwm_max` after; the start writes `pwm_max` before the torque on; with `torque_off_ramp_time: 0` no register is written at the stop.
4. Real robot, glassy/lboloni: one start/stop cycle with A only (`torque_off_ramp_time: 0`), one with A and B, listening for the knock and logging the joints at 10 Hz (the drop at the torque off is visible as a step in shoulder and elbow).

### 10.6 Result on the real robot (2026-11-27, glassy/lboloni)

Two cycles from the torqued-off arm on the cradle (shoulder -1.911, elbow 1.649), joints logged at 10 Hz:

- **A only** (`torque_off_ramp_time: 0`, two runs): the stop went to the clamped rest joints (-1.885, 1.606); the arm settled at -1.893 / 1.612, the torque went off, and within one sample (0.2-0.3 s) it was on the cradle at -1.911 / 1.649: a drop of about 0.02 / 0.04 rad instead of 0.11 / 0.08 without A. Stop 8.4 s. (The first reading of the first run took this drop for a move with the torque on; the second run, logged around the stop, showed the order.)
- **A and B** (`torque_off_ramp_time: 2.0`): the arm held at -1.893 / 1.614 during the whole ramp, then fell to -1.910 / 1.649 within one sample (0.3 s) at its end: the same drop as with A only, no gradual sinking. Stop 18.9 s: the ramp takes about 9 s instead of 2, since every one of the 168 `Goal_PWM` writes is a separate service call (about 50 ms each).

The remaining drop comes from the clamping: the cradle is 0.026 rad below the shoulder limit and 0.043 rad above the elbow limit. These limits are also the position limits in the EEPROM of the motors (`wx250s.yaml`), so the motors cannot be commanded onto the cradle.

Open: whether `Goal_PWM` limits the output in position mode at all (read it back during a ramp with `robot_get_motor_registers`), or only the last steps matter (holding the arm against gravity may need only a few percent of the PWM, so a linear ramp from 885 releases it in its last step). If B is kept: ramp only the motors that carry the arm (shoulder, shoulder_shadow, elbow, elbow_shadow, wrist_angle), and spend the ramp in the low range (for instance from 20 % to 0).

**First decision (2026-11-27)**: A and B together, B still ramping from `pwm_max`. Listening at the robot: with A only there was still a knock; with A and B the stop was quiet ("nice"). In the joint log of that run the shoulder began to sink in the last step before 0 (5 % of `pwm_max`) and reached the cradle over 0.2 s instead of at once: `Goal_PWM` does limit the output in position mode, but holding the arm needs only a few percent of it. Possible refinement, not needed now: start the ramp at about 10 % and ramp only the motors that carry the arm, which would spread the sinking over the ramp and shorten the stop (now 18.9 s).

**Refinement tried (2026-11-27, glassy/lboloni)**: `pwm_ramp_start: 90`, 5 motors. The ramp took about 5.5 s (the 5 writes per step take longer than the 0.1 s step), the stop 16.1 s instead of 18.9 s. The arm held at -1.893 / 1.614 until the last quarter of the ramp, then sank over 0.3 s to -1.908 / 1.646, and the torque off moved it only the last 0.003 rad onto the cradle (before the refinement: 0.017 / 0.035 rad at once). The sinking starts at about 20 of 885 (2 %); `pwm_ramp_start` could go down to about 45 to spread it over the whole ramp.

**Decision (2026-11-27): this is the way to stop the WidowX 250s.** Confirmed by listening at the robot in two runs ("fine"). Every stop of a `robot_widowx` controller with `shutdown_pose: sleep`:

1. joint-space move to `READY_JOINTS`, settle (§6);
2. joint-space move to the rest joints recorded at the start, clamped into the joint limits, settle (A, §10.2);
3. `Goal_PWM` of `shoulder`, `shoulder_shadow`, `elbow`, `elbow_shadow`, `wrist_angle` from 90 down to 0 in 20 steps over `torque_off_ramp_time: 2.0` (B, §10.3); the arm sinks onto the cradle over about 0.3 s near the end;
4. torque off for all motors; `Goal_PWM` back to `pwm_max`.

Keep these settings (`torque_off_ramp_time: 2.0`, `pwm_ramp_start: 90`, the five `pwm_ramp_motors`) in both `robot_widowx` exp/runs, and do not stop the real robot by switching off the torque, killing `xs_sdk`, or with Interbotix's `go_to_sleep_pose` (the arm drops onto the cradle with a knock). Lowering `pwm_ramp_start` to about 45 would spread the sinking further, but is not needed.
