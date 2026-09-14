# WidowX XBox end-effector controller

The WidowX XBox end-effector controller converts XBox-style gamepad input into
Cartesian WidowX commands. It controls the end-effector position and
orientation through the Interbotix end-effector pose interface. The gripper is
commanded separately.

The implementation is in
[widowx_xbox_end_effector_controller.py](widowx_xbox_end_effector_controller.py).
Its default experiment is
[widowx_xbox_end_effector_controller_00.yaml](../../data/expruns/controllers/widowx_xbox_end_effector_controller_00.yaml).

## Requirements

The controller requires Linux, the <code>approxeng.input</code> package, and a
connected XBox-compatible controller. The current configuration was developed
for the Voyee controller described by
[microsoft_xbox_360_pad_v1118_p654.yaml](../install/microsoft_xbox_360_pad_v1118_p654.yaml).

Controller names reported by <code>approxeng.input</code> can differ between
devices. The experiment configuration uses the names from that Voyee profile.

## Controls

All three end-effector rotations have direct mappings. No orientation mode or
mode-switch button is used.

| Physical control | Input name | End-effector action |
|---|---|---|
| Left stick vertical | <code>ly</code> | Move along x |
| Left stick horizontal | <code>lx</code> | Move along y |
| Right stick vertical | <code>ry</code> | Move along z |
| Right stick horizontal | <code>rx</code> | Change yaw |
| Left trigger minus right trigger | <code>lt - rt</code> | Change roll |
| D-pad up/down | <code>dy</code> | Change pitch |
| X | <code>square</code> | End collection |
| Guide/Home | <code>home</code> | Restore the captured startup pose |
| LB | <code>l1</code> | Release the gripper |
| RB | <code>r1</code> | Grasp with the configured pressure |

The current controller profile is expected to report D-pad up as
<code>dy=-1</code>. The controller negates this value, so D-pad up increases
pitch and D-pad down decreases pitch. The D-pad is digital and therefore
changes pitch at the full configured pitch velocity while held. Stick and
trigger inputs remain analog.

Roll, pitch, and yaw inputs are applied together during the same collection
tick. The Y button, A button, B button, horizontal D-pad, and stick buttons are
currently unused.

## Experiment configuration

The controller experiment defines its buttons, input timestep limit, gripper
pressure, and Cartesian velocities:

~~~yaml
button_exit: square
button_home: home
button_release: l1
button_grasp: r1

max_input_dt: 0.25
gripper_pressure: 0.5

velocity:
  x: 0.10
  y: 0.10
  z: 0.08
  roll: 0.50
  pitch: 0.50
  yaw: 0.50
~~~

Translation velocities are meters per second. Rotation velocities are radians
per second. Each input is scaled by the elapsed collection time, limited by
both <code>max_input_dt</code> and the robot controller's
<code>moving_time</code>.

Configure the controller in the local machine profile with:

~~~yaml
widowx_xbox_end_effector:
  factory: widowx_xbox_end_effector_leader
  exp: controllers
  run: widowx_xbox_end_effector_controller_00
  available: true
  resources: [xbox]
~~~

Only enable the binding after verifying that the controller is connected and
its input names match the configured device profile.

## Starting and stopping

Start the physical or simulated WidowX services required by the selected
demonstration recipe. Then open
[Collect_Demonstration.ipynb](../demonstration/Collect_Demonstration.ipynb) and
select one of these collector runs:

- <code>xbox_widowx_cameras</code> for the physical WidowX and USB cameras.
- <code>xbox_simulated_widowx_cameras</code> for the in-process simulated
  WidowX and USB cameras.
- <code>xbox_gazebo_widowx_gazebo_cameras</code> for the Gazebo WidowX and its
  rendered camera.
- <code>xbox_gazebo_widowx_usb_gazebo_cameras</code> for the Gazebo WidowX,
  USB cameras, and its rendered camera.

The controller captures the actual end-effector pose on its first collection
update. This pose becomes both the initial command target and the target
restored by Guide/Home. It is not the Interbotix named home pose.

Press X to request an orderly stop. Disconnecting the gamepad also stops the
collection. Full robot, camera, and simulator startup instructions are in
[HOWTO-DEMONSTRATION.md](../demonstration/HOWTO-DEMONSTRATION.md).

## Command validation

Before sending a changed pose to the robot, the controller:

1. Limits every Cartesian component to the configured WidowX pose range.
2. Asks the robot controller to perform an inverse-kinematics reachability
   check.
3. Retains the preceding target if the candidate is unreachable.
4. Increments <code>rejected_target_count</code> for rejected targets.

LB and RB issue one-shot gripper commands. Holding both simultaneously raises
an exception. If both presses accumulated between collection polls, the
currently held button determines the action; if neither remains held, the
controller keeps the previous gripper state.

The sampled controller telemetry contains the current target,
<code>last_target_rejected</code>, and <code>rejected_target_count</code>.
Detailed input and IK timing diagnostics use the
<code>remote_control.widowx_xbox_end_effector_controller</code> logger.
