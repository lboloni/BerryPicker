# How to collect demonstrations

This guide describes the participant-based demonstration collector. A
collection recipe says what participates in a demonstration; the local machine
profile says which physical devices are available on the current computer.

## Set up a machine

1. Point `~/.config/BerryPicker/mainsettings.yaml` at that machine's local
   `settings-<machine>.yaml` file.

2. In the directory named by `experiment_system_dependent_dir`, create these
   files from the templates in `src/experiment_configs`:

   ```text
   machine/current_sysdep.yaml
   controllers/fixed_cameras_sysdep.yaml
   robot_al5d/pulse_controller_00_sysdep.yaml
   ```

3. In `machine/current_sysdep.yaml`, set `machine_name` and configure every
   binding that the machine can actually use. Set every unverified or absent
   device to `available: false`. The collector raises an exception if a recipe
   requests an unavailable binding.

4. In `controllers/fixed_cameras_sysdep.yaml`, give cameras stable view names
   and their local OpenCV device numbers:

   ```yaml
   views:
     overhead:
       device: 0
     side:
       device: 1
   saved_image_size: [512, 512]
   ```

5. If the machine has a physical AL5D, set its serial devices in
   `robot_al5d/pulse_controller_00_sysdep.yaml`:

   ```yaml
   device: "/dev/ttyUSB0"
   device_backup: "/dev/ttyUSB1"
   ```

6. Verify cameras and physical devices before enabling their bindings. Do not
   mark a device available merely because its configuration file exists.

The committed `machine/current.yaml` contains no hardware details. Keep actual
device mappings and serial ports in the local `*_sysdep.yaml` files.

## Start a collection

Open `Collect_Demonstration.ipynb`. Its collection cell contains:

```python
collection_exp = Config().get_experiment(
    "demonstration_collector", "xbox_al5d_cameras")
```

Replace the run name with one of the recipes below, then run the notebook from
top to bottom. It creates a timestamped demonstration directory, starts the
configured participants, and saves metadata when collection stops.

The configured participant order is meaningful. In particular, a mobile camera
must be placed after the AL5D and before fixed cameras if fixed camera images
must show the new mobile-camera viewpoint.

## Collect with AutoMove

Select the recipe:

```python
"automove_al5d_cameras"
```

For a machine without the physical AL5D, select:

```python
"automove_simulated_cameras"
```

The machine profile's `automove` binding selects the AutoMove run. The normal
run is `automove_random_robot_position_00`, which has a required random seed,
fixed wrist/gripper values, and explicit velocities for every `RobotPosition`
field.

Other supplied AutoMove runs are:

```text
automove_random_ee_box_00
automove_random_ee_plane_z5_00
```

To use one, change the `run` in the local machine profile's `automove` binding.
Each run is reproducible with its `random_seed`; change that integer to collect
a different path. The collector records the AutoMove type and seed with every
demonstration timestep.

For end-effector box or plane runs, the AL5D initial position must already be
inside the configured box or plane. The collector raises an exception instead
of moving into the workspace from outside it.

## Collect with an XBox controller

Select:

```python
"xbox_al5d_cameras"
```

The local machine profile must have an available `xbox` binding, an available
physical `al5d` binding, and an available `fixed_cameras` binding. Connect the
controller before starting the notebook. The configured exit button ends
collection and releases the controller, robot, and cameras.

If the profile says `available: false` for XBox, verify the controller and its
`approxeng` dependency first, then change only that local binding to `true`.

## Collect with the keyboard controller

Select:

```python
"keyboard_al5d_cameras"
```

This recipe captures the fixed-camera image first, then reads the OpenCV key
from that display, then sends the resulting target to the AL5D. Keep the camera
window focused while controlling the robot. Key mappings are configured in the
`controllers/keyboard_controller` experiment; `x` is the configured exit key
in the supplied run.

The keyboard recipe requires available `keyboard`, `al5d`, and
`fixed_cameras` bindings.

## Collect with WidowX as leader

Select:

```python
"widowx_al5d_cameras"
```

The WidowX provides the leader pose; the AL5D follows the converted target.
The local profile must enable `widowx_leader`, `al5d`, and `fixed_cameras`.
Confirm that the WidowX is in the intended backdrivable leader configuration
before collection.

For the mobile-camera variant, select:

```python
"widowx_al5d_mobile_camera"
```

This additionally requires an available `mobile_camera` binding. If the leader
and mobile camera are the same physical WidowX, configure both bindings with
the same `resources` value; the collector will reject the invalid recipe rather
than attempt to acquire that robot twice.

## Stop and inspect a collection

Every recorded timestep contains one AL5D action, participant telemetry, and
images from every fixed camera view. The recorder saves metadata when the run
ends. The final notebook cell can convert saved image sequences to video; run
it only after confirming that the recorded images and metadata are correct.
