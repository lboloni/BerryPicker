# Running demonstration collection

This directory contains
[`Collect_Demonstration.ipynb`](Collect_Demonstration.ipynb), the interactive
entry point for recording demonstrations. The collector recipe chosen in the
notebook determines the controller, robot, and cameras used during a run.

## Machine configuration

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

## Common collection workflow

Commands in this guide assume the current directory is the BerryPicker
repository root. Before opening the notebook, activate the BerryPicker Python
environment. WidowX configurations also require the ROS 2 and Interbotix
environments:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
```

For each demonstration, start a fresh notebook kernel from this directory:

```bash
cd src/demonstration
jupyter lab Collect_Demonstration.ipynb
```

In the recipe-selection cell, leave exactly one active `collector_run`
assignment. Run the notebook through the `recorder.run()` cell. End collection
with the exit control configured for the selected leader. Inspect the generated
demonstration directory and `collection.log` before running the final video
conversion cell: that cell uses `delete_img_files=True` and removes the source
images after conversion.

The configured participant order is meaningful. In particular, a mobile camera
must be placed after the AL5D and before fixed cameras if fixed camera images
must show the new mobile-camera viewpoint.

## AL5D robot with USB cameras

This configuration talks directly to the physical AL5D and collects every view
listed in the local `controllers/fixed_cameras_sysdep.yaml`.

The AL5D controller opens its configured serial device when the recorder is
created, so it needs no separate ROS launch process.

### XBox control

Connect and verify the AL5D, USB cameras, and XBox controller. Enable the
`xbox`, `al5d`, and `fixed_cameras` bindings in the machine configuration.
Then select the XBox-controlled recipe in the notebook:

```python
collector_run = "xbox_al5d_cameras"
```

The configured XBox exit button ends collection and releases the controller,
robot, and cameras. If the XBox binding is unavailable, verify the controller
and its `approxeng` dependency before enabling that local binding.

### Automatic motion

For an automatically generated trajectory, select one of these recipes instead:

```python
collector_run = "automove_al5d_cameras_random_robot_position_00"
# collector_run = "automove_al5d_cameras_random_robot_position_01"
# collector_run = "automove_al5d_cameras_random_ee_box_00"
# collector_run = "automove_al5d_cameras_random_ee_plane_z5_00"
```

AutoMove recipes require the `automove`, `al5d`, and `fixed_cameras` bindings.
For box and plane runs, place the end effector inside the configured workspace
before starting collection.

For a machine without the physical AL5D, select:

```python
collector_run = "automove_simulated_cameras"
```

The collector recipe selects the AutoMove run. The machine profile's
`automove` binding only says whether this machine can run AutoMove and which
participant factory to use.

Each collector recipe points to an `automove` experiment config. That config
contains the AutoMove type, random seed, workspace bounds or plane definition,
and motion parameters. Add a new `automove/*.yaml` and matching
`demonstration_collector/*.yaml` recipe to collect a different parameter set.

### Keyboard control

Select:

```python
collector_run = "keyboard_al5d_cameras"
```

This recipe captures the fixed-camera image first, then reads the OpenCV key
from that display, then sends the resulting target to the AL5D. Keep the camera
window focused while controlling the robot. Key mappings are configured in the
`controllers/keyboard_controller` experiment; `x` is the configured exit key
in the supplied run.

The keyboard recipe requires available `keyboard`, `al5d`, and
`fixed_cameras` bindings.

### WidowX leader

Select:

```python
collector_run = "widowx_al5d_cameras"
```

The WidowX provides the leader pose; the AL5D follows the converted target.
The local profile must enable `widowx_leader`, `al5d`, and `fixed_cameras`.
Confirm that the WidowX is in the intended backdrivable leader configuration
before collection.

For the mobile-camera variant, select:

```python
collector_run = "widowx_al5d_mobile_camera"
```

This additionally requires an available `mobile_camera` binding. If the leader
and mobile camera are the same physical WidowX, configure both bindings with
the same `resources` value; the collector will reject the invalid recipe rather
than attempt to acquire that robot twice.

## Gazebo-simulated WidowX

This configuration starts a Gazebo WidowX named `wx250s`, an Interbotix SDK
simulator named `wx250s_sdk`, a bridge between them, and a fixed rendered camera
on `/berrypicker/gazebo/camera_0/image_raw`.

Install the Gazebo ROS camera plugin once if it is not present:

```bash
sudo apt install ros-humble-gazebo-plugins
```

In the first terminal, source ROS and start the complete simulation:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
python src/robot/widowx/simulation/run_gazebo.py
```

Wait for the launcher to print that the Gazebo camera and Interbotix controller
are ready. The machine configuration must enable `widowx_xbox`,
`gazebo_widowx`, and `gazebo_cameras`. Enable `fixed_cameras` as well when USB
views will be recorded.

In the notebook terminal, source the same ROS environments and select one of
these recipes:

```python
# XBox control and the rendered Gazebo camera
collector_run = "xbox_gazebo_widowx_gazebo_cameras"

# XBox control with both USB and rendered Gazebo cameras
# collector_run = "xbox_gazebo_widowx_usb_gazebo_cameras"

# Automatic motion and the rendered Gazebo camera
# collector_run = "automove_gazebo_widowx_gazebo_cameras"
```

To confirm the rendered camera before collecting, run this in another sourced
terminal:

```bash
ros2 topic hz /berrypicker/gazebo/camera_0/image_raw
```

Stop the notebook recorder before pressing `Ctrl+C` in the simulation terminal.

## Real-world WidowX with a Gazebo twin

This configuration controls the physical WidowX named `wx250s` while a second
robot named `wx250s_gazebo` follows its published joint states in Gazebo. The
separate names keep the physical driver and Gazebo controllers in distinct ROS
namespaces.

Use the individual launch commands in this section. The complete
`run_gazebo.py` launcher is intended for fully simulated collection and starts
an additional SDK simulator.

### 1. Start the physical WidowX driver

Connect the robot and start its Interbotix driver in the first terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
ros2 launch interbotix_xsarm_control xsarm_control.launch.py \
  robot_model:=wx250s \
  robot_name:=wx250s \
  use_sim:=false \
  use_rviz:=false
```

Confirm that the physical robot publishes joint states:

```bash
ros2 topic echo /wx250s/joint_states --once
```

### 2. Start the namespaced Gazebo robot

In a second terminal, start Gazebo with the twin named `wx250s_gazebo`:

```bash
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source /usr/share/gazebo/setup.sh
ros2 launch interbotix_xsarm_sim xsarm_gz_classic.launch.py \
  robot_model:=wx250s \
  robot_name:=wx250s_gazebo \
  hardware_type:=gz_classic \
  use_gazebo_gui:=true \
  use_rviz:=false
```

After `/spawn_entity` becomes available, spawn the fixed rendered camera from a
third sourced terminal:

```bash
ros2 run gazebo_ros spawn_entity.py \
  -entity berrypicker_fixed_camera \
  -file src/robot/widowx/simulation/fixed_camera.sdf
```

### 3. Mirror the physical joint states into Gazebo

Keep the camera command's terminal available or open another sourced terminal,
then run:

```bash
python src/robot/widowx/simulation/interbotix_gazebo_bridge.py \
  --sdk-robot-name wx250s \
  --gazebo-robot-name wx250s_gazebo
```

Moving the physical arm should now move the Gazebo twin. The bridge only reads
the physical robot's joint-state topic and writes trajectory commands to the
Gazebo namespace.

### 4. Collect the physical demonstration

The machine configuration must enable `widowx_xbox`, `widowx_robot`, and
`fixed_cameras`. Select the existing real-robot recipe:

```python
collector_run = "xbox_widowx_cameras"
```

This recipe records the configured USB cameras while the external bridge keeps
the Gazebo twin synchronized. To include the rendered camera in the same
demonstration, create
`src/experiment_configs/demonstration_collector/xbox_widowx_gazebo_twin_cameras.yaml`
with these participants:

```yaml
demonstration:
  exp: demonstration
  run: freeform
tick_interval: 0.1
participants:
  - name: xbox
    binding: widowx_xbox
    emits: widowx_target
    target_robot: widowx
  - name: widowx
    binding: widowx_robot
    command: widowx_target
    blocking: false
  - name: usb_cameras
    binding: fixed_cameras
  - name: gazebo_cameras
    binding: gazebo_cameras
```

That combined recipe additionally requires the `gazebo_cameras` binding to be
available. Its topic remains `/berrypicker/gazebo/camera_0/image_raw`; the robot
namespace change does not change the fixed camera topic.

Select the combined recipe in the notebook with:

```python
collector_run = "xbox_widowx_gazebo_twin_cameras"
```

At shutdown, end the notebook collection first so the physical controller can
apply its configured shutdown pose. Then stop the bridge and Gazebo, followed
by the physical Interbotix driver.

## Recorded output

Every recorded timestep contains the robot action, participant telemetry, and
images from every camera view selected by the recipe. The recorder saves
metadata when collection ends. Review the images, metadata, and diagnostic log
before converting the image sequences to video.
