# How to install the WidowX ROS interface and Gazebo simulation?

The WidowX code (`runtime.py`, `simulation/run_gazebo.py`, `simulation/interbotix_gazebo_bridge.py`) needs:

* ROS 2 Humble (`ros2`, `rclpy`, `sensor_msgs`, `trajectory_msgs`)
* Gazebo Classic with `gazebo_plugins` (`run_gazebo.py` checks for the `gazebo` executable and `libgazebo_ros_camera.so`)
* the Interbotix workspace in `~/interbotix_ws` (robot description, `interbotix_xs_modules`)

ROS 2 Humble is only packaged for Ubuntu 22.04. Gazebo Classic is end of life, and it is not packaged for Ubuntu 24.04 / ROS 2 Jazzy. 

* On **Ubuntu 22.04**, install directly on the machine: skip to [Install ROS 2 Humble, Gazebo Classic and Interbotix](#install-ros-2-humble-gazebo-classic-and-interbotix), and run the steps outside a container.
* On **Ubuntu 24.04** (e.g. glassy), run an Ubuntu 22.04 container with distrobox, as described below. 

## Ubuntu 24.04: Ubuntu 22.04 container with distrobox

Distrobox runs a rootless podman container that shares the home directory, the X11 display and the GPU with the host. Inside the container the user has sudo. The checkouts in `~/Documents/GitHub` and the data in `~/WORK` are visible inside the container. 

### Once, by a sudo capable user

```
sudo apt install podman distrobox
```

The user needs entries in `/etc/subuid` and `/etc/subgid` (adduser creates them by default): 

```
grep <username> /etc/subuid /etc/subgid
```

### Create the container

```
distrobox create --name humble --image ubuntu:22.04 --nvidia
distrobox enter humble
```

The `--nvidia` flag mounts the host NVIDIA driver into the container. Check from inside the container that the GPU is visible: 

```
nvidia-smi
```

Inside the container the user has passwordless sudo. Note that `distrobox enter humble -- <command>` evaluates the command with `set -u`, so a command referring to an unset variable fails; put longer commands into a script and run `distrobox enter humble -- bash script.sh`. 

### Only if the GPU is not visible: nvidia-container-toolkit

The `nvidia-container-toolkit` package is not in the Ubuntu repositories (`apt install` fails with "unable to locate package"); the NVIDIA repository needs to be added first. By a sudo capable user, on the host (curl is not necessarily installed, so this uses wget): 

```
wget -qO- https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
wget -qO- https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

If the URLs changed, see "Installing the NVIDIA Container Toolkit" in the NVIDIA documentation. The GPU is optional for the simulation, but Gazebo renders much faster with it. 

### Keep the host shell clean

The home directory is shared, so the lines that the Interbotix install script appends to `~/.bashrc` (under `# Interbotix Configurations`: sourcing ROS and the workspace, and setting `ROS_IP`) would also run on the host, where they fail. Back up `~/.bashrc` before the install, and afterwards wrap the appended block so it only runs inside the container: 

```
# Interbotix Configurations, only inside the humble distrobox container
if [ -n "$CONTAINER_ID" ]; then
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
export ROS_IP=...   # the lines appended by the script
fi
```

## Install ROS 2 Humble, Gazebo Classic and Interbotix

Inside the container (or on an Ubuntu 22.04 machine). The Interbotix arm install script `xsarm_amd64_install.sh` installs ROS 2 Humble and builds `~/interbotix_ws`: 

```
wget https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh
printf 'n\nn\ny\n' | bash xsarm_amd64_install.sh -d humble
```

The script asks three questions: perception packages, MATLAB API, and confirmation. The `printf` answers no, no, yes. Do not use `-n`, as it also installs the perception (RealSense, AprilTag) and MATLAB packages. Expected harmless messages: rosdep cannot resolve `interbotix_xsarm_perception` (because perception is not installed), dbus / `invoke-rc.d` / `udevadm` failures (there are no system services in a container). It builds 29 packages in `~/interbotix_ws`. 

Then add Gazebo Classic and its ROS plugins: 

```
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
```

### numpy and transforms3d

The ROS Humble python packages (`cv_bridge`, the apt `python3-transforms3d` 0.3 used by the Interbotix arm module) only work with numpy 1.x, and transforms3d 0.3 does not even work with numpy >= 1.24 (`np.float`). The Interbotix script installs `modern_robotics` with `pip --user`, which pulls numpy 2 into `~/.local/lib/python3.10`, which breaks `from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS` (`np.maximum_sctype was removed`, `_ARRAY_API not found`). Fix, in the container: 

```
/usr/bin/python3 -m pip install --user --upgrade transforms3d "numpy<2"
```

`~/.local/lib/python3.10` is only used by python 3.10, so this does not affect the host python. 

## Python venv for the ROS side

`rclpy` is built for the system python of Ubuntu 22.04 (python 3.10), so the regular BerryPicker venv (created by `berry_install.sh` with the host python) cannot be used. Create a second venv that sees the ROS and Interbotix packages: 

```
python3 -m venv --system-site-packages ~/WORK/BerryPicker/vm/berrypickervenv-ros
source ~/WORK/BerryPicker/vm/berrypickervenv-ros/bin/activate
```

and install into it the same pip packages as `src/install/berry_install.sh` (torch cu132 has python 3.10 wheels), then pin the versions that work with ROS Humble (opencv-python 4.12 and later require numpy 2): 

```
pip install --upgrade transforms3d "numpy<2" "opencv-python<4.12"
```

Check: 

```
python -c "import rclpy, cv_bridge; from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS; print('ok')"
```

## Run the simulation

Inside the container: 

```
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
source ~/WORK/BerryPicker/vm/berrypickervenv-ros/bin/activate
cd ~/Documents/GitHub/BerryPicker
python src/robot/widowx/simulation/run_gazebo.py
```

Check: 

```
pytest src/test/robot/widowx
ros2 topic hz /berrypicker/gazebo/camera_0/image_raw
```

The launcher prints `Gazebo camera ready on /berrypicker/gazebo/camera_0/image_raw; Interbotix controller ready as wx250s_sdk`; the camera publishes at about 8-10 Hz. Without a display, add `--no-gazebo-gui`. If a previous run did not shut down, the launcher refuses to start (`A Gazebo server is already running`); stop the leftover `gzserver` and `ros2 launch` processes first. 

## Real robot

The same container drives the real WidowX (`xs_sdk` instead of `xs_sdk_sim`). The USB side has to be set up on the host, by a sudo capable user: 

* udev rules are applied by the host, not the container, so install the Interbotix rule on the host. It creates `/dev/ttyDXL` and sets the U2D2 latency timer to 1 ms: 
  ```
  sudo cp ~<username>/interbotix_ws/src/interbotix_ros_core/interbotix_ros_xseries/interbotix_xs_sdk/99-interbotix-udev.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
* the user needs the `dialout` group for the serial port (and `input` for the gamepad), then log out and in: 
  ```
  sudo usermod -aG dialout,input <username>
  ```

Distrobox shares `/dev` with the host. Check with the robot plugged in, inside the container: `ls -l /dev/ttyDXL; id` (dialout should be listed). The simulation and the real robot use the same robot names, so run only one of them at a time. 

For VS Code, either run it from inside the container (`distrobox enter humble -- code`), or attach to the running container with the Dev Containers extension, so that the interpreter and the ROS paths resolve. 

## Without sudo: RoboStack (untested)

[RoboStack](https://github.com/robostack/ros-humble) packages ROS 2 Humble with conda, installed entirely in the user's home. It is not known whether its Humble channel provides Gazebo Classic and `gazebo_ros_pkgs` on Linux, and the Interbotix workspace would need to be built from source against the conda packages. Use only if the distrobox route is not possible. 

## Longer term

ROS 2 Humble is end of life in May 2027, and Gazebo Classic is already end of life. The lasting solution is to port the simulation to ROS 2 Jazzy and the new Gazebo, which run natively on Ubuntu 24.04. This requires changing `run_gazebo.py` and the bridge, and depends on Interbotix support for Jazzy. 
