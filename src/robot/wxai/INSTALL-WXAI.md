# How to install the WidowX AI driver and simulation?

The WidowX AI code (`src/robot/wxai`) needs:

* `trossen-arm`, the Python driver of the arm (`trossen_arm`). The API changes often; BerryPicker is written against **1.11.0**, which is pinned in the install script.
* `mujoco`, for the reachability check (inverse kinematics) of every driver and for the simulation.
* the `trossen_arm_mujoco` checkout next to BerryPicker, for its MJCF models (`assets/wxai/wxai_follower.xml`). It is installed with `--no-deps`: BerryPicker uses only the models, not its scripts and their dependencies (`dm_control`, `h5py`, ...).

There is no ROS and no container: unlike the WidowX, the WidowX AI runs on Ubuntu and on macOS.

`src/install/berry_install.sh` installs all of this. In an existing venv:

```
cd ~/Documents/GitHub   # the directory of the BerryPicker checkout
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco
~/WORK/BerryPicker/vm/berrypickervenv/bin/pip install trossen-arm==1.11.0 mujoco
~/WORK/BerryPicker/vm/berrypickervenv/bin/pip install --no-deps -e trossen_arm_mujoco
```

## The network of the arm

The arm controller is on Ethernet, with the factory address `192.168.1.2` (mask `255.255.255.0`). Give the PC's Ethernet interface an unused address of the same subnet, for example `192.168.1.1`. A leader arm needs its own address, `192.168.1.3` in `controllers/wxai_leader_00.yaml`.

The address of the arm is a machine fact: override `ip_address` in

```
<experiment_system_dependent_dir>/robot_wxai/position_controller_wxai_00_sysdep.yaml
<experiment_system_dependent_dir>/controllers/wxai_leader_00_sysdep.yaml
```

## The machine bindings

`data/expruns/machine/current_sysdep.template.yaml` lists the WidowX AI bindings, all unavailable:

* `wxai_robot`: the real arm;
* `fake_wxai`: the real-arm controller on the kinematic fake driver;
* `simulated_wxai`: the strict headless simulator;
* `mujoco_wxai` and `mujoco_cameras`: the MuJoCo simulation and its rendered cameras;
* `wxai_leader`, `fake_wxai_leader`: the leader arm, real and fake.

Set `available: true` in the machine's `current_sysdep.yaml` for those it has.

## Verify

Run `src/demonstration/Verify_Robot_WXAI.ipynb`, first with `run = "position_controller_mujoco_wxai_00"` or the fake run, then with `position_controller_wxai_00` and `ALLOW_ROBOT_MOTION = True` after checking the workspace.

## Headless rendering on Linux

MuJoCo renders the simulated cameras offscreen. On macOS this works without a display. On a Linux machine without a display, set `MUJOCO_GL=egl` (NVIDIA or Mesa EGL) or `MUJOCO_GL=osmesa` before starting Python.
