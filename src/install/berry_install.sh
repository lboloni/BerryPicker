#!/bin/bash
set -x # turn on echo
# the BerryPicker checkout containing this script, and the directory it is in
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CHECKOUTS=$(dirname "$REPO")

# create the config
mkdir -p ~/.config/BerryPicker
echo configpath: \"~/WORK/BerryPicker/cfg/settings.yaml\" > ~/.config/BerryPicker/mainsettings.yaml

# check out Conv-VAE next to BerryPicker
cd "$CHECKOUTS"
[ -d Conv-VAE-PyTorch ] || git clone https://github.com/julian-8897/Conv-VAE-PyTorch
# check out the ExpRunFlow exp/run and flow library next to BerryPicker
[ -d ExpRunFlow ] || git clone https://github.com/lboloni/ExpRunFlow
# check out the WidowX AI MuJoCo models next to BerryPicker (src/robot/wxai)
[ -d trossen_arm_mujoco ] || git clone https://github.com/TrossenRobotics/trossen_arm_mujoco

# create the data dirs
mkdir -p ~/WORK/BerryPicker/data
mkdir -p ~/WORK/BerryPicker-Flows
mkdir -p ~/WORK/BerryPicker-Demopacks

# create the config
mkdir -p ~/WORK/BerryPicker/cfg
cd ~/WORK/BerryPicker/cfg
sed "s#~/WORK/BerryPicker/src#$CHECKOUTS#" "$REPO/src/install/settings-sample.yaml" > settings.yaml

# create the vm
mkdir -p ~/WORK/BerryPicker/vm
cd ~/WORK/BerryPicker/vm
python -m venv berrypickervenv
source berrypickervenv/bin/activate
pip install ipykernel
pip install pyyaml papermill ipywidgets numpy pyserial opencv-python
pip install approxeng.input
pip install pillow matplotlib pandas scipy tqdm tensorboardX pytest graphviz
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e "$CHECKOUTS/ExpRunFlow[flow]"
# the WidowX AI driver and simulation; only the models of trossen_arm_mujoco are used,
# so its own dependencies (dm_control, h5py, ...) are not installed
pip install trossen-arm==1.11.0 mujoco
pip install --no-deps -e "$CHECKOUTS/trossen_arm_mujoco"
# flows run their notebooks with the kernel named berrypicker (src/exp_run_config.py)
python -m ipykernel install --user --name berrypicker --display-name "BerryPicker"

# install the script for approxeng
mkdir -p ~/.approxeng.input
cp "$REPO/src/install/microsoft_xbox_360_pad_v1118_p654.yaml" ~/.approxeng.input/
