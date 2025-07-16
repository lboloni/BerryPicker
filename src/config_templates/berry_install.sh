#!/bin/bash
set -x # turn on echo
# create the config
mkdir -p ~/.config/BerryPicker
echo configpath: \"~/WORK/BerryPicker/cfg/settings.yaml\" > ~/.config/BerryPicker/mainsettings.yaml

# check out the src
mkdir -p ~/WORK/BerryPicker/src 
cd ~/WORK/BerryPicker/src/
git clone https://github.com/lboloni/BerryPicker
git clone https://github.com/julian-8897/Conv-VAE-PyTorch
# check out the 

# create the data dirs
mkdir -p ~/WORK/BerryPicker/data


# create the config
mkdir -p ~/WORK/BerryPicker/cfg
cd ~/WORK/BerryPicker/cfg
cp ~/WORK/BerryPicker/src/BerryPicker/src/config_templates/settings-sample.yaml settings.yaml

# create the vm
mkdir -p ~/WORK/BerryPicker/vm
cd ~/WORK/BerryPicker/vm
python -m venv berrypickervenv
source berrypickervenv/bin/activate
pip install pyyaml papermill numpy pyserial opencv-python
pip install approxeng.input

# install the script for approxeng
cp ~/WORK/BerryPicker/src/BerryPicker/src/config_templates/microsoft_xbox_360_pad_v1118_p654.yaml ~/.approxeng.input/