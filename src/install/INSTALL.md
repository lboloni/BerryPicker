# How to install BerryPicker on a new Linux environment?

## Things to configure before running the install script

### Install python, venv and system tools

```
sudo apt install python3-venv
sudo apt install python-is-python3
sudo apt install ffmpeg graphviz
```

### Make the user sudo capable:

```
sudo usermod -aG sudo <username>
```

or add username into  /etc/sudoers 

### Access to the robot and the gamepad

The serial port of the robot and the gamepad need these groups (log out and in afterwards): 

```
sudo usermod -aG dialout,input <username>
```

## Install BerryPicker

Clone BerryPicker anywhere, and run the install script from that checkout. Conv-VAE-PyTorch is cloned next to it, the data, the config and the venv go into ~/WORK. 

```
git clone https://github.com/lboloni/BerryPicker ~/Documents/GitHub/BerryPicker
~/Documents/GitHub/BerryPicker/src/install/berry_install.sh
```

The install script also registers the venv as the Jupyter kernel `berrypicker`, which the flows use to run their notebooks. If the venv is recreated, register it again: 

```
~/WORK/BerryPicker/vm/berrypickervenv/bin/python -m ipykernel install --user --name berrypicker --display-name "BerryPicker"
```

## Activate BerryPicker

This is needed before running vscode from the same terminal, or before running the flow notebooks from the command line. 

```
source berry_activate.sh
```

This activates the virtual environment, and changes into the src directory of the checkout. 

## Uninstall BerryPicker

```
berry_uninstall.sh
```

This removes the venv, the config, the data in ~/WORK/BerryPicker and the `berrypicker` kernel. The checkouts are not removed. 
