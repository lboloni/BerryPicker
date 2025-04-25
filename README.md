# BerryPicker
This software package contains code for vision driven robot manipulation using an inexpensive robotic arm.

The type of skills we aim to develop include (but are not limited to) agricultural manipulations such as picking fruits and berries, checking the ripeness of fruits, or detecting plant diseases. 

## Libraries needed
* approxeng.input == 2.5
* torch, torchvision, pandas, numpy
* matplotlib
* tqdm
* pyyaml
* tensorboardX
* pyserial
* opencv-python, opencv-contrib-python

Notes:
* The software for the gamepad input, approxeng.input needs to be at version 2.5, and it requires python not higher than 3.10 (as it needs something called evdev, which seem to have problems with higher python)

# Configuration files

Developing and validating robotics requires extensive experimentation with various hardware and software components. The BerryPicker package uses a relatively intricate yaml based configuration system that allows moving the package from accross different systems, and run experiments. 

## Main configuration

The main configuration object is the singleton Config class in settings. From anywhere in the system, the various top level configuration settings can be used as follows:

```
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

Config()["robot"]["usb_port"]
```

The configuration is loaded as follows. Config first reads the file HOMEDIR/.config/BerryPicker/mainsettings.yaml. This file contains one field, the location of the main config file 

```
# Change the path here to run different configurations on different machines. 
configpath: "/path/to/the/setting.yaml"
```

FIXME: describe the content of the main config file

## Experiment configuration

FIXME: what is a group, what is a run
FIXME: experiment data directory
FIXME: group config, run config, system dependent run config
FIXME: how to use an experiment configuration from code

### Obsolete from here

The recommended way to organize this code is as follows:

```
top directory
    \data
        \demos
            <<< this is where the demonstrations go
    \github
        \VisionBasedDataManipulator
            <<< this git checkout
    \venv
        .venv
            <<< this is where the Python 3.10 environment goes
```
### End obsolete

