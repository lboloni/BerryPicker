"""
exp_run_config.py

The BerryPicker settings of the exp/run framework, which is implemented by
the ExpRunFlow library (exprunflow.exp_run_config).
"""

import pathlib

from exprunflow.exp_run_config import Config, Experiment

Config.PROJECTNAME = "BerryPicker"
Config.SRC_ROOT = pathlib.Path(__file__).resolve().parent
# flows run their notebooks with the kernel named berrypicker
Config.KERNEL_NAME = "berrypicker"
