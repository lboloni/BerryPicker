"""
conv_vae.py

Helper functions for calling the Julian-8897-Conv-VAE-PyTorch ConvVAE implementation. As that implementation is heavily reliant on json file input and command line, we are generating the json here from our own code. 
"""

import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# these imports are from the Conv-VAE-Torch package
import data_loader.data_loaders as module_data
from sensorprocessing.sp_helper import get_transform_to_sp
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
# end of Conv-VAE-Torch imports

from PIL import Image

from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
#import os
import matplotlib.pyplot as plt
import argparse
import logging
from copy import deepcopy



def get_conv_vae_config(vae_config_yaml): 
    """Creates the configuration object of the conv-vae package. It is doing this by emulating the command line, but essentially most of the information is in a json file."""
    # FIXME: pytorch template??? this is probably some bogus stuff
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    value = ["this-script", f"-c{vae_config_yaml}"]
    # we are changing the parameters from here, to avoid changing the github 
    # downloaded package
    sys.argv = value
    config = ConfigParser.from_args(args)
    print(config)
    return config


def train(config):
    """Train the model with the parameters described from the configuration object. Returns the trainer object, which then can be queried for the final metrics.
    
    """
    # this would require a logger_config.json file
    # logger = config.get_logger('train')
    # let us just create a simple logger here
    logger = logging.getLogger(__name__)


    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    # If validation loader included then training will use validation loss
    if 'valid_loader' in config.config:
        valid_data_loader = config.init_obj('valid_loader', module_data)
    else:
        # valid_data made equal to None to replicate previous behavior.
        valid_data_loader = data_loader.split_validation()
        valid_data_loader = None

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()
    # return the trainer, as the metrics might be useful
    return trainer


def create_configured_vae_json(exp):
    """As the Conv-VAE codebase is using a json file to set the configuration, this code takes a template json, modifies the values based on our own configuration in settings.Config and sets writes it out into a json file again. The new json file is called "conv-vae-configured.json and it goes into the root of the model directory
    """

    # FIXME: the template should come from the exp
    current_directory = Path(__file__).resolve().parent
    json_template_path = Path(current_directory, exp["json_template_name"])
    # "encoding_conv_vae/conv-vae-config-default.json"
    # json_template_path = Config().values["conv_vae"]["json_template"]
    print(json_template_path)

    with open(json_template_path, 'r') as file:
        data = json.load(file)

    data["name"] = exp["model_name"]
    data["data_loader"]["args"]["data_dir"] = str(
        Path(exp["data_dir"],exp["training_data_dir"]))
    # Creates a validation dataloader if defined in the config file
    if "validation_data_dir" in exp:
        data["valid_loader"] = deepcopy(data["data_loader"])
        data["valid_loader"]["args"]["data_dir"] = str(
            Path(exp["data_dir"],exp["validation_data_dir"]))
        # Extending early_stop from 10 to 100 as VAE will stop too early on bad runs
        data['trainer']['early_stop'] = 100
        # Making validation_split equal to 0 to use all data present in data loaders
        data["valid_loader"]['args']["validation_split"] = 0.0
        data["data_loader"]['args']["validation_split"] = 0.0

    # update the vae_config based on parameters from the experiment
    data["trainer"]["epochs"] = exp["epochs"] # was 100
    data["trainer"]["save_period"] = exp["save_period"] # 5
    data["arch"]["args"]["latent_dims"] = exp["latent_size"]

    # Config().values["conv_vae"]["training_data_dir"]
    model_dir = Path(exp["data_dir"],exp["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    data["trainer"]["save_dir"] = str(model_dir)
    # Config()["conv_vae"]["model_dir"]

    print(data)

    # the temporary json file: above the models in conv-vae-temp.json
    #json_temporary_path = Path(Config().values["conv_vae"]["model_dir"], "conv-vae-configured.json")
    json_temporary_path = Path(
        exp["data_dir"], exp["model_dir"], "conv-vae-configured.json")

    # Open a file in write mode
    with open(json_temporary_path, 'w') as file:
        # Write the dictionary to the file in JSON format
        json.dump(data, file, indent=4)  # The `indent=4` makes the JSON more readable
    return json_temporary_path

def latest_model(run_path):
    """Returns the filename of the latest checkpoint from the training_run path or none if there is no such model"""
    models = run_path.glob("*.pth")
    highest = -1
    model = None
    for m in models:
        match = re.search(r'\d+', m.name)
        if match:
            number = int(match.group())
            if number > highest:
                model = m.name
                highest = number
    return model

def latest_training_run(model_path):
    """Returns the directory name of the latest training run path
    These have the format: 0901_125042, which seems to be based on the
    day / minute when they were created
    """
    subdirs= [d.name for d in model_path.iterdir() if d.is_dir()]
    latest = sorted(subdirs, reverse=True)[0]
    return latest


def latest_json_and_model():
    """Returns the latest Conv-Vae path and model, taking the information from the values dict of the config
    FIXME: this needs to be converted to exp"""
    model_dir = Path(Config()["conv_vae"]["model_dir"])
    model_path = Path(model_dir, "models", Config()["conv_vae"]["model_name"])
    latest = latest_training_run(model_path)
    # print(latest)
    model_path = Path(model_path, latest)
    model = latest_model(model_path)
    if model is None:
        print(f"latest_json_and_model: there is no model in path:\n {model_path}\n Likely a spuriously created directory that needs to be removed.")
        return None, None
    # The model from which we are starting        
    resume_model = Path(model_path, model)
    jsonfile = Path(model_path, "config.json")
    print(f"resume_model and jsonfile are:\n\tresume_model={resume_model}\n\tjsonfile={jsonfile}")
    return jsonfile, resume_model 


def get_conv_vae_config(jsonfile, resume_model = None, inference_only = True):
    """Returns the configuration object of the Experiment-Conv-Vae"""
    # As the code is highly dependent on the command line, emulating it here
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')


    # value = ["this-script", f"-c{file}", f"-r{model}"]
    if resume_model is None:
        value = ["this-script", f"-c{jsonfile}"]
    else:
        value = ["this-script", f"-c{jsonfile}", f"-r{resume_model}"]

    # we are changing the parameters from here, to avoid changing the github downloaded package
    savedargv = sys.argv
    sys.argv = value
    config = ConfigParser.from_args(args)
    sys.argv = savedargv
    # print(json.dumps(config.config, indent=4))
    #
    # THIS was an attempt to fix some kind of weird bug where an empty 
    # directory was created... it is not needed on 2024.11.17???
    # if it is inference only, remove the superfluously created directories.
    #
    inffix = False
    if inference_only and inffix:
        remove_dir = Path(jsonfile.parent.parent, latest_training_run(jsonfile.parent.parent))
        remove_json = Path(remove_dir, "config.json")
        print(f"Removing unnecessarily created json file: {remove_json.absolute()}")
        remove_json.unlink()
        print(f"Removing unnecessarily created package directory: {remove_dir.absolute()}")
        remove_dir.rmdir()
    return config


