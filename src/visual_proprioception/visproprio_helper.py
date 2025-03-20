"""
visproprio_helper.py

Helper functions for visual proprioception training
"""

import pathlib
import torch
import numpy as np
from settings import Config
from behavior_cloning.demo_to_trainingdata import BCDemonstration
from robot.al5d_position_controller import RobotPosition
from sensorprocessing import sp_conv_vae, sp_propriotuned_cnn, sp_aruco, sp_vit

def load_demonstrations_as_proprioception_training(sp, task, proprioception_input_file, proprioception_target_file):
    """
    FIXME: it is not clear if this is the one where this data is put together - if that is the case, this is actually not the right function to call

    Iterates over all the demonstrations of a task. Loads the images and the corresponding positions. Preprocesses the images with the sensor processor sp passed. Saves these data into proprioception_input_file, proprioception_target_file.

    These data are cached, if the files exists, it does not iterate over the demonstrations, just loads the cached data. Remove those files to recalculate.

    Shuffles the dataset thus created, and splits it into training and validation data. 

    Returns a dictionary with the training and validation data. 
    """
    retval = {}
    if proprioception_input_file.exists():
        retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
    else:
        demos_dir = pathlib.Path(Config()["demos"]["directory"])
        task_dir = pathlib.Path(demos_dir, "demos", task)
        
        inputlist = []
        targetlist = []

        for demo_dir in task_dir.iterdir():
            if not demo_dir.is_dir():
                pass
            bcd = BCDemonstration(demo_dir, sensorprocessor=sp)
            # print(bcd)
            z, a = bcd.read_z_a()
            # normalize the actions
            #print(z.shape)
            #print(a.shape)
            anorm = np.zeros(a.shape, np.float32)
            for i in range(a.shape[0]):
                rp = RobotPosition.from_vector(a[i])
                anorm[i,:] = rp.to_normalized_vector()        
            # FIXME the repeated name for inputs and targets
            #print(z.shape)
            #print(anorm.shape)

            for i in range(z.shape[0]):
                inp = torch.from_numpy(z[i])
                tgt = torch.from_numpy(anorm[i])
                inputlist.append(inp)
                targetlist.append(tgt)

        retval["inputs"] = torch.stack(inputlist)
        retval["targets"] = torch.stack(targetlist)
        torch.save(retval["inputs"], proprioception_input_file)
        torch.save(retval["targets"], proprioception_target_file)

    # Separate the training and validation data. 
    # We will be shuffling the demonstrations 
    length = retval["inputs"].size(0)
    rows = torch.randperm(length) 
    shuffled_inputs = retval["inputs"][rows]
    shuffled_targets = retval["targets"][rows]

    training_size = int( length * 0.67 )
    retval["inputs_training"] = shuffled_inputs[1:training_size]
    retval["targets_training"] = shuffled_targets[1:training_size]

    retval["inputs_validation"] = shuffled_inputs[training_size:]
    retval["targets_validation"] = shuffled_targets[training_size:]

    return retval


def get_visual_proprioception_sp(exp, device):
    """Gets the sensor processing component specified by the 
    visual_proprioception experiment."""
    spexp = Config().get_experiment(exp['sp_experiment'], exp['sp_run'])
    if exp["sensor_processing"] == "ConvVaeSensorProcessing":
        return sp_conv_vae.ConvVaeSensorProcessing(spexp, device)
    if exp['sensor_processing']=="VGG19ProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.VGG19ProprioTunedSensorProcessing(spexp, device)
    if exp['sensor_processing']=="ResNetProprioTunedSensorProcessing":
        return sp_propriotuned_cnn.ResNetProprioTunedSensorProcessing(spexp, device)
    if exp['sensor_processing']=="Aruco":
        return sp_aruco.ArucoSensorProcessing(spexp, device)
    if exp['sensor_processing']=="Vit":
        return sp_vit.VitSensorProcessing(spexp, device)
    raise Exception('Unknown sensor processing {exp["sensor_processing"]}')
