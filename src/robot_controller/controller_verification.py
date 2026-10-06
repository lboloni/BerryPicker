"""Teacher-forced verification of a trained robot controller.

See robot_controller/DESIGN-BehaviorCloningFlow.md.
"""

import numpy as np
import torch

from robot_controller.training_data import RobotControllerSequenceDataset


def teacher_forcing(
        controller, entry, sensor_exp, robot_exp, output_size,
        input_label="image_input", output_label="robot_output",
        **dataset_kwargs):
    """Return (predicted, target) arrays of shape [T, output_size] for one
    demonstration, feeding each recorded frame to the controller. The
    frames are preprocessed as in training, and the target at step t is the
    recorded normalized action at t+1. A step without a controller output
    (the warm-up of a sliding-window LSTM) is a row of NaN."""
    dataset = RobotControllerSequenceDataset(
        [entry], sensor_exp, robot_exp, sequence_length=1,
        output_size=output_size, **dataset_kwargs)
    controller.reset_context()
    predicted, target = [], []
    for index in range(len(dataset)):
        images, action = dataset[index]
        _, timestep = dataset.samples[index]
        controller.receive_input(input_label, images[-1].unsqueeze(0), timestep)
        controller.propagate()
        output = controller.read_output(output_label)
        if output is None:
            predicted.append(np.full(output_size, np.nan))
        else:
            predicted.append(
                torch.as_tensor(output).detach().cpu().reshape(-1).numpy())
        target.append(action.numpy())
    return np.stack(predicted), np.stack(target)
