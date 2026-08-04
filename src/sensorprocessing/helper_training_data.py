"""Build and cache single-view proprioception training datasets."""

from pathlib import Path

import torch

from demonstration.demonstration import Demonstration
from exp_run_config import Config, Experiment
from sensorprocessing.sp_helper import get_transform_to_sp

Config.PROJECTNAME = "BerryPicker"


def load_images_as_proprioception_training(
    exp: Experiment,
    exp_robot: Experiment,
    *,
    training_fraction: float = 0.67,
    generator: torch.Generator | None = None,
):
    """Load, cache, shuffle, and split single-view proprioception data.

    ``exp["training_data"]`` contains ``(run, demonstration, camera)`` items.
    The processed image and target tensors are cached using the filenames in
    ``exp``. Delete either cache file to rebuild both tensors.
    """
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must be between 0 and 1")

    input_path = Path(exp["data_dir"]) / exp["proprioception_input_file"]
    target_path = Path(exp["data_dir"]) / exp["proprioception_target_file"]

    if input_path.exists() and target_path.exists():
        inputs = torch.load(input_path, weights_only=True)
        targets = torch.load(target_path, weights_only=True)
    else:
        input_list = []
        target_list = []
        transform = get_transform_to_sp(exp)

        for run, demo_name, camera in exp["training_data"]:
            exp_demo = Config().get_experiment("demonstration", run)
            demo = Demonstration(exp_demo, demo_name)
            for index in range(demo.metadata["maxsteps"]):
                sensor_readings, _ = demo.get_image(
                    index, transform=transform, camera=camera
                )
                input_list.append(sensor_readings[0])

                position = demo.get_action(
                    index, "rc-position-target", exp_robot
                )
                normalized = position.to_normalized_vector(exp_robot)
                target_list.append(torch.from_numpy(normalized))

        if not input_list:
            raise ValueError("exp['training_data'] produced no training examples")

        inputs = torch.stack(input_list)
        targets = torch.stack(target_list)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inputs, input_path)
        torch.save(targets, target_path)

    if inputs.size(0) != targets.size(0):
        raise ValueError(
            "proprioception inputs and targets contain different sample counts"
        )

    length = inputs.size(0)
    rows = torch.randperm(length, generator=generator)
    shuffled_inputs = inputs[rows]
    shuffled_targets = targets[rows]
    training_size = int(length * training_fraction)

    return {
        "inputs": inputs,
        "targets": targets,
        "inputs_training": shuffled_inputs[:training_size],
        "targets_training": shuffled_targets[:training_size],
        "inputs_validation": shuffled_inputs[training_size:],
        "targets_validation": shuffled_targets[training_size:],
    }
