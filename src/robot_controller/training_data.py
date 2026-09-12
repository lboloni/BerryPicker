"""Lazy demonstration sequences for robot-controller behavior cloning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from demonstration.demonstration import Demonstration
from exp_run_config import Config
from sensorprocessing.sp_helper import get_transform_to_sp


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _entry_key(entry):
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        raise ValueError(
            "Training data entries must be [demonstration_run, "
            "demonstration_name, camera]"
        )
    return tuple(entry[:2])


class RobotControllerSequenceDataset(Dataset):
    """Map a sequence ending at ``t`` to the normalized action at ``t+1``."""

    def __init__(
        self, entries, sensor_exp, robot_exp, sequence_length, output_size,
        *, action_type="rc-position-target", frame_stride=1,
        demonstration_factory=Demonstration, experiment_loader=None,
    ):
        if not isinstance(entries, list) or not entries:
            raise ValueError("Dataset entries must be a nonempty list")
        self.sequence_length = _positive_int(sequence_length, "sequence_length")
        self.output_size = _positive_int(output_size, "output_size")
        self.frame_stride = _positive_int(frame_stride, "frame_stride")
        self.robot_exp = robot_exp
        self.action_type = action_type
        self.transform = get_transform_to_sp(sensor_exp)
        self.image_size = tuple(sensor_exp["image_size"])
        self.sources = []
        self.samples = []
        loader = experiment_loader or Config().get_experiment

        for entry in entries:
            _entry_key(entry)
            run, demo_name, camera = entry
            demo = demonstration_factory(
                loader("demonstration", run), demo_name
            )
            cameras = demo.metadata.get("cameras", [])
            if cameras and camera not in cameras:
                raise ValueError(
                    f"Demonstration {demo_name!r} has no camera {camera!r}"
                )
            maxsteps = demo.metadata.get("maxsteps", 0)
            if type(maxsteps) is not int or maxsteps <= self.sequence_length:
                raise ValueError(
                    f"Demonstration {demo_name!r} needs more than "
                    f"{self.sequence_length} timesteps"
                )
            if len(demo.actions) < maxsteps:
                raise ValueError(
                    f"Demonstration {demo_name!r} has {len(demo.actions)} "
                    f"actions for {maxsteps} timesteps"
                )
            source = len(self.sources)
            self.sources.append((demo, camera))
            self.samples.extend(
                (source, timestep)
                for timestep in range(
                    self.sequence_length - 1, maxsteps - 1, self.frame_stride
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, timestep = self.samples[index]
        demo, camera = self.sources[source]
        frames = []
        first = timestep - self.sequence_length + 1
        for frame in range(first, timestep + 1):
            tensor, _ = demo.get_image(
                frame, camera=camera, transform=self.transform
            )
            if tensor is None:
                raise ValueError(
                    f"Could not read {demo.demo!r} camera {camera!r} frame {frame}"
                )
            if tensor.ndim == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            expected = (3, *self.image_size)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Preprocessed frame has shape "
                    f"{getattr(tensor, 'shape', None)}; expected {expected}"
                )
            frames.append(tensor)
        position = demo.get_action(
            timestep + 1, type=self.action_type, exp=self.robot_exp
        )
        target = torch.as_tensor(
            np.asarray(position.to_normalized_vector(self.robot_exp)),
            dtype=torch.float32,
        )
        if tuple(target.shape) != (self.output_size,):
            raise ValueError(
                f"Normalized target has shape {tuple(target.shape)}; expected "
                f"({self.output_size},)"
            )
        images = torch.stack(frames)
        if not torch.isfinite(images).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Training sample contains non-finite values")
        return images, target


def make_controller_dataloaders(
    exp, sensor_exp, robot_exp, sequence_length, output_size, **dataset_kwargs
):
    training_entries = exp["training_data"]
    validation_entries = exp["validation_data"]
    overlap = {_entry_key(item) for item in training_entries} & {
        _entry_key(item) for item in validation_entries
    }
    if overlap:
        raise ValueError(
            "Training and validation must use distinct demonstrations; "
            f"overlap: {sorted(overlap)}"
        )
    common_dataset = {
        "sensor_exp": sensor_exp,
        "robot_exp": robot_exp,
        "sequence_length": sequence_length,
        "output_size": output_size,
        "action_type": exp.get("action_type", "rc-position-target"),
        "frame_stride": exp.get("frame_stride", 1),
        **dataset_kwargs,
    }
    training = RobotControllerSequenceDataset(
        training_entries, **common_dataset
    )
    validation = RobotControllerSequenceDataset(
        validation_entries, **common_dataset
    )
    batch_size = _positive_int(exp["batch_size"], "batch_size")
    workers = exp.get("num_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    pin_memory = exp.get("pin_memory", False)
    if type(pin_memory) is not bool:
        raise ValueError("pin_memory must be boolean")
    seed = exp.get("random_seed", 0)
    if type(seed) is not int:
        raise ValueError("random_seed must be an integer")
    arguments = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(training, shuffle=True, generator=generator, **arguments),
        DataLoader(validation, shuffle=False, **arguments),
    )
