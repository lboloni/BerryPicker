"""Lazy synchronized demonstration data for multiview sensor processing."""

from __future__ import annotations

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


def configured_cameras(exp):
    """Return and validate the fixed ordered camera list in an exp/run."""
    num_views = _positive_int(exp["num_views"], "num_views")
    cameras = exp["cameras"]
    if not isinstance(cameras, (list, tuple)):
        raise ValueError("exp['cameras'] must be an ordered list")
    if len(cameras) != num_views:
        raise ValueError(
            f"exp['cameras'] contains {len(cameras)} cameras; expected {num_views}"
        )
    if not all(isinstance(camera, str) and camera for camera in cameras):
        raise ValueError("exp['cameras'] entries must be nonempty strings")
    if len(set(cameras)) != len(cameras):
        raise ValueError("exp['cameras'] must not contain duplicate cameras")
    return list(cameras)


def _dataset_entries(exp, dataset_name):
    expected_cameras = configured_cameras(exp)
    entries = exp[dataset_name]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"exp['{dataset_name}'] must be a nonempty list")

    validated = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                f"exp['{dataset_name}'] entries must be "
                "[demonstration_run, demonstration_name, cameras]"
            )
        run, demonstration_name, cameras = entry
        if not isinstance(cameras, (list, tuple)):
            raise ValueError(
                f"Cameras for demonstration {demonstration_name} must be an "
                "ordered list"
            )
        if list(cameras) != expected_cameras:
            raise ValueError(
                f"Camera order for demonstration {demonstration_name} is "
                f"{list(cameras)}; expected {expected_cameras}"
            )
        validated.append((run, demonstration_name, tuple(cameras)))
    return validated


def validate_multiview_partitions(exp):
    """Reject demonstration leakage between training and validation."""
    training = {
        (run, demonstration_name)
        for run, demonstration_name, _ in _dataset_entries(exp, "training_data")
    }
    validation = {
        (run, demonstration_name)
        for run, demonstration_name, _ in _dataset_entries(exp, "validation_data")
    }
    overlap = training & validation
    if overlap:
        raise ValueError(
            "training_data and validation_data contain the same demonstrations: "
            f"{sorted(overlap)}"
        )


class DemonstrationMultiViewDataset(Dataset):
    """Lazy synchronized camera views, optionally paired with robot position."""

    def __init__(
        self,
        exp,
        dataset_name,
        *,
        robot_exp=None,
        demonstration_factory=Demonstration,
        experiment_loader=None,
    ):
        self.exp = exp
        self.dataset_name = dataset_name
        self.cameras = configured_cameras(exp)
        image_size = exp["image_size"]
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("image_size must contain [height, width]")
        self.image_size = tuple(
            _positive_int(value, f"image_size[{index}]")
            for index, value in enumerate(image_size)
        )
        self.frame_stride = _positive_int(
            exp.get("frame_stride", 1), "frame_stride"
        )
        self.transform = get_transform_to_sp(exp)
        self.robot_exp = robot_exp
        self.sources = []
        self.samples = []
        load_experiment = experiment_loader or Config().get_experiment

        for run, demonstration_name, cameras in _dataset_entries(exp, dataset_name):
            demonstration_exp = load_experiment("demonstration", run)
            demonstration = demonstration_factory(
                demonstration_exp, demonstration_name
            )
            available_cameras = demonstration.metadata.get("cameras")
            if not isinstance(available_cameras, (list, tuple)):
                raise ValueError(
                    f"Demonstration {demonstration_name} has no camera metadata"
                )
            missing = [
                camera for camera in cameras if camera not in available_cameras
            ]
            if missing:
                raise ValueError(
                    f"Demonstration {demonstration_name} is missing cameras {missing}"
                )
            maxsteps = demonstration.metadata.get("maxsteps", 0)
            if type(maxsteps) is not int or maxsteps <= 0:
                raise ValueError(
                    f"Demonstration {demonstration_name} contains no frames"
                )
            source_index = len(self.sources)
            self.sources.append(demonstration)
            self.samples.extend(
                (source_index, timestep)
                for timestep in range(0, maxsteps, self.frame_stride)
            )

        if not self.samples:
            raise ValueError(f"exp['{dataset_name}'] produced no examples")

    def __len__(self):
        return len(self.samples)

    def _load_views(self, demonstration, timestep):
        views = []
        expected = (3, *self.image_size)
        for camera in self.cameras:
            sensor_readings, _ = demonstration.get_image(
                timestep, camera=camera, transform=self.transform
            )
            if sensor_readings is None:
                raise ValueError(
                    f"Could not load timestep {timestep} from camera {camera}"
                )
            if not isinstance(sensor_readings, torch.Tensor):
                raise TypeError("Demonstration.get_image() must return a tensor")
            if sensor_readings.ndim == 4 and sensor_readings.size(0) == 1:
                view = sensor_readings[0]
            elif sensor_readings.ndim == 3:
                view = sensor_readings
            else:
                raise ValueError(
                    f"Camera {camera} returned shape "
                    f"{tuple(sensor_readings.shape)}; expected [1, 3, H, W]"
                )
            if tuple(view.shape) != expected:
                raise ValueError(
                    f"Camera {camera} produced shape {tuple(view.shape)}; "
                    f"expected {expected}"
                )
            if not view.is_floating_point():
                raise TypeError("Preprocessed camera views must be floating point")
            if not torch.isfinite(view).all():
                raise FloatingPointError(
                    f"Camera {camera} produced non-finite image values"
                )
            views.append(view)
        return views

    def _load_target(self, demonstration, timestep):
        position = demonstration.get_action(
            timestep, "rc-position-target", self.robot_exp
        )
        normalized = np.asarray(
            position.to_normalized_vector(self.robot_exp), dtype=np.float32
        )
        if normalized.ndim != 1:
            raise ValueError("Normalized robot position must be a vector")
        if normalized.size != self.exp["output_size"]:
            raise ValueError(
                f"Normalized robot position has size {normalized.size}; "
                f"expected {self.exp['output_size']}"
            )
        if not np.all(np.isfinite(normalized)):
            raise FloatingPointError("Normalized robot position is not finite")
        return torch.from_numpy(normalized)

    def __getitem__(self, index):
        source_index, timestep = self.samples[index]
        demonstration = self.sources[source_index]
        views = self._load_views(demonstration, timestep)
        if self.robot_exp is None:
            return views
        return views, self._load_target(demonstration, timestep)


def collate_multiview_images(batch):
    """Collate ordered view lists into one batch tensor per camera."""
    if not batch:
        raise ValueError("Cannot collate an empty multiview batch")
    num_views = len(batch[0])
    if num_views < 1:
        raise ValueError("A multiview batch must contain at least one view")
    for sample in batch:
        if len(sample) != num_views:
            raise ValueError("Multiview batch samples have different view counts")
    return [
        torch.stack([sample[view_index] for sample in batch])
        for view_index in range(num_views)
    ]


def collate_multiview_proprioception(batch):
    """Collate synchronized views paired with robot-position targets."""
    if not batch:
        raise ValueError("Cannot collate an empty multiview batch")
    views, targets = zip(*batch)
    return collate_multiview_images(views), torch.stack(targets)


def _loader_arguments(exp):
    batch_size = _positive_int(exp["batch_size"], "batch_size")
    num_workers = exp.get("num_workers", 0)
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    pin_memory = exp.get("pin_memory", False)
    if type(pin_memory) is not bool:
        raise ValueError("pin_memory must be boolean")
    seed = exp.get("random_seed", 0)
    if type(seed) is not int:
        raise ValueError("random_seed must be an integer")
    return batch_size, num_workers, pin_memory, seed


def make_multiview_dataloaders(exp, *, robot_exp=None):
    """Create lazy synchronized training and validation dataloaders."""
    validate_multiview_partitions(exp)
    training_dataset = DemonstrationMultiViewDataset(
        exp, "training_data", robot_exp=robot_exp
    )
    validation_dataset = DemonstrationMultiViewDataset(
        exp, "validation_data", robot_exp=robot_exp
    )
    batch_size, num_workers, pin_memory, seed = _loader_arguments(exp)
    if robot_exp is not None:
        if batch_size < 2:
            raise ValueError("Fusion training batch_size must be at least 2")
        if len(training_dataset) < batch_size:
            raise ValueError(
                "Fusion training dataset must contain at least batch_size examples"
            )
        collate = collate_multiview_proprioception
        drop_last = True
    else:
        collate = collate_multiview_images
        drop_last = False
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
    }
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            training_dataset,
            shuffle=True,
            generator=generator,
            drop_last=drop_last,
            **common,
        ),
        DataLoader(
            validation_dataset,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    )
