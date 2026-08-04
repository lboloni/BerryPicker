"""Build and cache image-based proprioception training datasets."""

import json
from pathlib import Path

import torch

from demonstration.demonstration import Demonstration
from exp_run_config import Config, Experiment
from sensorprocessing.sp_helper import get_transform_to_sp

Config.PROJECTNAME = "BerryPicker"

MULTIVIEW_CACHE_VERSION = 1


def _validate_multiview_tensors(view_inputs, targets, num_views, description):
    """Validate the cached or newly built multiview tensor structure."""
    if not isinstance(view_inputs, (list, tuple)):
        raise ValueError(f"{description} inputs must be a list of view tensors")
    if len(view_inputs) != num_views:
        raise ValueError(
            f"{description} inputs contain {len(view_inputs)} views; "
            f"expected {num_views}"
        )
    if not isinstance(targets, torch.Tensor):
        raise ValueError(f"{description} targets must be a tensor")

    sample_count = targets.size(0)
    for view_index, view in enumerate(view_inputs):
        if not isinstance(view, torch.Tensor):
            raise ValueError(
                f"{description} view {view_index} must be a tensor"
            )
        if view.size(0) != sample_count:
            raise ValueError(
                f"{description} view {view_index} and targets contain "
                "different sample counts"
            )


def _build_multiview_tensors(exp, exp_robot, dataset_name, transform):
    """Build ordered raw-image view tensors for one configured dataset."""
    num_views = exp["num_views"]
    view_lists = [[] for _ in range(num_views)]
    target_list = []

    for run, demo_name, cameras in exp[dataset_name]:
        if not isinstance(cameras, (list, tuple)):
            raise ValueError(
                f"exp['{dataset_name}'] cameras for {demo_name} must be "
                "an ordered list"
            )
        if len(cameras) != num_views:
            raise ValueError(
                f"exp['{dataset_name}'] specifies {len(cameras)} cameras "
                f"for {demo_name}; expected {num_views}"
            )
        if len(set(cameras)) != len(cameras):
            raise ValueError(
                f"exp['{dataset_name}'] contains duplicate cameras "
                f"for {demo_name}"
            )

        exp_demo = Config().get_experiment("demonstration", run)
        demo = Demonstration(exp_demo, demo_name)
        available_cameras = demo.metadata.get("cameras", [])
        missing_cameras = [
            camera for camera in cameras if camera not in available_cameras
        ]
        if missing_cameras:
            raise ValueError(
                f"demonstration {demo_name} does not contain configured "
                f"cameras {missing_cameras}"
            )

        for index in range(demo.metadata["maxsteps"]):
            frame_views = []
            for camera in cameras:
                sensor_readings, _ = demo.get_image(
                    index, transform=transform, camera=camera
                )
                if sensor_readings is None:
                    raise ValueError(
                        f"could not load {demo_name} frame {index} "
                        f"from camera {camera}"
                    )
                frame_views.append(sensor_readings[0].cpu())

            position = demo.get_action(
                index, "rc-position-target", exp_robot
            )
            normalized = position.to_normalized_vector(exp_robot)
            target = torch.from_numpy(normalized).cpu()

            # Commit an example only after every view and its target succeeded.
            for view_index, view in enumerate(frame_views):
                view_lists[view_index].append(view)
            target_list.append(target)

    if not target_list:
        raise ValueError(f"exp['{dataset_name}'] produced no examples")

    view_inputs = [torch.stack(view_list) for view_list in view_lists]
    targets = torch.stack(target_list)
    _validate_multiview_tensors(
        view_inputs, targets, num_views, dataset_name
    )
    return view_inputs, targets


def _multiview_cache_manifest(exp, dataset_name):
    """Describe inputs that determine the contents of a multiview cache."""
    manifest = {
        "version": MULTIVIEW_CACHE_VERSION,
        "dataset_name": dataset_name,
        "dataset": exp[dataset_name],
        "image_size": exp["image_size"],
        "num_views": exp["num_views"],
    }
    # Normalize tuples and other JSON-compatible sequences before comparison.
    return json.loads(json.dumps(manifest))


def _load_or_build_multiview_tensors(
    exp,
    exp_robot,
    dataset_name,
    input_path,
    target_path,
    transform,
):
    """Load a complete cache pair, or rebuild and save both members."""
    manifest_path = input_path.with_name(input_path.name + ".manifest.json")
    expected_manifest = _multiview_cache_manifest(exp, dataset_name)
    cached_manifest = None
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                cached_manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            cached_manifest = None

    if (
        input_path.exists()
        and target_path.exists()
        and cached_manifest == expected_manifest
    ):
        view_inputs = torch.load(
            input_path, map_location="cpu", weights_only=True
        )
        targets = torch.load(
            target_path, map_location="cpu", weights_only=True
        )
    else:
        view_inputs, targets = _build_multiview_tensors(
            exp, exp_robot, dataset_name, transform
        )
        input_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(view_inputs, input_path)
        torch.save(targets, target_path)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(expected_manifest, handle, indent=2)
            handle.write("\n")

    _validate_multiview_tensors(
        view_inputs, targets, exp["num_views"], dataset_name
    )
    return list(view_inputs), targets


def load_multiview_images_as_proprioception_training(
    exp: Experiment,
    exp_robot: Experiment,
    *,
    training_fraction: float = 0.67,
    generator: torch.Generator | None = None,
):
    """Load, cache, and partition ordered multiview image data.

    Each item in ``exp["training_data"]`` and ``exp["validation_data"]``
    has the form ``(demonstration_run, demonstration_name, cameras)``.
    ``cameras`` is an ordered list whose length equals ``exp["num_views"]``.

    When ``validation_data`` is configured, it is loaded as the validation
    set and cached using the ``proprioception_test_*`` filenames. Otherwise,
    the training data is synchronously shuffled and split.
    """
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must be between 0 and 1")
    if exp["num_views"] < 1:
        raise ValueError("exp['num_views'] must be at least 1")

    data_dir = Path(exp["data_dir"])
    input_path = data_dir / exp["proprioception_input_file"]
    target_path = data_dir / exp["proprioception_target_file"]
    transform = get_transform_to_sp(exp)

    view_inputs, targets = _load_or_build_multiview_tensors(
        exp,
        exp_robot,
        "training_data",
        input_path,
        target_path,
        transform,
    )

    result = {
        "view_inputs": view_inputs,
        "targets": targets,
    }

    if exp.get("validation_data"):
        validation_input_path = (
            data_dir / exp["proprioception_test_input_file"]
        )
        validation_target_path = (
            data_dir / exp["proprioception_test_target_file"]
        )
        validation_inputs, validation_targets = (
            _load_or_build_multiview_tensors(
                exp,
                exp_robot,
                "validation_data",
                validation_input_path,
                validation_target_path,
                transform,
            )
        )
        result.update(
            {
                "view_inputs_training": view_inputs,
                "targets_training": targets,
                "view_inputs_validation": validation_inputs,
                "targets_validation": validation_targets,
            }
        )
        return result

    length = targets.size(0)
    rows = torch.randperm(length, generator=generator)
    training_size = int(length * training_fraction)
    result.update(
        {
            "view_inputs_training": [
                view[rows[:training_size]] for view in view_inputs
            ],
            "targets_training": targets[rows[:training_size]],
            "view_inputs_validation": [
                view[rows[training_size:]] for view in view_inputs
            ],
            "targets_validation": targets[rows[training_size:]],
        }
    )
    return result


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
