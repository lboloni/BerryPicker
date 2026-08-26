"""
visproprio_helper.py

Helper functions for visual proprioception training.

This module provides functions for:
- Setting up external experiment directories
- Loading demonstrations as proprioception training data (single-view and multi-view)
- Creating sensor processing objects for visual proprioception
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import numpy as np
from demonstration.demonstration import Demonstration
from robot.al5d import RobotPosition
import sensorprocessing.sp_helper as sp_helper
import sensorprocessing.sp_factory as sp_factory
# The multi-view dataset helpers live with the training-data code; they are
# re-exported here so existing imports keep working.
from sensorprocessing.helper_training_data import (  # noqa: F401
    MultiViewDataset,
    collate_multiview,
    make_multiview_loaders,
)


def external_setup(setupname, rootdir: pathlib.Path):
    """Create an external directory 'setupname' under rootdir, where the generated
    exp/runs and results will go. This allows separating a set of experiments both
    for training and robot running.

    Under this directory, there will be two directories:
    * 'expruns' - contains the copied necessary expruns from the source code +
                 the programatically generated expruns.
    * 'results' - contains the training data and the trained models.

    The training data should go into results/demonstration under some directory
    (eg. touch-apple).

    Args:
        setupname: Name for this experimental setup
        rootdir: Root directory path where setup will be created

    Returns:
        tuple: (expruns_path, results_path)
    """
    rootdir = pathlib.Path(rootdir).expanduser()
    setup_path = pathlib.Path(rootdir, setupname)
    expruns_path = pathlib.Path(setup_path, "expruns")
    results_path = pathlib.Path(setup_path, "results")

    print(f"***Path for external experiments:\n{expruns_path}")
    expruns_path.mkdir(exist_ok=True, parents=True)
    print(f"***Path for external data:\n{results_path}")
    results_path.mkdir(exist_ok=True, parents=True)

    Config().set_exprun_path(expruns_path)
    Config().set_results_path(results_path)

    # Copy the necessary experiments into the external directory
    Config().copy_experiment("robot_al5d")
    Config().copy_experiment("demonstration")

    # Copy ALL sensor processing experiments (even if not all used in this flow)
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")
    Config().copy_experiment("sensorprocessing_aruco")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn_multiview")
    Config().copy_experiment("sensorprocessing_conv_vae_concat_multiview")
    Config().copy_experiment("sensorprocessing_conv_vae_multiview")
    Config().copy_experiment("visual_proprioception")
    Config().copy_experiment("visual_proprioception_collections")

    return expruns_path, results_path


def get_visual_proprioception_sp(exp):
    """Get the sensor processing component for a visual proprioception experiment.

    Args:
        exp: Visual proprioception experiment configuration

    Returns:
        Sensor processing object
    """
    spexp = Config().get_experiment(exp["sp_experiment"], exp["sp_run"])
    return sp_factory.create_sp(spexp)


def load_demonstrations_as_proprioception_training(
    sp,
    exp: Experiment,
    spexp: Experiment,
    exp_robot: Experiment,
    datasetname,
    proprioception_input_file,
    proprioception_target_file,
):
    """Loads all the images from the specified dataset and creates the input
    and target tensors for single-view proprioception training.

    This function processes images through the sensor processor to create
    encoded latent representations, then saves them for faster subsequent loading.

    Args:
        sp: Sensor processing object for encoding images
        exp: Visual proprioception experiment config
        spexp: Sensor processing experiment config
        exp_robot: Robot experiment for normalization
        datasetname: "training_data" or "validation_data"
        proprioception_input_file: Path to save/load processed inputs
        proprioception_target_file: Path to save/load processed targets

    Returns:
        Dictionary with 'inputs' and 'targets' tensors
    """
    if proprioception_input_file.exists():
        retval = {}
        retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
        print(f"***load_demonstrations_as_proprioception_training*** \n\t"
              f"Successfully loaded from cached files {proprioception_input_file} etc")
        return retval

    inputlist = []
    targetlist = []
    transform = sp_helper.get_transform_to_sp(spexp)

    for val in exp[datasetname]:
        run, demo_name, camera = val
        exp_demo = Config().get_experiment("demonstration", run)
        demo = Demonstration(exp_demo, demo_name)

        for i in range(demo.metadata["maxsteps"]):
            sensor_readings, _ = demo.get_image(
                i, camera=camera, transform=transform)
            z = sp.process(sensor_readings)
            rp = demo.get_action(i, "rc-position-target", exp_robot)
            anorm = rp.to_normalized_vector(exp_robot)
            inp = torch.from_numpy(z)
            tgt = torch.from_numpy(anorm)
            inputlist.append(inp)
            targetlist.append(tgt)

    retval = {}
    retval["inputs"] = torch.stack(inputlist)
    retval["targets"] = torch.stack(targetlist)
    torch.save(retval["inputs"], proprioception_input_file)
    torch.save(retval["targets"], proprioception_target_file)
    print(f"***load_demonstrations_as_proprioception_training*** \n\t"
          f"Successfully recalculated the proprioception training and saved it to "
          f"{proprioception_input_file} etc")
    return retval


def load_multiview_demonstrations_as_proprioception_training(
    sp,
    exp: Experiment,
    spexp: Experiment,
    exp_robot: Experiment,
    datasetname,
    proprioception_input_file,
    proprioception_target_file,
):
    """Loads all the images from the specified dataset from multiple cameras and creates
    the input and target tensors for visual proprioception training.

    This function is for VP TRAINING - it uses a pre-trained multiview sensor processor
    to encode images from multiple views into a single latent representation.

    Args:
        sp: Pre-trained multiview sensor processor
        exp: VP experiment config
        spexp: SP experiment config
        exp_robot: Robot experiment for normalization
        datasetname: "training_data" or "validation_data"
        proprioception_input_file: Path to save/load processed inputs
        proprioception_target_file: Path to save/load processed targets

    Returns:
        Dictionary with inputs and targets (encoded latents, not raw images)
    """
    if proprioception_input_file.exists():
        retval = {}
        retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
        print(f"***load_multiview_demonstrations_as_proprioception_training*** \n\t"
              f"Successfully loaded from cached files {proprioception_input_file}")
        return retval

    inputlist = []
    targetlist = []
    transform = sp_helper.get_transform_to_sp(spexp)
    num_views = spexp.get("num_views", 2)

    print(f"Loading multiview VP training data with {num_views} views...")

    # Loop through demonstrations using demopack system
    for val in exp[datasetname]:
        run, demo_name, cameras = val  # cameras can be list ["dev2", "dev3"] or string "dev2,dev3"

        # Handle cameras as either list or comma-separated string
        if isinstance(cameras, str):
            cameras = [c.strip() for c in cameras.split(",")]

        exp_demo = Config().get_experiment("demonstration", run)
        demo = Demonstration(exp_demo, demo_name)

        for i in range(demo.metadata["maxsteps"]):
            process_demonstration = getattr(sp, "process_demonstration", None)
            if callable(process_demonstration):
                try:
                    z = process_demonstration(
                        demo, i, cameras[:num_views], transform=transform
                    )
                except Exception as e:
                    print(
                        f"Skipping demo {demo_name} frame {i} - "
                        f"could not load camera views: {e}"
                    )
                    continue
            else:
                # Compatibility path for multiview processors that have not
                # yet adopted MultiViewSensorProcessing.
                view_images = []
                failed_to_load_view = False
                for camera in cameras[:num_views]:
                    try:
                        sensor_readings, _ = demo.get_image(
                            i, camera=camera, transform=transform
                        )
                        view_images.append(sensor_readings)
                    except Exception as e:
                        print(
                            f"Skipping demo {demo_name} frame {i} - "
                            f"missing camera {camera}: {e}"
                        )
                        failed_to_load_view = True
                        break
                if failed_to_load_view:
                    continue
                else:
                    z = sp.process(view_images)

            # Get robot position
            rp = demo.get_action(i, "rc-position-target", exp_robot)
            anorm = rp.to_normalized_vector(exp_robot)

            # Store encoded latent and target
            inp = torch.from_numpy(z)
            tgt = torch.from_numpy(anorm)
            inputlist.append(inp)
            targetlist.append(tgt)

    retval = {}
    retval["inputs"] = torch.stack(inputlist)
    retval["targets"] = torch.stack(targetlist)
    torch.save(retval["inputs"], proprioception_input_file)
    torch.save(retval["targets"], proprioception_target_file)
    print(f"***load_multiview_demonstrations_as_proprioception_training*** \n\t"
          f"Successfully recalculated and saved to {proprioception_input_file}")
    return retval


def load_multiview_raw_images_as_training(
    exp: Experiment,
    spexp: Experiment,
    exp_robot: Experiment,
    datasetname,
    view_inputs_file,
    targets_file,
):
    """Load raw (not encoded) multi-view images for sensor-processing training.

    Superseded by
    :func:`sensorprocessing.helper_training_data.load_multiview_images_as_proprioception_training`,
    which the multi-view training notebooks now use. This wrapper is kept for
    older notebooks; it delegates to the same ordered loader so the view order
    always follows the camera list of each training-data entry (the previous
    implementation sorted cameras alphabetically, which could silently swap
    views between training and inference).
    """
    from sensorprocessing.helper_training_data import (
        load_multiview_images_as_proprioception_training,
    )

    values = dict(getattr(spexp, "values", spexp))
    values["training_data"] = exp[datasetname]
    values.pop("validation_data", None)
    values["proprioception_input_file"] = pathlib.Path(view_inputs_file).name
    values["proprioception_target_file"] = pathlib.Path(targets_file).name
    values["data_dir"] = str(pathlib.Path(view_inputs_file).parent)
    result = load_multiview_images_as_proprioception_training(values, exp_robot)
    return {"view_inputs": result["view_inputs"], "targets": result["targets"]}


def split_training_validation(data, train_ratio=0.67, shuffle=True):
    """Split data dictionary into training and validation sets.

    Args:
        data: Dictionary with 'inputs'/'view_inputs' and 'targets'
        train_ratio: Fraction of data to use for training
        shuffle: Whether to shuffle before splitting

    Returns:
        Dictionary with training and validation splits
    """
    is_multiview = "view_inputs" in data

    if is_multiview:
        length = len(data["targets"])
    else:
        length = data["inputs"].size(0)

    if shuffle:
        rows = torch.randperm(length)
    else:
        rows = torch.arange(length)

    training_size = int(length * train_ratio)

    retval = {}

    if is_multiview:
        # Shuffle targets
        retval["targets_training"] = data["targets"][rows[:training_size]]
        retval["targets_validation"] = data["targets"][rows[training_size:]]

        # Shuffle each view using same indices
        retval["view_inputs_training"] = [
            view[rows[:training_size]] for view in data["view_inputs"]
        ]
        retval["view_inputs_validation"] = [
            view[rows[training_size:]] for view in data["view_inputs"]
        ]
    else:
        retval["inputs_training"] = data["inputs"][rows[:training_size]]
        retval["inputs_validation"] = data["inputs"][rows[training_size:]]
        retval["targets_training"] = data["targets"][rows[:training_size]]
        retval["targets_validation"] = data["targets"][rows[training_size:]]

    print(f"Split data: {training_size} training, {length - training_size} validation")
    return retval
