"""
visproprio_helper.py

Helper functions for visual proprioception training
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import numpy as np
from demonstration.demonstration import Demonstration
# from demonstration.encoded_demonstrations import BCDemonstration

from robot.al5d_position_controller import RobotPosition

import sensorprocessing.sp_helper as sp_helper


def external_setup(setupname, rootdir: pathlib.Path):
    """Create an external directory 'setupname' under rootdir, where the generated exp/runs and results will go. This allows separating a set of experiments both for training and robot running.

    Under this directory, there will be two directories:
    * 'exprun' - contains the copied necessary expruns from the source code + the programatically generated expruns.
    * 'result' - contains the training data and the trained models.

    The training data should go into result/demonstration under some directory (eg. touch-apple).
    """
    rootdir = pathlib.Path(rootdir).expanduser()
    setup_path = pathlib.Path(rootdir, setupname)
    exprun_path = pathlib.Path(setup_path, "exprun")
    result_path = pathlib.Path(setup_path, "result")

    print(f"***Path for external experiments:\n{exprun_path}")
    exprun_path.mkdir(exist_ok=True, parents=True)
    print(f"***Path for external data:\n{result_path}")
    result_path.mkdir(exist_ok=True, parents=True)

    Config().set_exprun_path(exprun_path)
    Config().set_results_path(result_path)

    # Copy the necessary experiments into the external directory
    # Config().copy_experiment("robot_al5d")  # ← ADD THIS LINE
    Config().copy_experiment("demonstration")
    Config().copy_experiment("sensorprocessing_conv_vae")
    Config().copy_experiment("sensorprocessing_propriotuned_cnn")
    Config().copy_experiment("sensorprocessing_propriotuned_Vit")
    Config().copy_experiment("sensorprocessing_aruco")
    Config().copy_experiment("visual_proprioception")

    return exprun_path, result_path




def load_demonstrations_as_proprioception_training(sp, exp: Experiment, spexp: Experiment, exp_robot: Experiment, datasetname, proprioception_input_file, proprioception_target_file, device=None):
    """Loads all the images from the specified dataset and creates the input and target tensors. """
    if proprioception_input_file.exists():
        retval = {}
        retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
        print(f"***load_demonstrations_as_proprioception_training*** \n\tSuccessfully loaded from cached files {proprioception_input_file} etc")
        return retval

    inputlist = []
    targetlist = []
    transform = sp_helper.get_transform_to_sp(spexp)

    for val in exp[datasetname]:
        run, demo_name, camera = val
        exp_demo = Config().get_experiment("demonstration", run)
        demo = Demonstration(exp_demo, demo_name)
        for i in range(demo.metadata["maxsteps"]):
            sensor_readings, _ = demo.get_image(i, camera=camera, transform=transform, device=device)
            z = sp.process(sensor_readings)
            # a = demo.get_action(i)
            rp = demo.get_action(i, "rc-position-target", exp_robot)
            #anorm = np.zeros(a.shape, np.float32)
            # rp = RobotPosition.from_vector(exp_robot, a)
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
    print(f"***load_demonstrations_as_proprioception_training*** \n\tSuccessfully recalculated the proprioception training and saved it to {proprioception_input_file} etc")
    return retval

def load_multiview_demonstrations_as_proprioception_training(exp_robot, task, proprioception_input_file, proprioception_target_file, num_views=2):
    """

    FIXME: Sahara: this needs to be changed to match the single-view one above.

    Loads all the images of a task from multiple camera views, and processes it as two tensors
    as input and target data for proprioception training.

    Unlike the single-view version, this function doesn't use a sensor processor during data loading,
    as the multi-view processing is handled separately.

    Caches the processed results into the input and target file pointed in the config.
    Remove those files to recalculate.

    Args:
        task: Task name to load demonstrations from
        proprioception_input_file: Path to save/load processed inputs
        proprioception_target_file: Path to save/load processed targets
        num_views: Number of camera views to process

    Returns:
        Dictionary containing training and validation data splits
    """

    ### FIXME, draft from Lotzi

    # for val in exp[datasetname]:
    #     run, demo_name, cameras = val
    #     exp_demo = Config().get_experiment("demonstration", run)
    #     demo = Demonstration(exp_demo, demo_name)
    #     for i in range(demo.metadata["maxsteps"]):
    #         S = []
    #         for cam in cameras:
    #             sensor_readings, _ = demo.get_image(i, camera=cam, transform=transform, device=device)
    #             z = sp.process(sensor_readings)
    #             S.append(sensor_readings)
    #         # create the concatenated ...
    #         a = demo.get_action(i)
    #         #anorm = np.zeros(a.shape, np.float32)
    #         rp = RobotPosition.from_vector(exp_robot, a)
    #         anorm = rp.to_normalized_vector(exp_robot)
    #         inp = torch.from_numpy(z)
    #         tgt = torch.from_numpy(anorm)
    #         inputlist.append(inp)
    #         targetlist.append(tgt)

    ### END FIXME, draft from Lotzi

    retval = {}
    if proprioception_input_file.exists():
        print(f"Loading cached data from {proprioception_input_file}")
        retval["view_inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
    else:
        demos_dir = pathlib.Path(Config()["demos"]["directory"])
        task_dir = pathlib.Path(demos_dir, "demos", task)

        # Lists to store multi-view images and targets
        view_lists = {}  # Dictionary to organize views by camera
        targetlist = []

        print(f"Loading demonstrations from {task_dir}")
        for demo_dir in task_dir.iterdir():
            if not demo_dir.is_dir():
                continue

            print(f"Processing demonstration: {demo_dir.name}")
            # Create BCDemonstration with multi-camera support
            bcd = BCDemonstration(
                demo_dir,
                sensorprocessor=None,
                cameras=None  # This will detect all available cameras
            )

            # Initialize view lists if not already done
            if not view_lists:
                for camera in bcd.cameras:
                    view_lists[camera] = []

            # Process each timestep
            for i in range(bcd.trim_from, bcd.trim_to):
                # Get all images for this timestep
                all_images = bcd.get_all_images(i)

                # If we don't have all required views, skip this timestep
                if len(all_images) < num_views:
                    print(f"  Skipping timestep {i} - only {len(all_images)}/{num_views} views available")
                    continue

                # Collect images from each camera
                for camera, (sensor_readings, _) in all_images.items():
                    if camera in view_lists:
                        view_lists[camera].append(sensor_readings[0])

                # Get the robot action for this timestep
                a = bcd.get_a(i)
                rp = RobotPosition.from_vector(exp_robot, a)
                anorm = rp.to_normalized_vector()
                targetlist.append(torch.from_numpy(anorm))

        # Ensure we have the same number of frames for each view
        min_frames = min(len(view_list) for view_list in view_lists.values())
        if min_frames < len(targetlist):
            print(f"Truncating dataset to {min_frames} frames (from {len(targetlist)})")
            targetlist = targetlist[:min_frames]
            for camera in view_lists:
                view_lists[camera] = view_lists[camera][:min_frames]

        # Stack tensors for each view
        view_tensors = []
        for camera in sorted(view_lists.keys())[:num_views]:  # Take only the required number of views
            view_tensors.append(torch.stack(view_lists[camera]))

        # Create multi-view input tensor list [num_views, num_samples, C, H, W]
        retval["view_inputs"] = view_tensors
        retval["targets"] = torch.stack(targetlist)

        # Save processed data
        torch.save(retval["view_inputs"], proprioception_input_file)
        torch.save(retval["targets"], proprioception_target_file)
        print(f"Saved {len(targetlist)} training examples with {num_views} views each")

    # Separate the training and validation data
    length = len(retval["targets"])
    rows = torch.randperm(length)

    # Shuffle targets
    shuffled_targets = retval["targets"][rows]

    # Shuffle each view input using the same row indices
    shuffled_view_inputs = []
    for view_tensor in retval["view_inputs"]:
        shuffled_view_inputs.append(view_tensor[rows])

    # Split into training (67%) and validation (33%) sets
    training_size = int(length * 0.67)

    # Training data
    retval["view_inputs_training"] = [view[:training_size] for view in shuffled_view_inputs]
    retval["targets_training"] = shuffled_targets[:training_size]

    # Validation data
    retval["view_inputs_validation"] = [view[training_size:] for view in shuffled_view_inputs]
    retval["targets_validation"] = shuffled_targets[training_size:]

    print(f"Created {training_size} training examples and {length - training_size} validation examples")
    return retval


