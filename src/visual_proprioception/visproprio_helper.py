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
from sensorprocessing import sp_conv_vae, sp_propriotuned_cnn, sp_aruco, sp_vit, sp_vit_multiview

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

def load_multiview_demonstrations_as_proprioception_training(task, proprioception_input_file, proprioception_target_file, num_views=2):
    """
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
                rp = RobotPosition.from_vector(a)
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
    if exp['sensor_processing'] == "Vit_multiview":
        return sp_vit_multiview.MultiViewVitSensorProcessing(spexp, device)
    raise Exception('Unknown sensor processing {exp["sensor_processing"]}')
