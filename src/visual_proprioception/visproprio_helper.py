"""
visproprio_helper.py

Helper functions for visual proprioception training
"""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
import torch
import numpy as np
from demonstration.demonstration import Demonstration
# from demonstration.encoded_demonstrations import BCDemonstration

from robot.al5d_position_controller import RobotPosition
from sensorprocessing import sp_conv_vae, sp_propriotuned_cnn, sp_aruco, sp_vit, sp_vit_multiview, sp_vit_concat_images
import sensorprocessing.sp_helper as sp_helper

def load_demonstrations_as_proprioception_training(sp, exp, spexp, datasetname, proprioception_input_file, proprioception_target_file, device=None):
    """Loads all the images from the specified dataset and creates the input and target tensors. """
    if proprioception_input_file.exists():
        retval = {}        
        retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
        retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
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
            a = demo.get_action(i)
            #anorm = np.zeros(a.shape, np.float32)
            rp = RobotPosition.from_vector(a)
            anorm = rp.to_normalized_vector()
            inp = torch.from_numpy(z)
            tgt = torch.from_numpy(anorm)
            inputlist.append(inp)
            targetlist.append(tgt)
    retval = {}
    retval["inputs"] = torch.stack(inputlist)
    retval["targets"] = torch.stack(targetlist)
    torch.save(retval["inputs"], proprioception_input_file)
    torch.save(retval["targets"], proprioception_target_file)
    return retval            



def load_multiview_demonstrations_as_proprioception_training(task, proprioception_input_file, proprioception_target_file, num_views=2):
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
    if exp['sensor_processing'] == "Vit_concat_images":
        return sp_vit_concat_images.ConcatImageVitSensorProcessing(spexp, device)
    raise Exception('Unknown sensor processing {exp["sensor_processing"]}')
