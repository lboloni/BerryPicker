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
from robot.al5d_position_controller import RobotPosition
import sensorprocessing.sp_helper as sp_helper
import sensorprocessing.sp_factory as sp_factory


def external_setup(setupname, rootdir: pathlib.Path):
    """Create an external directory 'setupname' under rootdir, where the generated
    exp/runs and results will go. This allows separating a set of experiments both
    for training and robot running.

    Under this directory, there will be two directories:
    * 'exprun' - contains the copied necessary expruns from the source code +
                 the programatically generated expruns.
    * 'result' - contains the training data and the trained models.

    The training data should go into result/demonstration under some directory
    (eg. touch-apple).

    Args:
        setupname: Name for this experimental setup
        rootdir: Root directory path where setup will be created

    Returns:
        tuple: (exprun_path, result_path)
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
    Config().copy_experiment("visual_proprioception")

    return exprun_path, result_path


def get_visual_proprioception_sp(exp, device):
    """Get the sensor processing component for a visual proprioception experiment.

    Args:
        exp: Visual proprioception experiment configuration
        device: Device to load the model on

    Returns:
        Sensor processing object
    """
    spexp = Config().get_experiment(exp["sp_experiment"], exp["sp_run"])
    return sp_factory.create_sp(spexp, device)


def load_demonstrations_as_proprioception_training(
    sp,
    exp: Experiment,
    spexp: Experiment,
    exp_robot: Experiment,
    datasetname,
    proprioception_input_file,
    proprioception_target_file,
    device=None
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
        device: Device to load tensors to

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
                i, camera=camera, transform=transform, device=device
            )
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
    device=None
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
        device: Device to load tensors to

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
            # Collect images from all cameras for this timestep
            view_images = []
            skip_frame = False

            for camera in cameras[:num_views]:
                try:
                    sensor_readings, _ = demo.get_image(
                        i, camera=camera, transform=transform, device=device
                    )
                    view_images.append(sensor_readings)  # Keep batch dimension
                except Exception as e:
                    print(f"Skipping demo {demo_name} frame {i} - missing camera {camera}: {e}")
                    skip_frame = True
                    break

            if skip_frame:
                continue

            # Process through multiview sensor processor to get latent encoding
            # The sp.process expects a list of image tensors (one per view)
            z = sp.process(view_images)  # Returns encoded latent vector

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
    device=None
):
    """Loads raw images from multiple cameras for SP encoder training.

    Unlike load_multiview_demonstrations_as_proprioception_training, this function
    returns RAW images (not encoded) for training the sensor processing encoder itself.

    Args:
        exp: SP experiment config (NOT VP experiment)
        spexp: Same as exp for SP training
        exp_robot: Robot experiment for normalization
        datasetname: "training_data" or "validation_data"
        view_inputs_file: Path to save/load raw view images
        targets_file: Path to save/load targets
        device: Device to load tensors to

    Returns:
        Dictionary with:
        - 'view_inputs': List of tensors, one per camera view [N, C, H, W]
        - 'targets': Tensor of robot positions [N, 6]
    """
    if view_inputs_file.exists() and targets_file.exists():
        retval = {}
        retval["view_inputs"] = torch.load(view_inputs_file, weights_only=True)
        retval["targets"] = torch.load(targets_file, weights_only=True)
        print(f"***load_multiview_raw_images_as_training*** \n\t"
              f"Successfully loaded from cached files {view_inputs_file}")
        return retval

    transform = sp_helper.get_transform_to_sp(spexp)
    num_views = spexp.get("num_views", 2)

    # Dictionary to organize views by camera
    view_lists = {}
    targetlist = []

    print(f"Loading raw multiview training data with {num_views} views...")

    for val in exp[datasetname]:
        run, demo_name, cameras = val  # cameras can be list ["dev2", "dev3"] or string "dev2,dev3"

        # Handle cameras as either list or comma-separated string
        if isinstance(cameras, str):
            cameras = [c.strip() for c in cameras.split(",")]

        exp_demo = Config().get_experiment("demonstration", run)
        demo = Demonstration(exp_demo, demo_name)

        # Initialize view lists for cameras on first demo
        if not view_lists:
            for camera in cameras[:num_views]:
                view_lists[camera] = []

        for i in range(demo.metadata["maxsteps"]):
            # Get images from all cameras
            frame_images = {}
            skip_frame = False

            for camera in cameras[:num_views]:
                try:
                    sensor_readings, _ = demo.get_image(
                        i, device=device, transform=transform, camera=camera
                    )
                    frame_images[camera] = sensor_readings[0]  # Remove batch dim
                except Exception as e:
                    print(f"Skipping frame {i} - missing camera {camera}: {e}")
                    skip_frame = True
                    break

            if skip_frame:
                continue

            # Store images by camera
            for camera in cameras[:num_views]:
                view_lists[camera].append(frame_images[camera])

            # Get robot position
            rp = demo.get_action(i, "rc-position-target", exp_robot)
            anorm = rp.to_normalized_vector(exp_robot)
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
    for camera in sorted(view_lists.keys())[:num_views]:
        view_tensors.append(torch.stack(view_lists[camera]))

    retval = {}
    retval["view_inputs"] = view_tensors
    retval["targets"] = torch.stack(targetlist)

    # Save processed data
    torch.save(retval["view_inputs"], view_inputs_file)
    torch.save(retval["targets"], targets_file)
    print(f"Saved {len(targetlist)} training examples with {num_views} views each")

    return retval


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


class MultiViewDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for multi-view image data.

    This dataset handles multiple camera views per sample,
    returning a list of image tensors along with the target.
    """

    def __init__(self, view_inputs, targets):
        """
        Args:
            view_inputs: List of tensors, one per view. Each tensor has shape [N, C, H, W]
            targets: Tensor of targets with shape [N, output_dim]
        """
        self.view_inputs = view_inputs
        self.targets = targets
        self.num_samples = len(targets)
        self.num_views = len(view_inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (list of view images, target)
        """
        views = [view[idx] for view in self.view_inputs]
        target = self.targets[idx]
        return views, target


def collate_multiview(batch):
    """Custom collate function for multiview data.

    This function properly batches multi-view data where each sample
    contains a list of views.

    Args:
        batch: List of (views, target) tuples

    Returns:
        tuple: (list of batched view tensors, batched targets)
    """
    views_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    # Transpose views_list from [batch, views] to [views, batch]
    num_views = len(views_list[0])
    batched_views = []
    for v in range(num_views):
        view_batch = torch.stack([views_list[i][v] for i in range(len(views_list))])
        batched_views.append(view_batch)

    return batched_views, targets




# """
# visproprio_helper.py

# Helper functions for visual proprioception training
# """

# from exp_run_config import Config, Experiment
# Config.PROJECTNAME = "BerryPicker"

# import pathlib
# import torch
# import numpy as np
# from demonstration.demonstration import Demonstration
# # from demonstration.encoded_demonstrations import BCDemonstration

# from robot.al5d_position_controller import RobotPosition

# import sensorprocessing.sp_helper as sp_helper


# def external_setup(setupname, rootdir: pathlib.Path):
#     """Create an external directory 'setupname' under rootdir, where the generated exp/runs and results will go. This allows separating a set of experiments both for training and robot running.

#     Under this directory, there will be two directories:
#     * 'exprun' - contains the copied necessary expruns from the source code + the programatically generated expruns.
#     * 'result' - contains the training data and the trained models.

#     The training data should go into result/demonstration under some directory (eg. touch-apple).
#     """
#     rootdir = pathlib.Path(rootdir).expanduser()
#     setup_path = pathlib.Path(rootdir, setupname)
#     exprun_path = pathlib.Path(setup_path, "exprun")
#     result_path = pathlib.Path(setup_path, "result")

#     print(f"***Path for external experiments:\n{exprun_path}")
#     exprun_path.mkdir(exist_ok=True, parents=True)
#     print(f"***Path for external data:\n{result_path}")
#     result_path.mkdir(exist_ok=True, parents=True)

#     Config().set_exprun_path(exprun_path)
#     Config().set_results_path(result_path)

#     # Copy the necessary experiments into the external directory
#     Config().copy_experiment("robot_al5d")  # ✓ Needed by ALL training
#     Config().copy_experiment("demonstration")  # ✓ Needed by ALL training

#     # Copy ALL sensor processing experiments (even if not all used in this flow)    Config().copy_experiment("demonstration")
#     Config().copy_experiment("sensorprocessing_conv_vae")
#     Config().copy_experiment("sensorprocessing_propriotuned_cnn")
#     Config().copy_experiment("sensorprocessing_propriotuned_Vit")
#     Config().copy_experiment("sensorprocessing_aruco")
#     Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
#     Config().copy_experiment("sensorprocessing_propriotuned_Vit_multiview")
#     Config().copy_experiment("sensorprocessing_propriotuned_cnn_multiview")
#     Config().copy_experiment("sensorprocessing_conv_vae_concat_multiview")





#     Config().copy_experiment("visual_proprioception")




#     return exprun_path, result_path




# def load_demonstrations_as_proprioception_training(sp, exp: Experiment, spexp: Experiment, exp_robot: Experiment, datasetname, proprioception_input_file, proprioception_target_file, device=None):
#     """Loads all the images from the specified dataset and creates the input and target tensors. """
#     if proprioception_input_file.exists():
#         retval = {}
#         retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
#         retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
#         print(f"***load_demonstrations_as_proprioception_training*** \n\tSuccessfully loaded from cached files {proprioception_input_file} etc")
#         return retval

#     inputlist = []
#     targetlist = []
#     transform = sp_helper.get_transform_to_sp(spexp)

#     for val in exp[datasetname]:
#         run, demo_name, camera = val
#         exp_demo = Config().get_experiment("demonstration", run)
#         demo = Demonstration(exp_demo, demo_name)
#         for i in range(demo.metadata["maxsteps"]):
#             sensor_readings, _ = demo.get_image(i, camera=camera, transform=transform, device=device)
#             z = sp.process(sensor_readings)
#             # a = demo.get_action(i)
#             rp = demo.get_action(i, "rc-position-target", exp_robot)
#             #anorm = np.zeros(a.shape, np.float32)
#             # rp = RobotPosition.from_vector(exp_robot, a)
#             anorm = rp.to_normalized_vector(exp_robot)
#             inp = torch.from_numpy(z)
#             tgt = torch.from_numpy(anorm)
#             inputlist.append(inp)
#             targetlist.append(tgt)
#     retval = {}
#     retval["inputs"] = torch.stack(inputlist)
#     retval["targets"] = torch.stack(targetlist)
#     torch.save(retval["inputs"], proprioception_input_file)
#     torch.save(retval["targets"], proprioception_target_file)
#     print(f"***load_demonstrations_as_proprioception_training*** \n\tSuccessfully recalculated the proprioception training and saved it to {proprioception_input_file} etc")
#     return retval

# def load_multiview_demonstrations_as_proprioception_training(
#     sp,                              # Multiview sensor processor object
#     exp: Experiment,                 # VP experiment
#     spexp: Experiment,               # SP experiment
#     exp_robot: Experiment,
#     datasetname,                     # "training_data" or "validation_data"
#     proprioception_input_file,       # Explicit file path
#     proprioception_target_file,      # Explicit file path
#     device=None
# ):
#     """Loads all the images from the specified dataset from multiple cameras and creates
#     the input and target tensors for VP training.

#     This is for VP TRAINING - uses pre-trained multiview SP to encode images

#     Args:
#         sp: Pre-trained multiview sensor processor
#         exp: VP experiment config
#         spexp: SP experiment config
#         exp_robot: Robot experiment for normalization
#         datasetname: "training_data" or "validation_data"
#         proprioception_input_file: Path to save/load processed inputs
#         proprioception_target_file: Path to save/load processed targets
#         device: Device to load tensors to

#     Returns:
#         Dictionary with inputs and targets (encoded latents, not raw images)
#     """
#     if proprioception_input_file.exists():
#         retval = {}
#         retval["inputs"] = torch.load(proprioception_input_file, weights_only=True)
#         retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
#         print(f"***load_multiview_demonstrations_as_proprioception_training*** \n\tSuccessfully loaded from cached files {proprioception_input_file}")
#         return retval

#     inputlist = []
#     targetlist = []
#     transform = sp_helper.get_transform_to_sp(spexp)
#     num_views = spexp.get("num_views", 2)

#     print(f"Loading multiview VP training data with {num_views} views...")

#     # Loop through demonstrations using demopack system
#     for val in exp[datasetname]:
#         run, demo_name, cameras = val  # cameras is a list like ["dev2", "dev3"]
#         exp_demo = Config().get_experiment("demonstration", run)
#         demo = Demonstration(exp_demo, demo_name)

#         for i in range(demo.metadata["maxsteps"]):
#             # Collect images from all cameras for this timestep
#             view_images = []
#             skip_frame = False

#             for camera in cameras[:num_views]:
#                 try:
#                     sensor_readings, _ = demo.get_image(
#                         i, camera=camera, transform=transform, device=device
#                     )
#                     view_images.append(sensor_readings)  # Keep batch dimension
#                 except Exception as e:
#                     print(f"Skipping demo {demo_name} frame {i} - missing camera {camera}: {e}")
#                     skip_frame = True
#                     break

#             if skip_frame:
#                 continue

#             # Process through multiview sensor processor to get latent encoding
#             # The sp.process expects a list of image tensors (one per view)
#             z = sp.process(view_images)  # Returns encoded latent vector

#             # Get robot position
#             rp = demo.get_action(i, "rc-position-target", exp_robot)
#             anorm = rp.to_normalized_vector(exp_robot)

#             # Store encoded latent and target
#             inp = torch.from_numpy(z)
#             tgt = torch.from_numpy(anorm)
#             inputlist.append(inp)
#             targetlist.append(tgt)

#     retval = {}
#     retval["inputs"] = torch.stack(inputlist)
#     retval["targets"] = torch.stack(targetlist)
#     torch.save(retval["inputs"], proprioception_input_file)
#     torch.save(retval["targets"], proprioception_target_file)
#     print(f"***load_multiview_demonstrations_as_proprioception_training*** \n\tSuccessfully recalculated and saved to {proprioception_input_file}")
#     return retval

# def load_multiview_demonstrations_as_proprioception_training(exp_robot, task, proprioception_input_file, proprioception_target_file, num_views=2):
#     """

#     FIXME: Sahara: this needs to be changed to match the single-view one above.

#     Loads all the images of a task from multiple camera views, and processes it as two tensors
#     as input and target data for proprioception training.

#     Unlike the single-view version, this function doesn't use a sensor processor during data loading,
#     as the multi-view processing is handled separately.

#     Caches the processed results into the input and target file pointed in the config.
#     Remove those files to recalculate.

#     Args:
#         task: Task name to load demonstrations from
#         proprioception_input_file: Path to save/load processed inputs
#         proprioception_target_file: Path to save/load processed targets
#         num_views: Number of camera views to process

#     Returns:
#         Dictionary containing training and validation data splits
#     """

#     ### FIXME, draft from Lotzi

#     # for val in exp[datasetname]:
#     #     run, demo_name, cameras = val
#     #     exp_demo = Config().get_experiment("demonstration", run)
#     #     demo = Demonstration(exp_demo, demo_name)
#     #     for i in range(demo.metadata["maxsteps"]):
#     #         S = []
#     #         for cam in cameras:
#     #             sensor_readings, _ = demo.get_image(i, camera=cam, transform=transform, device=device)
#     #             z = sp.process(sensor_readings)
#     #             S.append(sensor_readings)
#     #         # create the concatenated ...
#     #         a = demo.get_action(i)
#     #         #anorm = np.zeros(a.shape, np.float32)
#     #         rp = RobotPosition.from_vector(exp_robot, a)
#     #         anorm = rp.to_normalized_vector(exp_robot)
#     #         inp = torch.from_numpy(z)
#     #         tgt = torch.from_numpy(anorm)
#     #         inputlist.append(inp)
#     #         targetlist.append(tgt)

#     ### END FIXME, draft from Lotzi

#     retval = {}
#     if proprioception_input_file.exists():
#         print(f"Loading cached data from {proprioception_input_file}")
#         retval["view_inputs"] = torch.load(proprioception_input_file, weights_only=True)
#         retval["targets"] = torch.load(proprioception_target_file, weights_only=True)
#     else:
#         demos_dir = pathlib.Path(Config()["demos"]["directory"])
#         task_dir = pathlib.Path(demos_dir, "demos", task)

#         # Lists to store multi-view images and targets
#         view_lists = {}  # Dictionary to organize views by camera
#         targetlist = []

#         print(f"Loading demonstrations from {task_dir}")
#         for demo_dir in task_dir.iterdir():
#             if not demo_dir.is_dir():
#                 continue

#             print(f"Processing demonstration: {demo_dir.name}")
#             # Create BCDemonstration with multi-camera support
#             bcd = BCDemonstration(
#                 demo_dir,
#                 sensorprocessor=None,
#                 cameras=None  # This will detect all available cameras
#             )

#             # Initialize view lists if not already done
#             if not view_lists:
#                 for camera in bcd.cameras:
#                     view_lists[camera] = []

#             # Process each timestep
#             for i in range(bcd.trim_from, bcd.trim_to):
#                 # Get all images for this timestep
#                 all_images = bcd.get_all_images(i)

#                 # If we don't have all required views, skip this timestep
#                 if len(all_images) < num_views:
#                     print(f"  Skipping timestep {i} - only {len(all_images)}/{num_views} views available")
#                     continue

#                 # Collect images from each camera
#                 for camera, (sensor_readings, _) in all_images.items():
#                     if camera in view_lists:
#                         view_lists[camera].append(sensor_readings[0])

#                 # Get the robot action for this timestep
#                 a = bcd.get_a(i)
#                 rp = RobotPosition.from_vector(exp_robot, a)
#                 anorm = rp.to_normalized_vector()
#                 targetlist.append(torch.from_numpy(anorm))

#         # Ensure we have the same number of frames for each view
#         min_frames = min(len(view_list) for view_list in view_lists.values())
#         if min_frames < len(targetlist):
#             print(f"Truncating dataset to {min_frames} frames (from {len(targetlist)})")
#             targetlist = targetlist[:min_frames]
#             for camera in view_lists:
#                 view_lists[camera] = view_lists[camera][:min_frames]

#         # Stack tensors for each view
#         view_tensors = []
#         for camera in sorted(view_lists.keys())[:num_views]:  # Take only the required number of views
#             view_tensors.append(torch.stack(view_lists[camera]))

#         # Create multi-view input tensor list [num_views, num_samples, C, H, W]
#         retval["view_inputs"] = view_tensors
#         retval["targets"] = torch.stack(targetlist)

#         # Save processed data
#         torch.save(retval["view_inputs"], proprioception_input_file)
#         torch.save(retval["targets"], proprioception_target_file)
#         print(f"Saved {len(targetlist)} training examples with {num_views} views each")

#     # Separate the training and validation data
#     length = len(retval["targets"])
#     rows = torch.randperm(length)

#     # Shuffle targets
#     shuffled_targets = retval["targets"][rows]

#     # Shuffle each view input using the same row indices
#     shuffled_view_inputs = []
#     for view_tensor in retval["view_inputs"]:
#         shuffled_view_inputs.append(view_tensor[rows])

#     # Split into training (67%) and validation (33%) sets
#     training_size = int(length * 0.67)

#     # Training data
#     retval["view_inputs_training"] = [view[:training_size] for view in shuffled_view_inputs]
#     retval["targets_training"] = shuffled_targets[:training_size]

#     # Validation data
#     retval["view_inputs_validation"] = [view[training_size:] for view in shuffled_view_inputs]
#     retval["targets_validation"] = shuffled_targets[training_size:]

#     print(f"Created {training_size} training examples and {length - training_size} validation examples")
#     return retval


