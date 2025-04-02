"""
demo_to_trainingdata.py

Create training data from demonstrations with support for multiple camera views.
"""
import sys
sys.path.append("..")

import torch
import helper
import pathlib
import json
from pprint import pformat
import numpy as np
from sensorprocessing.sp_helper import load_picturefile_to_tensor


def create_RNN_training_sequence_xy(x_seq, y_seq, sequence_length):
    """Create supervised training data for RNNs such as LSTM from two sequences. In this data, from a string of length sequence_length in x_seq we are predicting the next item in y_seq.
    Returns the results as tensors
    """
    # Prepare training data
    total_length = x_seq.shape[0]
    inputs = []
    targets = []
    for i in range(total_length - sequence_length):
        # Input is a subsequence of length `sequence_length`
        input_seq = x_seq[i:i + sequence_length]
        # Shape: [sequence_length, latent_size]

        # Target is the next vector after the input sequence
        target = y_seq[i + sequence_length]
        # Shape: [output_size]

        # Append to lists
        inputs.append(torch.tensor(input_seq))
        targets.append(torch.tensor(target))

    # Convert lists to tensors for training
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets


class BCDemonstration:
    """This class encapsulates loading a demonstration with the intention to convert it into training data.

    This code is a training helper which encapsulates one behavior cloning demonstration, which is a sequence of form $\{(s_0, a_0), ...(s_n, a_n)\}$.

    In practice, however, we want to create a demonstration that maps the latent encodings to actions $\{(z_0, a_0), ...(z_n, a_n)\}$

    The transformation of $s \rightarrow z$ is done through an object of type AbstractSensorProcessing.

    The class now supports multiple camera views per timestep.
    """

    def __init__(self, source_dir, sensorprocessor, actiontype="rc-position-target", cameras=None):
        self.source_dir = source_dir
        self.sensorprocessor = sensorprocessor
        assert actiontype in ["rc-position-target", "rc-angle-target", "rc-pulse-target"]
        self.actiontype = actiontype

        # analyze the directory
        self.available_cameras, self.maxsteps = helper.analyze_demo(source_dir)

        # Set cameras to use
        if cameras is None:
            # Default to using all available cameras
            self.cameras = self.available_cameras
        else:
            # Verify that requested cameras exist
            for cam in cameras:
                if cam not in self.available_cameras:
                    raise ValueError(f"Camera {cam} not found in demonstration data")
            self.cameras = cameras

        # read in _demonstration.json, load the trim values
        demo_config_path = pathlib.Path(self.source_dir, "_demonstration.json")
        if demo_config_path.exists():
            with open(demo_config_path) as file:
                data = json.load(file)
            self.trim_from = data.get("trim-from", 1)
            self.trim_to = data.get("trim-to", -1)
            if self.trim_to == -1:
                self.trim_to = self.maxsteps
        else:
            # Default values if config not found
            self.trim_from = 1
            self.trim_to = self.maxsteps

    def read_z_a(self, fusion_mode="concat"):
        """
        Reads in the demonstrations for z and a and returns them in the form of float32 numpy arrays

        Args:
            fusion_mode (str): How to combine multiple camera views. Options:
                - "concat": Concatenate feature vectors from all cameras
                - "average": Average feature vectors from all cameras
        """
        z = []
        a = []

        for i in range(self.trim_from, self.trim_to):
            if fusion_mode == "concat":
                # Concatenate features from all cameras
                z_combined = []
                for camera in self.cameras:
                    z_val = self.get_z(i, camera)
                    z_combined.append(z_val)
                zval = np.concatenate(z_combined)
            elif fusion_mode == "average":
                # Average features across cameras
                z_combined = []
                for camera in self.cameras:
                    z_val = self.get_z(i, camera)
                    z_combined.append(z_val)
                zval = np.mean(z_combined, axis=0)
            else:
                raise ValueError(f"Unknown fusion mode: {fusion_mode}")

            z.append(zval)
            a.append(self.get_a(i))

        return np.array(z, dtype=np.float32), np.array(a, dtype=np.float32)

    def __str__(self):
        return pformat(self.__dict__)

    def get_z(self, i, camera=None):
        """
        Get the processed sensor data for a specific timestep and camera

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
        """
        if camera is None:
            camera = self.cameras[0]

        filepath = pathlib.Path(self.source_dir, f"{i:05d}_{camera}.jpg")
        val = self.sensorprocessor.process_file(filepath)
        return val

    def get_image(self, i, camera=None, transform=None):
        """
        Gets the image as a torch batch

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
            transform: Optional transform to apply to the image
        """
        if camera is None:
            camera = self.cameras[0]

        filepath = pathlib.Path(self.source_dir, f"{i:05d}_{camera}.jpg")
        sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
        return sensor_readings, image

    def get_all_images(self, i, transform=None):
        """
        Gets images from all cameras for a specific timestep

        Args:
            i: The timestep index
            transform: Optional transform to apply to the images

        Returns:
            Dictionary mapping camera names to (sensor_readings, image) tuples
        """
        images = {}
        for camera in self.cameras:
            filepath = pathlib.Path(self.source_dir, f"{i:05d}_{camera}.jpg")
            if filepath.exists():
                sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
                images[camera] = (sensor_readings, image)

        return images

    def get_a(self, i):
        """Get the action data for a specific timestep"""
        filepath = pathlib.Path(self.source_dir, f"{i:05d}.json")
        with open(filepath) as file:
            data = json.load(file)
        if self.actiontype == "rc-position-target":
            datadict = data["rc-position-target"]
            a = list(datadict.values())
            return a
        if self.actiontype == "rc-angle-target":
            datadict = data["rc-angle-target"]
            a = list(datadict.values())
            return a
        if self.actiontype == "rc-pulse-target":
            datadict = data["rc-pulse-target"]
            a = list(datadict.values())
            return a
