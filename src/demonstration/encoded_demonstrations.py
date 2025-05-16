"""
encoded_demonstrations.py

A set of demonstrations that had already been encoded with the sensor processor and paired with the actions. Primarily used as preprocessed training data for behavior cloning. 
"""
import sys
sys.path.append("..")

import helper
import pathlib
import json
import numpy as np
from sensorprocessing.sp_helper import load_picturefile_to_tensor

from demonstration import Demonstration

class EncodedDemonstrations():
    """This class encapsulates loading a demonstration with the intention to convert it into training data.

    FIXME: not converted yet to the model. 
    FIXME: the parameters of this should be the exp/runs 

    This code is a training helper which encapsulates one behavior cloning demonstration, which is a sequence of form $\{(s_0, a_0), ...(s_n, a_n)\}$.

    In practice, however, we want to create a demonstration that maps the latent encodings to actions $\{(z_0, a_0), ...(z_n, a_n)\}$

    The transformation of $s \rightarrow z$ is done through an object of type AbstractSensorProcessing.

    The class now supports multiple camera views per timestep.


    """

    def __init__(self, exp_demo, demo, sensorprocessor, actiontype="rc-position-target", cameras=None):
        self.sensorprocessor = sensorprocessor
        assert actiontype in ["rc-position-target", "rc-angle-target", "rc-pulse-target"]
        self.actiontype = actiontype

        # analyze the directory
        self.available_cameras, self.maxsteps = helper.analyze_demo(self.demo_dir)

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
        demo_config_path = pathlib.Path(self.demo_dir, "_demonstration.json")
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


    def get_z(self, i, camera=None):
        """
        Get the processed sensor data for a specific timestep and camera

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
        """
        if camera is None:
            camera = self.cameras[0]

        filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
        val = self.sensorprocessor.process_file(filepath)
        return val



    def get_a(self, i):
        """Get the action data for a specific timestep"""
        filepath = pathlib.Path(self.demo_dir, f"{i:05d}.json")
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
