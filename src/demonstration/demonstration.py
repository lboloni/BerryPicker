"""
demonstration.py

A class that wraps the data files in the demonstration, and allows convenient accesss to them. 
"""
import sys
sys.path.append("..")

import pprint
import pathlib
from sensorprocessing.sp_helper import load_picturefile_to_tensor, load_capture_to_tensor


def list_demos(exp):
    """List all the demonstrations described in an exp/run. This can be passed as the second argument of the Demonstration constructor."""
    demos = [item.name for item in exp.data_dir().iterdir() if item.is_dir()]
    return demos

def select_demo(exp, force_choice=None):
    """Interactively select one one the demonstrations, or force the choice to a number."""
    demodirs = [item for item in exp.data_dir().iterdir() if item.is_dir()]
    demos_dict = {}
    for i, t in enumerate(demodirs):
        demos_dict[i] = t
    print("A pop up dialog will appear now. Enter the number of demonstration.", flush=True)
    for key in demos_dict:
        print(f"\t{key}: {demos_dict[key].name}")
    # FIXME: for easier debugging
    if not force_choice:
        inpval = input("Choose the demonstration: ")
    else:
        inpval = str(force_choice)
    # inpval = "0"
    # print(inpval)
    if inpval:
        demo_id = int(inpval)
        if demo_id in demos_dict:
            demo_dir = demos_dict[demo_id]
            print(f"You chose demonstration: {demo_dir.name}")
        else:
            print(f"No such demo: {demo_id}")
    return demo_dir.name

class Demonstration:
    """This class encapsulates all the convenience functions for a demonstration, including loading images etc. """
    
    def __init__(self, exp, demo):
        """Initializes the demonstration, based on an experiment"""
        self.exp = exp
        self.demo = demo
        self.demo_dir = pathlib.Path(exp.data_dir(), demo)
        self.maxsteps = -1
        # Analyzes the demonstration to get the list of cameras. 
        # FIXME: this does the analysis based on the picture names. Instead, everything should be in the _demonstration.json.
        cameraset = {}
        for a in self.demo_dir.iterdir():
            if a.name.endswith(".json") and a.name.startswith("0"):
                count = int(a.name.split(".")[0])
                self.maxsteps = max(self.maxsteps, count)
            if a.name.endswith(".jpg"):
                cameraname = a.name[6:-4]
                cameraset[cameraname] = cameraname
        self.cameras = sorted(cameraset.keys())
        self.videocap = {} # placeholder for open videos

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def get_image_path(self, i, camera=None):
        """Returns the path to the image, if the demo is stored as independent image files."""
        if camera is None:
            camera = self.cameras[0]
        filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
        return filepath
    
    def get_video_path(self, camera=None):
        """Returns the path to the video file for the specified camera, if it is stored as video file"""
        if not camera:
            camera = self.cameras[0]
        video_path = pathlib.Path(self.demo_dir, f"video_{camera}.mp4")
        return video_path

    def get_image(self, i, camera=None, transform=None):
        """
        Gets the image as a torch batch

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
            transform: Optional transform to apply to the image
        """

        filepath = self.get_image_path(i, camera)
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
            filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
            if filepath.exists():
                sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
                images[camera] = (sensor_readings, image)

        return images