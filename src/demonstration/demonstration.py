"""
demonstration.py

A class that wraps the data files in the demonstration, and allows convenient accesss to them. 
"""
import json
import sys

import cv2
sys.path.append("..")

import pprint
import pathlib
from sensorprocessing.sp_helper import load_picturefile_to_tensor, load_capture_to_tensor

def list_demos(exp):
    """List all the demonstrations described in an exp/run. This can be passed as the second argument of the Demonstration constructor."""
    demos = [item.name for item in exp.data_dir().iterdir() if item.is_dir()]
    return demos

def select_demo(exp, force_choice=None, force_name=None):
    """Interactively select one one the demonstrations, or force the choice to a number or a name."""
    if force_name:
        return force_name
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
        # load the _demonstration.json file, if it exists
        metadata_path = pathlib.Path(self.demo_dir, "_demonstration.json")
        self.metadata = json.load(metadata_path.open()) if metadata_path.exists() else {}
        if "maxsteps" in self.metadata:
            self.maxsteps = self.metadata["maxsteps"]
        else:
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
        self.maxsteps += 1 # make it a proper count
        self.cameras = sorted(cameraset.keys())
        self.videocap = {} # placeholder for open videos

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def move_to_video_per_camera(self, cam, delete_img_files = False):
        """Move the content of a specific camera into video"""
        if cam in self.exp["cameras"]:
            params = self.exp["cameras"][cam]
        else:
            params = self.exp["cameras"]["all"]
        video_path = self.get_video_path()
        image_paths = []
        # Initialize video writer
        if not video_path.exists():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, params["fps"], (params["width"], params["height"]))
            for i in range(self.maxsteps):
                img_path = self.get_image_path(i, camera=cam)
                image_paths.append(img_path)
                frame = cv2.imread(str(img_path))
                out.write(frame)
            out.release()
        # if specified, delete the image files
        if delete_img_files:
            for img_path in image_paths:
                img_path.unlink()
        
    def get_image_from_video(self, i, camera=None, cache=False):
        """Extracts an image from the video. 
        FIXME: this function opens the video file, seeks and closes it, so it should be very inefficient, we should store the open one in the demonstration instead
        If cache is False, the function closes the open file
        """
        if camera is None:
            camera = self.cameras[0]    
        if camera not in self.videocap:    
            video_path = self.get_video_path(camera)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
            self.videocap[camera] = cap
        else:
            cap = self.videocap[camera]
        ret, frame = cap.read()
        if ret:
            # CV2 reads by default in BGR... 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(output_image, frame)    
            image_to_process, image_to_show = load_capture_to_tensor(frame, transform=None)
        else:
            print(f"Could not read frame {i}")
            image_to_process = None
            image_to_show = None
        if not cache:
            self.videocap[camera].release()
            self.videocap.pop(camera)
        return image_to_process, image_to_show     

    def move_to_video(self, delete_img_files = False):
        """Moves to the video the content all all cameras"""
        for cam in self.cameras:
            self.move_to_video_per_camera(cam, delete_img_files)

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