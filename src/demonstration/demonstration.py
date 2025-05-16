"""
demonstration.py

A class that wraps the data files in the demonstration, and allows convenient accesss to them. 
"""
import json
import sys

import cv2
import yaml
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
    """This class encapsulates all the convenience functions for a demonstration, including loading images etc. A demonstration stores a sequence of images and the corresponding actions. 
    The images can be stored either as a sequence of images, or as a video. Metadata about the demonstration, including annotations are stored in the _metadata.yaml file.
    """
    
    def save_metadata(self):
        """Saves the metadata of the demonstration to a file"""
        metadata_path = pathlib.Path(self.demo_dir, "_metadata.yaml")
        with open(metadata_path, "w") as file:
            yaml.dump(self.metadata, file, indent=4)
        action_path = pathlib.Path(self.demo_dir, "_action.yaml")
        with open(action_path, "w") as file:
            yaml.dump(self.actions, file, indent=4)


    def parse_image_based_demonstration(self):
        """Utility function to parse a demonstration that is stored as a sequence of images. We assume that the images have the format of {i:05d}_{camera}.jpg, where i is the timestep and camera is the camera name. This function creates the metadata file. If the images are stored as a video, we assume that this is a new demonstration, and the metadata already exists. 
        """
        self.metadata = {}
        # Set default values for metadata
        self.metadata["stored_as_video"] = False
        # Analyzes the demonstration to get the list of cameras. 
        cameraset = {}
        maxsteps = 0
        for a in self.demo_dir.iterdir():
            if a.name.endswith(".json") and a.name.startswith("0"):
                count = int(a.name.split(".")[0])
                maxsteps = max(maxsteps, count)
            if a.name.endswith(".jpg"):
                cameraname = a.name[6:-4]
                cameraset[cameraname] = cameraname
                self.metadata["stored_as_images"] = True
        if not cameraset:
            raise ValueError("No cameras found in the demonstration directory")
        self.metadata["cameras"] = sorted(cameraset.keys())
        self.metadata["maxsteps"] = maxsteps + 1
        # load the content of json files into the metadata
        self.actions = []
        for i in range(self.metadata["maxsteps"]):
            json_path = pathlib.Path(self.demo_dir, f"{i:05d}.json")
            with open(json_path) as file:
                data = json.load(file)
            self.actions.append(data)
        self.save_metadata()

    def __init__(self, exp, demo):
        """Initializes the demonstration, based on an experiment"""
        self.exp = exp
        self.demo = demo
        self.demo_dir = pathlib.Path(exp.data_dir(), demo)
        # load the _metadata.yaml file, if it exists, otherwise infer it from the directory
        metadata_path = pathlib.Path(self.demo_dir, "_metadata.yaml")
        if metadata_path.exists():
            with open(metadata_path) as file:
                self.metadata = yaml.safe_load(file)
            action_path = pathlib.Path(self.demo_dir, "_action.yaml")
            with open(action_path) as file:
                self.actions = yaml.safe_load(file)
        else:
            self.parse_image_based_demonstration()    
        self.videocap = {} # placeholder for open videos

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def _move_to_video_per_camera(self, cam, delete_img_files = False):
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


    def get_image(self, i, camera=None, transform=None):
        """Gets the image as a pair of (sensor_readings, image) from the demonstration. Prefers loading it from image if it is stored as images, otherwise loads it from the video.

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
            transform: Optional transform to apply to the image
        """
        if self.metadata["stored_as_images"]:
            return self.get_image_from_imagefile(i, camera=camera, transform=transform)
        else:
            return self.get_image_from_video(i, camera=camera, cache=True)

    def get_image_from_video(self, i, camera=None, cache=False):
        """Extracts an image from the video. 
        If cache is False, the function closes the open file
        """
        assert self.metadata["stored_as_video"], "The demonstration is not stored as video"        
        if camera is None:
            camera = self.metadata["cameras"][0]    
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
        for cam in self.metadata["cameras"]:
            self._move_to_video_per_camera(cam, delete_img_files)
        self.metadata["stored_as_images"] = not delete_img_files
        self.metadata["stored_as_video"] = True
        self.save_metadata()

    def get_image_path(self, i, camera=None):
        """Returns the path to the image, if the demo is stored as independent image files."""
        if camera is None:
            camera = self.metadata["cameras"][0]
        filepath = pathlib.Path(self.demo_dir, f"{i:05d}_{camera}.jpg")
        return filepath
    
    def get_video_path(self, camera=None):
        """Returns the path to the video file for the specified camera, if it is stored as video file"""
        if not camera:
            camera = self.metadata["cameras"][0]
        video_path = pathlib.Path(self.demo_dir, f"video_{camera}.mp4")
        return video_path

    def get_image_from_imagefile(self, i, camera=None, transform=None):
        """
        Gets the image as a torch batch from an image file. 

        Args:
            i: The timestep index
            camera: Which camera to use. If None, uses the first camera.
            transform: Optional transform to apply to the image
        """
        assert self.metadata["stored_as_images"], "The demonstration is not stored as images"
        filepath = self.get_image_path(i, camera)
        sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
        return sensor_readings, image

    def get_all_images_from_imagefile(self, i, transform=None):
        """
        Gets images from all cameras for a specific timestep

        Args:
            i: The timestep index
            transform: Optional transform to apply to the images

        Returns:
            Dictionary mapping camera names to (sensor_readings, image) tuples
        """
        images = {}
        for camera in self.metadata["cameras"]:
            filepath = self.get_image_path(i, camera)
            if filepath.exists():
                sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
                images[camera] = (sensor_readings, image)

        return images