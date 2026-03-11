"""
sp_helper.py

Helper functions for sensor processing. Loading image files or video captures into formats that are suitable to be fed into the video models.
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from PIL import Image
import torch
from torchvision import transforms

def get_transform_to_sp(exp):
    """Creates a transform object that transforms a figure into the right size tensor that is suitable for the sensor processing model."""
    image_size = exp["image_size"][0]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return transform

def load_image_to_batch(image, transform):
    """Takes a PIL image and transforms it into a tensor representing a single element batch, moved to the appropriate device.
    This format should be approprate for feeding into the sensor processing model."""
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)
    image_tensor_for_pic = image_tensor.permute(1, 2, 0)
    image_batch = image_tensor.unsqueeze(0)
    image_batch_device = image_batch.to(Config().runtime["device"])
    return image_batch_device, image_tensor_for_pic

def load_picturefile_to_tensor(picture_file, transform):
    """Loads an image from a file using PIL and returns it as a single image batch and the image in a displayable format.
    """
    image = Image.open(picture_file)
    return load_image_to_batch(image, transform)

def load_capture_to_tensor(image_from_camera, transform):
    """Gets an image as returned by the camera controller and transforms it into a single element batch, moved to the appropriate device. Returns the batch and the image in a displayable format.
    """
    image = Image.fromarray(image_from_camera)
    return load_image_to_batch(image, transform)

