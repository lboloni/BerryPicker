"""
sp_helper.py

Transform image files or video captures into sensor-processing tensors.
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from PIL import Image
import torch
from torchvision import transforms

class SensorPreprocessor:
    """Shared image preprocessing for sensor-processing inference entry points.

    The preprocessor owns the geometric transform, RGB conversion, batch
    creation, and device placement.  Its output is a raw ``[0, 1]`` tensor;
    model-specific normalization (such as ImageNet normalization for ViT)
    remains the responsibility of the model encoder.
    """

    def __init__(self, exp=None, *, transform=None, device=None):
        if transform is None:
            if exp is None:
                raise ValueError("exp is required when no transform is supplied")
            self.image_size = (exp["image_size"][0], exp["image_size"][1])
            transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_size = None

        self.transform = transform
        self.device = device

    def _device(self):
        return self.device if self.device is not None else Config().runtime["device"]

    def prepare_image(self, image):
        """Return a device batch and a CPU image tensor for display."""
        image_tensor = self.transform(image.convert("RGB"))
        image_tensor_for_pic = image_tensor.permute(1, 2, 0)
        return image_tensor.unsqueeze(0).to(self._device()), image_tensor_for_pic

    def from_image(self, image):
        """Preprocess one PIL image into a single-item model batch."""
        return self.prepare_image(image)[0]

    def from_file(self, picture_file):
        """Load and preprocess one image file into a single-item model batch."""
        with Image.open(picture_file) as image:
            return self.from_image(image)

    def from_capture(self, image_from_camera):
        """Preprocess one RGB camera frame into a single-item model batch."""
        return self.from_image(Image.fromarray(image_from_camera))


def get_transform_to_sp(exp):
    """Return the shared geometric transform used by training and inference."""
    return SensorPreprocessor(exp).transform

def load_image_to_batch(image, transform):
    """Compatibility wrapper around :class:`SensorPreprocessor`."""
    return SensorPreprocessor(transform=transform).prepare_image(image)

def load_picturefile_to_tensor(picture_file, transform):
    """Compatibility wrapper for loading a file and its display tensor."""
    with Image.open(picture_file) as image:
        return SensorPreprocessor(transform=transform).prepare_image(image)

def load_capture_to_tensor(image_from_camera, transform):
    """Compatibility wrapper for processing a camera frame and display tensor."""
    return SensorPreprocessor(transform=transform).prepare_image(
        Image.fromarray(image_from_camera)
    )
