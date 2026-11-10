"""
transfer_data.py

Data for the transfer experiments (DESIGN-Transfer.md): the frames of groups of
demonstrations as uint8 tensors, the robot masks and the background
augmentation (design D3), the K-shot subsets and the conversion of the errors
to physical units.

An item is a [pack, demonstration, camera] triple, as in the training_data
fields of the other exp/runs. The pack must have been imported into the
current results path (import_demopack).
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import math
import torch
import torch.nn.functional as F

from demonstration.demonstration import Demonstration
from robot.position_factory import action_key, position_fields
from sensorprocessing.sp_helper import get_transform_to_sp


def load_frames(items, image_size, exp_robot=None, frame_step=1):
    """Loads every frame_step-th frame of the items.

    Returns a dict with "images" (uint8 [N, 3, H, W]), "targets" (float32
    [N, k], the normalized positions of exp_robot, or None when exp_robot is
    None) and "demo_index" (long [N], the index of the item of each frame).
    """
    transform = get_transform_to_sp({"image_size": image_size})
    images, targets, demo_index = [], [], []
    for index, (pack, demo_name, camera) in enumerate(items):
        demo = Demonstration(Config().get_experiment("demonstration", pack), demo_name)
        for i in range(0, demo.metadata["maxsteps"], frame_step):
            sensor_readings, _ = demo.get_image(i, camera=camera, transform=transform)
            images.append((sensor_readings[0].cpu() * 255).round().to(torch.uint8))
            if exp_robot is not None:
                position = demo.get_action(i, action_key(exp_robot), exp_robot)
                targets.append(torch.from_numpy(position.to_normalized_vector(exp_robot)).float())
            demo_index.append(index)
    return {"images": torch.stack(images),
            "targets": torch.stack(targets) if targets else None,
            "demo_index": torch.tensor(demo_index)}


def to_float(images):
    """uint8 images to float images in [0, 1]"""
    return images.float() / 255


# ---------------------------------------------------------------------------
# D3: robot masks against the median background of a demonstration
# ---------------------------------------------------------------------------

def median_background(images, step=10):
    """The per-pixel median of every step-th image: uint8 [3, H, W]"""
    return images[::step].float().median(dim=0).values.round().to(torch.uint8)


def foreground_mask(images, background, threshold, dilate):
    """The pixels that differ from the background by more than threshold (on
    [0, 1] images) in any channel, dilated by dilate pixels: uint8 [B, 1, H, W]"""
    diff = (to_float(images) - to_float(background)).abs().amax(dim=1, keepdim=True)
    mask = (diff > threshold).float()
    if dilate > 0:
        mask = F.max_pool2d(mask, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
    return mask.to(torch.uint8)


def foreground_masks(images, demo_index, threshold, dilate):
    """The masks of all frames, each against the median background of its
    own demonstration: uint8 [N, 1, H, W]"""
    masks = torch.zeros(images.shape[0], 1, *images.shape[2:], dtype=torch.uint8)
    for index in demo_index.unique():
        rows = demo_index == index
        background = median_background(images[rows])
        masks[rows] = foreground_mask(images[rows], background, threshold, dilate)
    return masks


def apply_mask(images, masks, fill):
    """Float images [B, 3, H, W] with the background replaced by fill, which is
    a scalar (gray: 0.5) or a float tensor [B, 3, H, W] or [3, H, W]"""
    masks = masks.float()
    return images * masks + fill * (1 - masks)


def synthetic_backgrounds(count, image_size, generator):
    """Smooth random color fields: count float [3, H, W] backgrounds"""
    coarse = torch.rand(count, 3, 4, 4, generator=generator)
    return F.interpolate(coarse, size=tuple(image_size), mode="bicubic", align_corners=False).clamp(0, 1)


def backgrounds_of_items(items, image_size):
    """The median backgrounds of the given demonstrations: float [M, 3, H, W]"""
    backgrounds = []
    for item in items:
        frames = load_frames([item], image_size, frame_step=10)
        backgrounds.append(to_float(median_background(frames["images"], step=1)))
    return torch.stack(backgrounds)


def color_jitter(images, strength, generator):
    """Random brightness, contrast and saturation per image, each in
    [1 - strength, 1 + strength]"""
    b = images.shape[0]
    factors = 1 + strength * (2 * torch.rand(3, b, 1, 1, 1, generator=generator) - 1)
    factors = factors.to(images.device)
    images = images * factors[0]
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * factors[1] + mean
    gray = images.mean(dim=1, keepdim=True)
    images = (images - gray) * factors[2] + gray
    return images.clamp(0, 1)


def masked_batch(images, masks, *, backgrounds=None, jitter=0.0, generator=None):
    """A float batch from uint8 images. With masks: the background is gray, or
    a random one of backgrounds (float [M, 3, H, W]) when given. With jitter,
    the colors are jittered after the masking."""
    x = to_float(images)
    if masks is not None:
        if backgrounds is None:
            x = apply_mask(x, masks, 0.5)
        else:
            choice = torch.randint(backgrounds.shape[0], (x.shape[0],), generator=generator)
            x = apply_mask(x, masks, backgrounds[choice])
    if jitter > 0:
        x = color_jitter(x, jitter, generator)
    return x


# ---------------------------------------------------------------------------
# K-shot subsets and physical units
# ---------------------------------------------------------------------------

def kshot_rows(n, k, seed):
    """k random rows of n (all n, in order, when k is None or k >= n)"""
    if k is None or k >= n:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:k]


# the AL5D position fields and their units (robot_al5d/position_controller_00:
# angles in degrees, the gripper in percent, height and distance in the
# units of the AL5D position controller)
AL5D_UNITS = {"height": "al5d", "distance": "al5d", "heading": "deg",
              "wrist_angle": "deg", "wrist_rotation": "deg", "gripper": "%"}


def physical_errors(exp_robot, rmse_normalized):
    """The per-joint RMSE in physical units: a list of (value, unit). The joint
    vectors of the Franka, the G1 and the Inspire hand are in radians and are
    converted to degrees."""
    retval = []
    for field, rmse in zip(position_fields(exp_robot), rmse_normalized):
        span = exp_robot["POS_MAX"][field] - exp_robot["POS_MIN"][field]
        if exp_robot["robot_name"] == "al5d":
            retval.append((rmse * span, AL5D_UNITS[field]))
        else:
            retval.append((math.degrees(rmse * span), "deg"))
    return retval
