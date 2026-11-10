"""
transfer_sp.py

The sensor processing (SP) classes of the transfer experiments
(DESIGN-Transfer.md), and create_transfer_sp, which creates them and falls
back to sensorprocessing.sp_factory.create_sp for all the other classes. The
classes are not registered in sp_factory: only the notebooks of this
directory use them.
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import numpy as np
import torch

from sensorprocessing import sp_factory
from sensorprocessing.sensor_processing import AbstractSensorProcessing, SingleViewEncoderSensorProcessing
from visual_proprioception.transfer.transfer_data import load_frames, masked_batch, to_float
from visual_proprioception.transfer.transfer_models import (
    MetaRidgeEncoder, TransferEncoder, ViTRandomProjection, adapt_batchnorm)


class TransferEncoderSensorProcessing(SingleViewEncoderSensorProcessing):
    """A trained TransferEncoder (single-source, D1, D2, D3)"""

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = TransferEncoder(exp)
        self.load_encoder_checkpoint(required=True, label="TransferEncoder")


class ViTRandomProjectionSensorProcessing(SingleViewEncoderSensorProcessing):
    """The frozen ImageNet ViT with a random projection (reference R1); not trained"""

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = ViTRandomProjection(exp)


class MetaRidgeSensorProcessing(SingleViewEncoderSensorProcessing):
    """A meta-trained MetaRidgeEncoder (D5)"""

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = MetaRidgeEncoder(exp)
        self.load_encoder_checkpoint(required=True, label="MetaRidgeEncoder")

    def log_lambda(self):
        return self.enc.log_lambda.item()


class AdaBNSensorProcessing(AbstractSensorProcessing):
    """A trained SP whose BatchNorm statistics are re-estimated on unlabeled
    frames of the target robot (D4a). The adaptation runs at construction.

    exp: base_sp_experiment, base_sp_run, image_size and latent_size (those of
    the base SP), adapt_scope (projection: the BatchNorm layers of the
    projection of a TransferEncoder; all: every BatchNorm layer, e.g. those of
    the Conv-VAE), adaptation_data ([pack, demo, camera] items; their actions
    are never read), adaptation_frame_step and adaptation_batch_size.
    """

    def __init__(self, exp):
        super().__init__(exp)
        base_exp = Config().get_experiment(exp["base_sp_experiment"], exp["base_sp_run"])
        self.base = create_transfer_sp(base_exp)
        network, forward = self.network_and_forward()
        if exp["adapt_scope"] == "projection":
            modules = list(network.projection.modules())
        else:
            modules = list(network.modules())
        frames = load_frames(exp["adaptation_data"], exp["image_size"],
                             frame_step=exp["adaptation_frame_step"])["images"]
        device = Config().runtime["device"]
        batches = (to_float(frames[i:i + exp["adaptation_batch_size"]]).to(device)
                   for i in range(0, frames.shape[0], exp["adaptation_batch_size"]))
        count = adapt_batchnorm(network, modules, forward, batches)
        print(f"AdaBN: re-estimated {count} BatchNorm layers on {frames.shape[0]} target frames")

    def network_and_forward(self):
        """The network of the base SP and the function that runs it on a batch"""
        if hasattr(self.base, "enc"):
            return self.base.enc, getattr(self.base.enc, self.base.encoder_method)
        # the Conv-VAE (sp_conv_vae.ConvVaeSensorProcessing) keeps its network in model
        return self.base.model, self.base.model

    def process(self, sensor_image):
        return self.base.process(sensor_image)


TRANSFER_SP_CLASSES = {
    "TransferEncoderSensorProcessing": TransferEncoderSensorProcessing,
    "ViTRandomProjectionSensorProcessing": ViTRandomProjectionSensorProcessing,
    "MetaRidgeSensorProcessing": MetaRidgeSensorProcessing,
    "AdaBNSensorProcessing": AdaBNSensorProcessing,
}


def create_transfer_sp(spexp):
    """The SP of spexp: one of the transfer classes, or any class of sp_factory"""
    if spexp["class"] in TRANSFER_SP_CLASSES:
        return TRANSFER_SP_CLASSES[spexp["class"]](spexp)
    return sp_factory.create_sp(spexp)


def encode_frames(sp, images, masks, batch_size):
    """The latents of uint8 images [N, 3, H, W]: float32 [N, latent]. With masks
    (D3), the background is replaced by gray, as in the training of the SP."""
    device = Config().runtime["device"]
    latents = []
    for i in range(0, images.shape[0], batch_size):
        batch_masks = None if masks is None else masks[i:i + batch_size]
        x = masked_batch(images[i:i + batch_size], batch_masks)
        z = sp.process(x.to(device))
        latents.append(torch.from_numpy(np.atleast_2d(z)).float())
    return torch.cat(latents)
