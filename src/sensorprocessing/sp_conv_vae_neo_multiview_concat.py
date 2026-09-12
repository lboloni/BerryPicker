"""Runtime sensor processing for the MultiView Concat Conv-VAE-Neo."""

from exp_run_config import Config
import torch

from .conv_vae_neo_multiview_concat import ConvVAENeoMultiViewConcatModel
from .sensor_processing import MultiViewEncoderSensorProcessing


class ConvVaeNeoMultiViewConcatSensorProcessing(MultiViewEncoderSensorProcessing):
    """Encode a fixed ordered camera set through a joint Concat VAE."""

    encoder_method = "encode_views"

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = exp["num_views"]
        self.cameras = list(exp["cameras"])
        self.enc = ConvVAENeoMultiViewConcatModel(exp).to(
            Config().runtime["device"]
        )
        self.load_encoder_checkpoint(
            required=True, label="Conv-VAE-Neo MultiView Concat"
        )

    def _warn_on_camera_order(self, cameras):
        if list(cameras) != self.cameras:
            raise ValueError(
                f"Camera order {list(cameras)} differs from trained order "
                f"{self.cameras}"
            )

    def process(self, sensor_readings):
        if not isinstance(sensor_readings, (list, tuple)):
            raise TypeError("sensor_readings must be an ordered view sequence")
        if not all(isinstance(view, torch.Tensor) for view in sensor_readings):
            raise TypeError("Every sensor view must be a torch.Tensor")
        device = Config().runtime["device"]
        return super().process([view.to(device) for view in sensor_readings])
