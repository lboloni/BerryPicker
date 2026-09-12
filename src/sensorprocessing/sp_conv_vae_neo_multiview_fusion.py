"""Runtime sensor processing for the Neo MultiView Fusion model."""

from exp_run_config import Config
import torch

from .conv_vae_neo_multiview_fusion import ConvVAENeoMultiViewFusionModel
from .sensor_processing import MultiViewEncoderSensorProcessing


class ConvVaeNeoMultiViewFusionSensorProcessing(MultiViewEncoderSensorProcessing):
    """Encode ordered camera views with the trained Neo Fusion model."""

    encoder_method = "encode_views"

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = exp["num_views"]
        self.cameras = list(exp["cameras"])
        self.fusion_type = exp.get("fusion_type", "concat_proj")
        self.enc = ConvVAENeoMultiViewFusionModel(exp).to(
            Config().runtime["device"]
        )
        self.load_encoder_checkpoint(
            required=True, label="Conv-VAE-Neo MultiView Fusion"
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
