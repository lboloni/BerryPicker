"""Inference wrappers for saved single-view and multi-view composites."""

from exp_run_config import Config
import torch

from .composite import CompositeModel, read_configuration
from .sensor_processing import (
    SingleViewEncoderSensorProcessing,
    MultiViewEncoderSensorProcessing,
)


class CompositeInference:
    """Use saved geometry and architecture, but the current run's model path."""

    def __init__(self, exp):
        config = read_configuration(exp)
        super().__init__(config)
        self.exp = exp
        self.enc = CompositeModel(config).to(Config().runtime["device"])
        self.temporal = self.enc.temporal
        self.load_encoder_checkpoint(required=True, label="composite")
        self.configuration = config

    def reset_context(self):
        self.context = None

    def process(self, sensor_readings, *, dt=None):
        if not self.temporal:
            return super().process(sensor_readings)
        if dt is None:
            dt = self.configuration["sample_interval"]
        self.enc.eval()
        with torch.no_grad():
            encoding, context = self.enc.advance(sensor_readings, self.context, dt=dt)
            result = torch.squeeze(encoding).cpu().numpy()
        self.context = context
        return result


class CompositeSensorProcessing(CompositeInference, SingleViewEncoderSensorProcessing):
    """Single-image composite with the existing NumPy inference interface."""


class CompositeMultiViewSensorProcessing(
    CompositeInference, MultiViewEncoderSensorProcessing
):
    """Composite receiving tensors in the saved camera order."""

    def __init__(self, exp):
        super().__init__(exp)
        self.num_views = self.configuration["num_views"]
        self.cameras = self.configuration["cameras"]

    def _warn_on_camera_order(self, cameras):
        if list(cameras) != self.cameras:
            raise ValueError(
                f"Camera order {list(cameras)} differs from trained order {self.cameras}"
            )
