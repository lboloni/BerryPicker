"""Inference wrappers for saved single-view and multi-view composites."""

from exp_run_config import Config

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
        self.load_encoder_checkpoint(required=True, label="composite")
        self.configuration = config


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
