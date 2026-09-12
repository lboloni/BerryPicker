"""Sensor processing using BerryPicker's internal convolutional VAE."""

from exp_run_config import Config

from .conv_vae_neo import ConvVAENeo
from .sensor_processing import SingleViewEncoderSensorProcessing


class ConvVaeNeoSensorProcessing(SingleViewEncoderSensorProcessing):
    """Encode one preprocessed RGB image batch as the VAE latent mean."""

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = ConvVAENeo(exp).to(Config().runtime["device"])
        self.load_encoder_checkpoint(required=True, label="Conv-VAE-Neo")
