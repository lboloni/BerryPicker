"""Sensor processing using the grouped vector-quantized VAE."""

from exp_run_config import Config

from .vq_vae import VQVAE
from .sensor_processing import SingleViewEncoderSensorProcessing


class VQVAESensorProcessing(SingleViewEncoderSensorProcessing):
    """Encode one preprocessed RGB image batch as concatenated codebook embeddings."""

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = VQVAE(exp).to(Config().runtime["device"])
        self.load_encoder_checkpoint(required=True, label="VQ-VAE")
