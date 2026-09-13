"""VAE-GAN sensor processing uses the exported Neo VAE, never the discriminator."""

from exp_run_config import Config
from .conv_vae_neo import ConvVAENeo
from .sensor_processing import SingleViewEncoderSensorProcessing


class VAEGANSensorProcessing(SingleViewEncoderSensorProcessing):
    def __init__(self, exp):
        super().__init__(exp)
        self.enc = ConvVAENeo(exp).to(Config().runtime["device"])
        self.load_encoder_checkpoint(required=True, label="VAE-GAN VAE export")
