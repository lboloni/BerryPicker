"""
sp_cnn.py

Sensor processing using pretrained CNN
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from .sensor_processing import SingleViewEncoderSensorProcessing
from .sp_helper import get_transform_to_sp

import torch
import torch.nn as nn
from torchvision import models

class VGG19ProprioTunedRegression(nn.Module):
    """Neural network used to create a latent embedding. Starts with a VGG19 neural network, without the classification head. The features are flattened, and fed into a regression MLP trained on visual proprioception. 
    
    When used for encoding, the processing happens only to an internal layer in the MLP.
    """

    def __init__(self, exp):
        super().__init__()
        self.latent_size = exp["latent_size"]
        self.output_size = exp["output_size"]
        # Pretrained vgg19
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = vgg19.features
        self.flatten = nn.Flatten()  # Flatten the output for the fully connected layer
        self.model = nn.Sequential(
            # The internal size seem to depend on the external size. 
            # the original with 7 * 7 corresponded to the 224 x 224 inputs
            #nn.Linear(512 * 7 * 7, hidden_size),
            nn.Linear(512 * 8 * 8, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.output_size)
        )
        # freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False        
            # Move the whole thing to the GPU if available
        self.feature_extractor.to(Config().runtime["device"])
        self.model.to(Config().runtime["device"])


    def forward(self, x):
        """Forward the input image through the complete network for the purposes of training using the proprioception. Return a vector of output_size which """
        features = self.feature_extractor(x)
        #print(features.shape)
        flatfeatures = self.flatten(features)
        #print(flatfeatures.shape)
        output = self.model(flatfeatures)
        # print(output.device)
        return output
    
    def encode(self, x):
        """Performs an encoding of the input image, by forwarding though the encoding and first three layers."""
        features = self.feature_extractor(x)
        flatfeatures = self.flatten(features)
        h1 = self.model[0](flatfeatures)
        h2 = self.model[1](h1)
        h3 = self.model[2](h2)
        return h3


    

class ResNetProprioTunedRegression(nn.Module):
    """Neural network used to create a latent embedding. Starts with a ResNet neural network, without the classification head. The features are flattened, and fed into a regression MLP trained on visual proprioception. 
    
    When used for encoding, the processing happens only to an internal layer in the MLP.
    """

    def __init__(self, exp):
        super(ResNetProprioTunedRegression, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Create the feature extractor by removing the last fully 
        # connected layer of the resnet (fc)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # freeze the parameters of the feature extractor
        if exp["freeze_feature_extractor"]:
            for param in self.resnet.parameters():
                param.requires_grad = False        

        self.flatten = nn.Flatten()  # Flatten the output for the fully
        # the reductor component
        self.reductor = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, exp["reductor_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["reductor_step_1"], exp["latent_size"])
        )

        # the proprioceptor auxiliary training component
        # not used in inference
        self.proprioceptor = nn.Sequential(
            nn.Linear(exp["latent_size"], exp["proprio_step_1"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_1"], exp["proprio_step_2"]),
            nn.ReLU(),
            nn.Linear(exp["proprio_step_2"], exp["output_size"])
        )

        # Move the whole thing to the GPU if available
        self.feature_extractor.to(Config().runtime["device"])
        self.reductor.to(Config().runtime["device"])
        self.proprioceptor.to(Config().runtime["device"])

    def forward(self, x):
        """Forward the input image through the complete network for the purposes of training using the proprioception. Return a vector of output_size which """
        features = self.feature_extractor(x)
        # print(f"Features shape {features.shape}")
        flatfeatures = self.flatten(features)
        latent = self.reductor(flatfeatures)
        output = self.proprioceptor(latent)
        # print(output.device)
        return output
    
    def encode(self, x):
        """Performs an encoding of the input image, by forwarding though the encoding and first three layers."""
        features = self.feature_extractor(x)
        flatfeatures = self.flatten(features)
        latent = self.reductor(flatfeatures)
        return latent

        
class _SingleViewCNNSensorProcessing(SingleViewEncoderSensorProcessing):
    """Common runtime wrapper for a proprioception-tuned CNN encoder."""

    encoder_class = None

    def __init__(self, exp):
        super().__init__(exp)
        self.enc = self.encoder_class(exp).to(Config().runtime["device"])
        self.load_encoder_checkpoint(required=True, label=self.encoder_class.__name__)


class ResNetProprioTunedSensorProcessing(_SingleViewCNNSensorProcessing):
    """Sensor processing using a pre-trained ResNet architecture."""

    encoder_class = ResNetProprioTunedRegression


class VGG19ProprioTunedSensorProcessing(_SingleViewCNNSensorProcessing):
    """Sensor processing using a pre-trained VGG19 architecture."""

    encoder_class = VGG19ProprioTunedRegression
