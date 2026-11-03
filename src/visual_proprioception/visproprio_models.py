"""
visproprio_models.py

Regression models for visual proprioception. These models take as input 
a latent representation of the visual input, and create as the output of regression the positional information of the robot. 

They are parameterized by an experiment yaml, which defines the internal structure and the location where the model will be saved and restored from. 

For the time being, we assume that the positional information is a normalized form of RobotPosition.
"""
import torch.nn as nn



class VisProprio_SimpleMLPRegression(nn.Module):
    """A simple two-hidden layer MLP-based model for 
    visual proprioception regression """

    def __init__(self, exp):
        super(VisProprio_SimpleMLPRegression, self).__init__()

        self.input_size = exp["encoding_size"]
        self.hidden_size_1 = exp["regressor_hidden_size_1"]
        self.hidden_size_2 = exp["regressor_hidden_size_2"]
        self.output_size = exp["output_size"] # normally, 6

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.output_size)
        )

    def forward(self, x):
        return self.model(x)