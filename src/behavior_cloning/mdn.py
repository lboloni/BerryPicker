"""
mdn.py

As of 2026-06-10 this is an implementation of a mixture density network created by Gemini. 

TODOs:
* separate the baseline code here and an Experiment-MDN notebook for running experiments and understanding how and whether it is working.
* integrate it to the end of the LSTM behavior cloning model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class MDN(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.num_gaussians = exp["num_gaussians"]
        self.output_dim = exp["output_dim"] 

        self.fc1 = nn.Linear(exp["input_dim"], exp["hidden_size"])
        self.fc2 = nn.Linear(exp["hidden_size"], exp["hidden_size"])
        # Each output_dim needs num_gaussians for mu, sigma, and pi
        self.fc3 = nn.Linear(exp["hidden_size"], exp["output_dim"] * exp["num_gaussians"] * 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        params = self.fc3(x)

        # Reshape to separate mu, sigma, and pi for each output dimension
        # params shape: (batch_size, output_dim * num_gaussians * 3)
        # We want to get:
        # mu: (batch_size, output_dim, num_gaussians)
        # sigma: (batch_size, output_dim, num_gaussians)
        # pi: (batch_size, output_dim, num_gaussians)

        mu, sigma, pi = torch.split(params, self.output_dim * self.num_gaussians, dim=1)

        # Reshape each to (batch_size, output_dim, num_gaussians)
        mu = mu.view(-1, self.output_dim, self.num_gaussians)
        sigma = sigma.view(-1, self.output_dim, self.num_gaussians)
        pi = pi.view(-1, self.output_dim, self.num_gaussians)

        # Apply activation functions:
        # sigma must be positive, use exp
        sigma = torch.exp(sigma)
        # pi must be probabilities (sum to 1), use softmax over the gaussian dimension
        pi = torch.softmax(pi, dim=-1)

        return mu, sigma, pi

    def sample(self, num_samples, mu, sigma, pi):
        """Pull num samples from a specific mixture of Gaussians described by mu, sigma, pi. First choosing the Gaussian, then sampling from that."""
        mixture_idx = torch.multinomial(pi[i, 0], num_samples=num_samples, replacement=True)
        # Collect the chosen mu and sigma for these samples
        chosen_mu = mu[i, 0].gather(0, mixture_idx)
        chosen_sigma = sigma[i, 0].gather(0, mixture_idx)
        # Sample from the corresponding Gaussian
        m_sample = torch.distributions.Normal(loc=chosen_mu, scale=chosen_sigma)
        retval = m_sample.sample()
        return retval



def mdn_loss(y, mu, sigma, pi):
    """
      Calculate the loss for a y
      y: (batch_size, output_dim) - true values
      mu: (batch_size, output_dim, num_gaussians)
      sigma: (batch_size, output_dim, num_gaussians)
      pi: (batch_size, output_dim, num_gaussians)
    """

    # Expand y to match the dimensions of mu, sigma, pi for broadcasting
    y_expanded = y.unsqueeze(-1) # (batch_size, output_dim, 1)

    # Calculate the probability density for each Gaussian component
    # Gaussian PDF: (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((y - mu) / sigma)^2)
    # Using torch.distributions.Normal is more numerically stable and convenient
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob_components = m.log_prob(y_expanded) # (batch_size, output_dim, num_gaussians)

    # Add log of mixing coefficients
    log_weighted_prob = log_prob_components + torch.log(pi) # (batch_size, output_dim, num_gaussians)

    # Use logsumexp to sum probabilities in log-space to avoid underflow
    # This sums over the num_gaussians dimension
    log_mixture_prob = torch.logsumexp(log_weighted_prob, dim=-1) # (batch_size, output_dim)

    # The negative log-likelihood is the negative sum of these log probabilities
    # We sum over the output_dim and then mean over the batch_size
    return -torch.mean(torch.sum(log_mixture_prob, dim=-1))


