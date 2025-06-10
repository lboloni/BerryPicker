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
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDN, self).__init__()
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Each output_dim needs num_gaussians for mu, sigma, and pi
        self.fc3 = nn.Linear(128, output_dim * num_gaussians * 3)

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

# Define the negative log-likelihood loss function for MDN
def mdn_loss(y, mu, sigma, pi):
    # y: (batch_size, output_dim) - true values
    # mu: (batch_size, output_dim, num_gaussians)
    # sigma: (batch_size, output_dim, num_gaussians)
    # pi: (batch_size, output_dim, num_gaussians)

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


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate some synthetic data (e.g., a noisy sine wave)
    np.random.seed(42)
    num_samples = 1000
    X = np.random.rand(num_samples, 1) * 10 - 5 # Input in range [-5, 5]
    y_true = np.sin(X) * 2 + np.random.randn(num_samples, 1) * 0.5 # Noisy sine wave

    # Introduce some multi-modality for demonstration
    y_true[X[:, 0] > 2] += 2 # Shift part of the sine wave up
    y_true[X[:, 0] < -2] -= 2 # Shift part of the sine wave down

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y_true)

    input_dim = 1
    output_dim = 1
    num_gaussians = 5 # Number of Gaussian components

    model = MDN(input_dim, output_dim, num_gaussians)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 2000
    batch_size = 64

    print("Starting training...")
    for epoch in range(num_epochs):
        # Mini-batch training
        permutation = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_tensor[indices], y_tensor[indices]

            optimizer.zero_grad()
            mu, sigma, pi = model(batch_X)
            loss = mdn_loss(batch_y, mu, sigma, pi)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    # 2. Make predictions and visualize
    # Create a range of input values for prediction
    X_test = torch.linspace(-5, 5, 500).unsqueeze(1)
    with torch.no_grad():
        mu_pred, sigma_pred, pi_pred = model(X_test)

    # Visualize the results
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y_true, s=10, alpha=0.3, label='True Data')

    # Plot the means
    for i in range(num_gaussians):
        plt.plot(X_test.numpy(), mu_pred[:, 0, i].numpy(), '--', alpha=0.7,
                 label=f'Mean {i+1}' if i == 0 else "")

    # Plot the predicted distribution's mean (weighted average of component means)
    # This is often not very informative for multimodal distributions, but good for understanding
    weighted_mean = torch.sum(pi_pred * mu_pred, dim=-1)
    plt.plot(X_test.numpy(), weighted_mean[:, 0].numpy(), 'k-', linewidth=2, label='Weighted Mean')

    # Plot the predictive distribution (e.g., by sampling or showing density contours)
    # A common way is to draw samples from the predicted mixture model
    num_samples_from_mdn = 2000
    y_samples = []
    for i in range(X_test.shape[0]):
        # For each X_test point, sample from its predicted mixture distribution
        mixture_idx = torch.multinomial(pi_pred[i, 0], num_samples=num_samples_from_mdn, replacement=True)
        # Collect the chosen mu and sigma for these samples
        chosen_mu = mu_pred[i, 0].gather(0, mixture_idx)
        chosen_sigma = sigma_pred[i, 0].gather(0, mixture_idx)
        # Sample from the corresponding Gaussian
        m_sample = torch.distributions.Normal(loc=chosen_mu, scale=chosen_sigma)
        y_samples.append(m_sample.sample())

    y_samples = torch.cat(y_samples).numpy()
    X_test_repeated = X_test.repeat_interleave(num_samples_from_mdn, dim=0).numpy()

    plt.hist2d(X_test_repeated.flatten(), y_samples.flatten(), bins=(50, 50), cmap='Blues', alpha=0.6, density=True)
    plt.colorbar(label='Density')


    plt.title('Mixed Density Network Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # You can also visualize individual Gaussian components for a specific input X
    # For example, let's pick a specific X value (e.g., X = 0)
    # x_example = torch.tensor([[0.0]])
    # with torch.no_grad():
    #     mu_ex, sigma_ex, pi_ex = model(x_example)
    #
    # print(f"\nFor X = {x_example.item()}:")
    # for i in range(num_gaussians):
    #     print(f"  Gaussian {i+1}:")
    #     print(f"    Mean (mu): {mu_ex[0, 0, i].item():.4f}")
    #     print(f"    Std Dev (sigma): {sigma_ex[0, 0, i].item():.4f}")
    #     print(f"    Mixing Coeff (pi): {pi_ex[0, 0, i].item():.4f}")
    #
    # # Plot the individual Gaussians for the example
    # plt.figure(figsize=(8, 5))
    # y_vals = np.linspace(-5, 5, 500)
    # for i in range(num_gaussians):
    #     pdf = (1 / (sigma_ex[0, 0, i] * np.sqrt(2 * np.pi))) * \
    #           np.exp(-0.5 * ((y_vals - mu_ex[0, 0, i]) / sigma_ex[0, 0, i])**2)
    #     plt.plot(y_vals, pdf * pi_ex[0, 0, i], label=f'Component {i+1} (weighted)')
    #
    # plt.title(f'Individual Gaussian Components for X = {x_example.item()}')
    # plt.xlabel('Y')
    # plt.ylabel('Probability Density (weighted)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()