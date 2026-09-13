"""Notebook helpers for fixed VAE-GAN reconstructions and training histories."""

import matplotlib.pyplot as plt
import torch


def plot_images(rows, labels):
    count = min(len(row) for row in rows)
    figure, axes = plt.subplots(len(rows), count, squeeze=False, figsize=(3 * count, 3 * len(rows)))
    for row, label, row_axes in zip(rows, labels, axes):
        for image, axis in zip(row, row_axes):
            axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axis.set_xticks([])
            axis.set_yticks([])
        row_axes[0].set_ylabel(label)
    figure.tight_layout()
    return figure


def plot_history(history):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, keys in zip(axes, [("encoder", "generator", "discriminator"),
                               ("validation_mse", "validation_kl"),
                               ("validation_feature", "adversarial")]):
        for key in keys:
            rows = [row for row in history if key in row]
            if rows:
                axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], label=key)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    return figure


def plot_reconstructions(model, images, latent):
    model.eval()
    with torch.no_grad():
        reconstruction = model.decode(model.encode(images))
        generated = model.decode(latent)
    return plot_images([images, reconstruction, generated], ["Input", "Mean reconstruction", "Fixed prior"])
