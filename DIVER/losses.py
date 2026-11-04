"""Differentiable image losses for (batch, channels, height, width) tensors."""

import torch
import torch.nn.functional as F


def pixelwise_luminous_loss(image, target_intensity=0.5):
    """Average squared pixel error against the fixed exposure target."""
    return (image - target_intensity).square().mean()


def color_constancy_loss(image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected an RGB image tensor with shape (B, 3, H, W).")
    means = image.mean(dim=(2, 3))
    losses = []
    for p, q in ((0, 1), (1, 2), (2, 0)):
        losses.append(((means[:, p] - means[:, q]) /
                       (means[:, p] + means[:, q] + 1e-6)).square())
    return torch.stack(losses, dim=1).sum(dim=1).mean()


def adaptive_huber(prediction, target, delta):
    """Elementwise standard Huber penalty; delta stays fixed during training."""
    if delta <= 0:
        raise ValueError("The Huber threshold delta must be positive.")
    error = (prediction - target).abs()
    return torch.where(error <= delta, 0.5 * error.square(),
                       delta * (error - 0.5 * delta))


def sobel_edge_loss(prediction, target):
    """Mean per-pixel Sobel residual across every sample and color channel."""
    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError("Sobel inputs must have matching (B, C, H, W) shapes.")
    channels = prediction.shape[1]
    kernel_x = prediction.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    def gradients(image):
        # Replication preserves constant borders, including one-pixel inputs.
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        return (F.conv2d(padded, kernel_x, groups=channels),
                F.conv2d(padded, kernel_y, groups=channels))

    pred_x, pred_y = gradients(prediction)
    target_x, target_y = gradients(target)
    return ((pred_x - target_x).abs() + (pred_y - target_y).abs()).mean()
