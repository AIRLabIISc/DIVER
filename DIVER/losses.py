import torch
import numpy as np
import cv2

def color_constancy_loss(image):
    """
    Computes the Color Constancy Loss for an input image.

    Args:
    - image: Tensor of shape (B, C, H, W) where B = batch size, 
             C = number of channels (3 for RGB), H = height, W = width.

    Returns:
    - loss: Color Constancy Loss (scalar).
    """
    # Ensure the image has 3 channels (RGB)
    if image.shape[1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Calculate mean for each color channel
    mu_R = torch.mean(image[:, 0, :, :], dim=(1, 2))  # Mean of Red channel
    mu_G = torch.mean(image[:, 1, :, :], dim=(1, 2))  # Mean of Green channel
    mu_B = torch.mean(image[:, 2, :, :], dim=(1, 2))  # Mean of Blue channel

    # Compute the loss components
    loss_RG = ((mu_R - mu_G) / (mu_R + mu_G + 1e-6))**2
    loss_GB = ((mu_G - mu_B) / (mu_G + mu_B + 1e-6))**2
    loss_BR = ((mu_B - mu_R) / (mu_B + mu_R + 1e-6))**2

    # Sum the losses and take the mean over the batch
    loss = torch.mean(loss_RG + loss_GB + loss_BR)
    
    return loss

def adaptive_huber(prediction, target, delta):
    abs_error = torch.abs(prediction - target)
    quadratic = torch.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    return quadratic


def sobel_edge_loss(I_pred, I_true):
            sobel_x = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
            sobel_y = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        
        
            # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
            I_pred = I_pred.squeeze().cpu().numpy()

            #    If the image is still 4D after squeezing, handle it
            if I_pred.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                I_pred = I_pred[0, 0]  # This will give you the height x width part, reducing to 2D

            #        If the image is 3D (like RGB), convert it to grayscale
            if I_pred.ndim == 3 and I_pred.shape[0] == 3:
                I_pred = np.mean(I_pred, axis=0)  # Convert to grayscale by averaging across channels

            # Ensure the result is 2D
            I_pred = np.squeeze(I_pred)

            #I_true = I_true.cpu().numpy()
            
            # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
            I_true = I_true.squeeze().cpu().numpy()

            #    If the image is still 4D after squeezing, handle it
            if I_true.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                I_true = I_true[0, 0]  # This will give you the height x width part, reducing to 2D

            #        If the image is 3D (like RGB), convert it to grayscale
            if I_true.ndim == 3 and I_true.shape[0] == 3:
                I_true = np.mean(I_true, axis=0)  # Convert to grayscale by averaging across channels

            # Ensure the result is 2D
            I_true = np.squeeze(I_true)

            if I_pred.ndim == 4:
                I_pred = I_pred[0]
                I_true = I_true[0]
        
            loss_total = 0

            for c in range(I_pred.shape[0]):
                sobel_x_pred = cv2.filter2D(I_pred[c], -1, sobel_x)
                sobel_x_true = cv2.filter2D(I_true[c], -1, sobel_x)
                sobel_y_pred = cv2.filter2D(I_pred[c], -1, sobel_y)
                sobel_y_true = cv2.filter2D(I_true[c], -1, sobel_y)
            
                loss_x = np.abs(sobel_x_pred - sobel_x_true)
                loss_y = np.abs(sobel_y_pred - sobel_y_true)
            
                loss_total += np.mean(loss_x + loss_y)
        
            return loss_total / I_pred.shape[0]