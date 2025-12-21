import numpy as np
import cv2
import os

def stretch(image):
    return (image - image.min()) / (image.max() - image.min())

def get_gradient(image, win):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    gradient = cv2.Sobel(image_yuv[:, :, 0], cv2.CV_64F, 1, 1, ksize=3)
    grad_dilated = cv2.dilate(gradient, cv2.getStructuringElement(cv2.MORPH_RECT, (win, win)))
    grad_stretched = stretch(grad_dilated)
    return grad_stretched

def get_depth(image, win):
    grad_map = get_gradient(image, win)
    depth_map = 1 - grad_map
    return depth_map

def atmospheric_light(image, depth_map):
    flat_image = image.reshape(-1, 3)
    flat_depth = depth_map.flatten()
    sorted_indices = np.argsort(flat_depth)
    num_pixels = max(1, len(sorted_indices) // 1000)
    brightest_indices = sorted_indices[-num_pixels:]
    return flat_image[brightest_indices].mean(axis=0)

def calc_transmission(image, atmospheric_light, win):
    norm_image = np.abs(image - atmospheric_light) / np.maximum(atmospheric_light, 1 - atmospheric_light)
    transmission = np.max(norm_image, axis=2)
    transmission_uint8 = (transmission * 255).astype(np.uint8)
    transmission_blurred = cv2.medianBlur(transmission_uint8, win)
    return np.clip(transmission_blurred / 255.0, 0, 1)

# Main function
def process_images(input_dir, output_dir, win=15, t0=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            filepath = os.path.join(input_dir, filename)
            image = cv2.imread(filepath).astype(np.float32) / 255.0
            
            # Calculate depth map
            depth_map = get_depth(image, win)
            
            # Calculate atmospheric light
            atm_light = atmospheric_light(image, depth_map)
            
            # Calculate transmission map
            trans_map = calc_transmission(image, atm_light, win)
            cv2.imwrite(os.path.join(output_dir, filename), (trans_map * 255).astype(np.uint8))