import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import pipeline
import matplotlib.pyplot as plt

def depth_gen(input_folder,output_folder):
    # ------------------ Paths ------------------
    input_folder = input_folder
    output_folder = output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Collect image paths
    ref_paths = [os.path.join(input_folder, x) for x in os.listdir(input_folder)]

    # ------------------ Load Depth Model ------------------
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )

    # ------------------ Generate Depth Maps ------------------
    for file_path in tqdm(ref_paths, desc="Generating PLASMA depth maps"):

        filename = os.path.basename(file_path)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        try:
            # Load image
            img = Image.open(file_path).convert("RGB")

            # Run depth estimation
            depth_output = pipe(img)
            depth_map = depth_output["depth"]

            # Convert depth to numpy
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif torch.is_tensor(depth_map):
                depth_map = depth_map.squeeze().cpu().numpy()

            # Normalize depth to [0, 1]
            depth_normalized = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min() + 1e-8
            )

            # ------------------ Apply PLASMA colormap ------------------
            colormap = plt.get_cmap("plasma")
            depth_colored = colormap(depth_normalized)  # (H, W, 4) RGBA

            # Drop alpha channel and convert to uint8
            depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

            # Save colored depth map
            depth_img = Image.fromarray(depth_colored)
            depth_img.save(os.path.join(output_folder, filename))

            print(f"Saved PLASMA depth for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
