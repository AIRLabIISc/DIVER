"""Image/depth loading with explicit filename pairing and preserved depth precision."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import kornia.morphology as morph

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class DatasetLoad(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth,
                 image_height, image_width, device):
        self.image_files = image_paths(image_path)
        if not self.image_files:
            raise ValueError(f"No input images in {image_path}")
        self.device = torch.device(device)
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        if (image_height is None) != (image_width is None):
            raise ValueError("Specify both image dimensions, or neither for native size.")
        self.size = None if image_height is None else (image_height, image_width)
        self.kernel = torch.ones(3, 3, device=self.device)
        candidates = {}
        for p in sorted(Path(depth_path).iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS | {".npy"}:
                continue
            existing = candidates.get(p.stem)
            if existing is None or p.suffix.lower() == ".npy":
                candidates[p.stem] = p
            elif existing.suffix.lower() != ".npy":
                raise ValueError(f"Ambiguous depth files for {p.stem}")
        self.depth_files = []
        for image in self.image_files:
            depth = next((candidates[key] for key in
                          (image.stem, image.stem + "_depth", image.stem + "-depth")
                          if key in candidates), None)
            if depth is None:
                raise ValueError(f"No matching depth/transmission map for {image.name}")
            self.depth_files.append(depth)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        path = self.image_files[index]
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        left = torch.from_numpy(rgb).permute(2, 0, 1)
        if self.size is not None and tuple(left.shape[-2:]) != self.size:
            left = F.interpolate(left.unsqueeze(0), size=self.size, mode="bilinear",
                                 align_corners=False)[0]
        left = left.to(self.device)
        depth_path = self.depth_files[index]
        if depth_path.suffix.lower() == ".npy":
            array = np.load(depth_path, allow_pickle=False)
        else:
            with Image.open(depth_path) as depth:
                if len(depth.getbands()) != 1:
                    raise ValueError(f"{depth_path} is a color visualization, not numeric depth.")
                array = np.array(depth)
        if array.ndim != 2:
            raise ValueError(f"Depth must be a 2D scalar array: {depth_path}")
        original_dtype = array.dtype
        array = array.astype(np.float32)
        if self.openni_depth:
            array /= 1000.0
        elif np.issubdtype(original_dtype, np.integer):
            # Relative grayscale maps use the full encoded integer range.
            maximum = 255.0 if original_dtype == np.uint8 else 65535.0
            array /= maximum
        array[~np.isfinite(array) | (array < 0)] = 0
        depth = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        target_size = tuple(left.shape[-2:])
        if tuple(depth.shape[-2:]) != target_size:
            depth = F.interpolate(depth, size=target_size, mode="bilinear",
                                  align_corners=False)
        depth = depth.to(self.device)
        if self.mask_max_depth:
            depth = torch.where(depth == 0, depth.max(), depth)
        low, high = torch.quantile(depth, 0.0001), torch.quantile(depth, 0.9999)
        depth = torch.where((depth < low) | (depth > high), 0.0, depth)
        depth = morph.closing(depth, self.kernel)[0]
        return left, depth, [path.name]


class LoggerToFile:
    def __init__(self, log_file_path):
        self.path = Path(log_file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, msg):
        with self.path.open("a") as stream:
            stream.write(str(msg) + "\n")
            
