"""Generate scalar relative DAV2 depth arrays; visualizations are kept separately."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_load_and_logger import image_paths


def normalize_relative_depth(prediction):
    """Normalize a finite relative prediction while keeping valid pixels positive."""
    prediction = np.asarray(prediction, dtype=np.float32)
    finite = np.isfinite(prediction)
    if not finite.any():
        raise ValueError("The depth estimator returned no finite values.")
    values = prediction[finite]
    low, high = float(values.min()), float(values.max())
    result = np.zeros_like(prediction)
    if high - low <= 1e-8:
        result[finite] = 1.0
    else:
        result[finite] = np.maximum((values - low) / (high - low), 1e-6)
    return result


def depth_gen(input_folder, output_folder,
              model_name="depth-anything/Depth-Anything-V2-Small-hf",
              device=None, cache_dir=None, local_files_only=False):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "previews"
    preview.mkdir(exist_ok=True)
    paths = image_paths(input_folder)
    if not paths:
        raise ValueError(f"No input images in {input_folder}")
    options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    processor = AutoImageProcessor.from_pretrained(model_name, **options)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, **options).to(device).eval()
    saved = []
    for path in tqdm(paths, desc="Generating scalar relative depth"):
        with Image.open(path) as image:
            image = image.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                prediction = model(**inputs).predicted_depth
                prediction = F.interpolate(prediction.unsqueeze(1),
                                           size=(image.height, image.width),
                                           mode="bicubic", align_corners=False)[0, 0]
        relative = normalize_relative_depth(prediction.cpu().numpy())
        destination = output / (path.stem + ".npy")
        np.save(destination, relative)
        Image.fromarray(np.rint(relative * 255).astype(np.uint8)).save(
            preview / (path.stem + ".png"))
        saved.append(str(destination))
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache_dir")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    depth_gen(args.images, args.output, args.model, args.device,
              args.cache_dir, args.local_files_only)
