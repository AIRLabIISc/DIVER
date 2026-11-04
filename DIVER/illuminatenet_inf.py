"""Frozen IlluminateNet inference; supports CPU and compressed legacy checkpoints."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_load_and_logger import DatasetLoad
from illuminatenet_model import IlluminateNetModel
from runtime_utils import load_illuminate_checkpoint


def main(args):
    device = torch.device(args.device)
    model = IlluminateNetModel().to(device).eval()
    load_illuminate_checkpoint(args.checkpoints, model, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = DatasetLoad(args.images, args.depth, args.depth_16u,
                          args.mask_max_depth, args.height, args.width, device)
    with torch.inference_mode():
        for image, depth, names in DataLoader(dataset, batch_size=1, shuffle=False):
            _, _, restored, _, _ = model(image, depth)
            if not torch.isfinite(restored).all():
                raise ValueError("IlluminateNet produced nonfinite output.")
            stem = Path(names[0][0]).stem
            save_image(restored.clamp(0, 1).cpu(), output / f"{stem}-corrected.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--height", type=int, default=1242)
    parser.add_argument("--width", type=int, default=1952)
    parser.add_argument("--depth_16u", action="store_true")
    parser.add_argument("--mask_max_depth", action="store_true")
    parser.add_argument("--init_lr", type=float, default=1e-2,
                        help="Accepted for compatibility; inference has no optimizer.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
