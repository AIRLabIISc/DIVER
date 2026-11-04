"""Frozen Hydro-OpticNet inference at each input image's original resolution."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_load_and_logger import DatasetLoad
from veilnet_model import VeilNet
from attennet_model import AttenNet
from runtime_utils import load_hon_checkpoint


def main(args):
    device = torch.device(args.device)
    veil_model = VeilNet().to(device).eval()
    atten_model = AttenNet().to(device).eval()
    load_hon_checkpoint(args.checkpoints, veil_model, atten_model, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Batch size one allows unrelated image sizes in the same input folder.
    # Align depth to the input image, never the image to a fixed depth/CLI size.
    dataset = DatasetLoad(args.images, args.depth, args.depth_16u,
                          args.mask_max_depth, None, None, device)
    with torch.inference_mode():
        for image, depth, names in DataLoader(dataset, batch_size=1, shuffle=False):
            direct, veil = veil_model(image, depth)
            gain, restored = atten_model(direct, depth)
            if not torch.isfinite(restored).all():
                raise ValueError("Hydro-OpticNet produced nonfinite output.")
            stem = Path(names[0][0]).stem
            save_image(restored.clamp(0, 1).cpu(), output / f"{stem}-corrected.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--height", type=int, default=None,
                        help="Legacy option, ignored: inference preserves input height.")
    parser.add_argument("--width", type=int, default=None,
                        help="Legacy option, ignored: inference preserves input width.")
    parser.add_argument("--depth_16u", action="store_true")
    parser.add_argument("--mask_max_depth", action="store_true")
    parser.add_argument("--init_lr", type=float, default=1e-2,
                        help="Accepted for compatibility; inference has no optimizer.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
