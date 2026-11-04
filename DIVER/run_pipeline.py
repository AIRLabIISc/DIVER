"""Run routing, first-stage correction, AOCM, scalar depth, and Hydro-OpticNet."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from aocm_algorithm import AOCM
from data_load_and_logger import image_paths
from depth_generation import depth_gen
from sef import spectral_equalization_filter
from transmap_generate import process_images
from runtime_utils import checkpoint_path
import illuminatenet_main
import illuminatenet_inf
import hydroopticnet_main
import hydroopticnet_inf


def execute_stage(name, function, options, root):
    log = root / f"{name}.log"
    print(f"Running {name}; log: {log}", flush=True)
    with log.open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        function(SimpleNamespace(**options))


def run(args):
    # Training batches need a shared shape; frozen inference can keep native sizes.
    if args.mode == "train" and args.height is None and args.width is None:
        args.height = args.width = 256
    root = Path(args.output).resolve()
    if (root / "run_summary.json").exists():
        raise ValueError("Choose a new output folder; this run already has a summary.")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    source = Path(__file__).parent
    directories = {name: root / name for name in
                   ("raw", "low_lit", "transmission", "illuminate_train",
                    "illuminate_inference", "first_stage", "aocm", "depth",
                    "hon_train", "final")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    inputs = image_paths(args.images)
    if not inputs:
        raise ValueError(f"No images in {args.images}")
    routes = []
    seen = set()
    for path in inputs:
        if path.stem in seen:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        seen.add(path.stem)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if args.height is not None:
                image = image.resize((args.width, args.height), Image.Resampling.LANCZOS)
            array = np.asarray(image)
            r, g, b = array.mean(axis=(0, 1))
            low = r < g / args.routing_k or r < b / args.routing_k
            destination = directories["raw"] / (path.stem + ".png")
            image.save(destination)
            if low:
                image.save(directories["low_lit"] / destination.name)
            else:
                result = spectral_equalization_filter(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(directories["first_stage"] / destination.name), result)
        routes.append({"source": str(path.resolve()), "name": destination.name,
                       "branch": "IlluminateNet" if low else "SEF",
                       "height": int(array.shape[0]), "width": int(array.shape[1]),
                       "mean_rgb": [float(r), float(g), float(b)]})
    low_routes = [entry for entry in routes if entry["branch"] == "IlluminateNet"]
    common = dict(depth_16u=False, mask_max_depth=False, height=args.height,
                  width=args.width, device=args.device)
    checks = {}
    if low_routes:
        process_images(str(directories["low_lit"]), str(directories["transmission"]))
        illuminate_checkpoint = args.illuminate_checkpoint
        if args.mode == "train":
            illuminate_checkpoint = str(root / "illuminate_checkpoints")
            execute_stage("illuminate_training", illuminatenet_main.main,
                          dict(common, images=str(directories["low_lit"]),
                               depth=str(directories["transmission"]),
                               output=str(directories["illuminate_train"]),
                               checkpoints=illuminate_checkpoint, seed=args.seed,
                               batch_size=args.batch_size, init_iters=args.illuminate_iters,
                               iters=args.illuminate_iters, init_lr=args.learning_rate,
                               save_intermediates=False), root)
        execute_stage("illuminate_inference", illuminatenet_inf.main,
                      dict(common, images=str(directories["low_lit"]),
                           depth=str(directories["transmission"]),
                           output=str(directories["illuminate_inference"]),
                           checkpoints=illuminate_checkpoint), root)
        for entry in low_routes:
            stem = Path(entry["name"]).stem
            restored = directories["illuminate_inference"] / f"{stem}-corrected.png"
            if args.mode == "train":
                trained = directories["illuminate_train"] / f"{stem}-enhanced.png"
                difference = np.abs(np.asarray(Image.open(restored), dtype=int) -
                                    np.asarray(Image.open(trained), dtype=int))
                checks[f"{stem}_illuminate_reload_max_pixel_error"] = int(difference.max())
                if difference.max() > 1:
                    raise AssertionError("IlluminateNet checkpoint output differs from training output.")
            with Image.open(restored) as image:
                image.save(directories["first_stage"] / entry["name"])
    for entry in routes:
        image = cv2.imread(str(directories["first_stage"] / entry["name"]))
        corrected = AOCM(image, return_all=False, hssf_strength=args.hssf_strength)
        cv2.imwrite(str(directories["aocm"] / entry["name"]), corrected)
    print("Running DAV2 scalar depth generation.", flush=True)
    depth_gen(directories["aocm"], directories["depth"], args.depth_model,
              args.device, args.cache_dir, args.local_files_only)
    hon_checkpoint = args.hon_checkpoint
    if args.mode == "train":
        hon_checkpoint = str(root / "hon_checkpoints")
        execute_stage("hon_training", hydroopticnet_main.main,
                      dict(common, images=str(directories["aocm"]),
                           depth=str(directories["depth"]), output=str(directories["hon_train"]),
                           checkpoints=hon_checkpoint, seed=args.seed, batch_size=args.batch_size,
                           init_iters=args.hon_iters, iters=args.hon_iters,
                           init_lr=args.learning_rate, save_intermediates=True), root)
    execute_stage("hon_inference", hydroopticnet_inf.main,
                  dict(common, images=str(directories["aocm"]),
                       depth=str(directories["depth"]), output=str(directories["final"]),
                       checkpoints=hon_checkpoint), root)
    outputs = []
    for entry in routes:
        stem = Path(entry["name"]).stem
        path = directories["final"] / f"{stem}-corrected.png"
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.shape != (entry["height"], entry["width"], 3) or not np.isfinite(array).all():
            raise AssertionError(f"Invalid restored image: {path}")
        if args.mode == "train":
            trained = directories["hon_train"] / path.name
            difference = np.abs(array.astype(int) - np.asarray(Image.open(trained), dtype=int))
            checks[f"{stem}_hon_reload_max_pixel_error"] = int(difference.max())
            if difference.max() > 1:
                raise AssertionError("HON checkpoint output differs from training output.")
        outputs.append({"path": str(path), "shape": list(array.shape)})
    summary = {"settings": vars(args), "torch_version": torch.__version__,
               "routes": routes, "checks": checks, "outputs": outputs,
               "hon_checkpoint": str(checkpoint_path(hon_checkpoint)),
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source.glob("*.py"))},
               "scope": "Execution/gradient/checkpoint consistency check; not a benchmark rerun.",
               "veil_objective": "Full-magnitude Huber regularization, delta=0.5; alone favors zero backscatter."}
    (root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Completed {len(outputs)} images. Summary: {root / 'run_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    source = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("train", "infer"), default="infer")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--illuminate-checkpoint", default=str(source / "IlluminateNet_Weights/SeeThru"))
    parser.add_argument("--hon-checkpoint", default=str(source / "Diver_checkpoints/Diver_ST_Final"))
    parser.add_argument("--height", type=int, default=None,
                        help="Optional input resize; inference defaults to native size.")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional input resize; training defaults to 256x256.")
    parser.add_argument("--illuminate-iters", type=int, default=150)
    parser.add_argument("--hon-iters", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--routing-k", type=float, default=5.0)
    parser.add_argument("--hssf-strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("Specify both --height and --width, or neither.")
    dimensions = [value for value in (args.height, args.width) if value is not None]
    if min(dimensions + [args.threads, args.batch_size,
                         args.illuminate_iters, args.hon_iters]) < 1:
        parser.error("Sizes, threads, batch size, and iteration counts must be positive.")
    if args.routing_k <= 0 or not 0 <= args.hssf_strength <= 1:
        parser.error("routing-k must be positive and hssf-strength must lie in [0,1].")
    run(args)
