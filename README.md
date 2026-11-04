# DIVER

DIVER combines IlluminateNet or spectral equalization, Adaptive Optical Correction Module (AOCM), relative depth estimation, and Hydro-OpticNet for underwater image restoration.

This package contains the restoration programs, a portable notebook, regression tests, the architecture diagram, workflow document, and the eleven original released checkpoints.

## Setup

Python 3.12 is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r DIVER/req.txt
```

Use a PyTorch/torchvision build appropriate to your CPU or CUDA installation. An existing Conda environment containing these dependencies can also be used.

The original runtime dependency pins are in `DIVER/req.txt`. The versions present during release verification are recorded separately in `tested_environment.json`; a fresh installation of the pinned environment was not performed.

## Complete pipeline: inference

Run commands from the repository root:

```bash
python DIVER/run_pipeline.py \
  --images /path/to/input_images \
  --output runs/my_inference \
  --mode infer \
  --illuminate-checkpoint DIVER/IlluminateNet_Weights/FishTrac \
  --hon-checkpoint DIVER/Diver_checkpoints/Diver_FishTrac_Final \
  --device cuda:0
```

Use `--device cpu` for CPU execution. The FishTrac checkpoints above are an example; choose the checkpoint corresponding to the intended dataset or your own trained checkpoint.

The pipeline performs per-image illumination routing, first-stage correction, AOCM, DAV2-Small depth estimation, and Hydro-OpticNet inference. It writes restored images, intermediate stages, scalar depth arrays, logs, and a JSON run summary.

Inference preserves each input image's original height and width, including mixed-size input folders. To explicitly resize inputs in the complete pipeline, supply both `--height` and `--width`. Standalone Hydro-OpticNet inference always preserves its input dimensions; its legacy size arguments are ignored.

DAV2-Small is downloaded on first use. To use an existing local model, supply:

```bash
--depth-model /path/to/Depth-Anything-V2-Small-hf --local-files-only
```

A custom download cache can be selected with `--cache-dir`.

## Training followed by checkpoint inference

```bash
python DIVER/run_pipeline.py \
  --images /path/to/training_images \
  --output runs/my_training \
  --mode train \
  --illuminate-iters 150 \
  --hon-iters 500 \
  --learning-rate 0.001 \
  --batch-size 10 \
  --height 256 --width 256 \
  --seed 1337 \
  --device cuda:0
```

Training mode initializes new models, optimizes them, saves new checkpoints, and reloads the final checkpoints for frozen inference. Released checkpoint files are not overwritten. Training images are saved after each batch's update, while the final inference pass uses the final checkpoint on every input; outputs from earlier batches can therefore differ.

When no training size is supplied, the complete runner uses 256×256. The individual training scripts retain their own command-line defaults. Iteration counts, batch size, learning rate, image size, seed, and device are configurable.

Use distinct training and evaluation folders for a held-out experiment. Running inference on the training inputs is a checkpoint/application check, not held-out evaluation.

## Standalone Hydro-OpticNet inference

```bash
python DIVER/hydroopticnet_inf.py \
  --images /path/to/aocm_images \
  --depth /path/to/scalar_depth_maps \
  --checkpoints DIVER/Diver_checkpoints/Diver_ST_Final \
  --output runs/hon_inference \
  --device cuda:0
```

The image keeps its native size. A mismatched depth map is resized to the image dimensions. Checkpoint arguments accept either a single `.pth` file or a directory; directory filenames are ordered naturally so checkpoint 10 follows checkpoint 9.

## Notebook

Open `DIVER/training+inference.ipynb` with the working directory set to the inner `DIVER/` folder. The notebook uses portable example paths such as `RAW/`. Change those paths to your input folder, select one first-stage branch, and choose training or inference.

Script cells use the notebook kernel's Python interpreter. Stored execution outputs have been removed. SEF and AOCM cells reload their modules so saved code changes take effect.

## Final processing configuration

| Component | Setting |
|---|---|
| Illumination routing | IlluminateNet if mean R < mean G/5 or mean R < mean B/5; otherwise SEF |
| SEF | Multiplicative channel-mean gain; alpha=0.3 |
| CEF | Original per-channel mean/median dual-intensity stretch and equal fusion |
| HSSF | OpenCV hue [70,100], saturation minimum 160, value minimum 180 |
| HSSF strength | 0.3, with HSV detection and CIELAB chroma suppression |
| Depth | Depth Anything V2 Small, fixed pretrained estimator |
| Luminous loss | Pixelwise mean squared error against 0.5 |
| Gray-world loss | On the IlluminateNet output, weight 0.25 |
| VeilNet loss | Mean Huber penalty on backscatter magnitude; fixed delta=0.5 |
| AttenNet loss | Equally weighted luminous, color-constancy, and differentiable Sobel losses |
| HON preprocessing | Same raw direct signal for training and inference; detached between separately optimized networks |

No experimental luminance-based CEF, isolated-component HSSF mask, gamma correction, unsharp masking, or extra red-scaling operation is enabled in this package.

RGB statistics support illumination routing. Training losses are logged. Image-quality scoring such as UIQM, UCIQE, PSNR, SSIM, or BRISQUE is not part of the restoration programs.

## Depth format

Depth generation writes normalized floating-point relative predictions as `.npy` arrays. These are relative model outputs, not calibrated metric range. Grayscale previews are stored separately and are not used as network inputs.

The loader pairs images and depth maps by filename stem. Matching `_depth` and `-depth` suffixes are also accepted. Floating-point arrays retain their numeric scale; integer grayscale relative maps are normalized by their encoding range. `--depth_16u` preserves 16-bit millimeter values and converts them to meters. Color depth visualizations are rejected.

## Checkpoints

The bundled checkpoint files are the original public release files. Their inclusion and compatibility checks do not mean they were retrained under the current processing settings. Separately trained local experiments are not included.

IlluminateNet checkpoints:

- `DIVER/IlluminateNet_Weights/SeeThru/model_checkpoint_st.pth`
- `DIVER/IlluminateNet_Weights/FishTrac/model_checkpoint_ft.pth`

Hydro-OpticNet checkpoints:

| Dataset | Directory under DIVER/Diver_checkpoints |
|---|---|
| SeaThru | Diver_ST_Final |
| OceanDark | Diver_OD_Final |
| USOD10K | Diver_USOD_Final |
| FishTrac | Diver_FishTrac_Final |
| U45 | Diver_U45_Final |
| UIEB | Diver_UIEB_Final |
| UFO-120 | Diver_UFO_Final |
| LSUI | Diver_LSUI_Final |
| EUVP | Diver_EUVP_Final |

Each directory contains one released `.pth` file. Legacy checkpoint keys and gzip-compressed IlluminateNet checkpoints are supported.

## Tests

```bash
python -m unittest discover -s tests -v
```

The twelve regression checks cover pixelwise loss behavior, gray-world input selection, trainable positive-veil and Sobel gradients, Sobel batch coverage, depth precision and pairing, native-resolution inference, and CPU prediction with all eleven released checkpoints.

The registered model parameter total is 54 + 24 + 13 = 91. Twelve legacy VeilNet coefficients remain registered for checkpoint compatibility and do not enter its forward pass.

## Repository layout

```text
DIVER/
  run_pipeline.py
  training+inference.ipynb
  illuminatenet_main.py
  illuminatenet_inf.py
  illuminatenet_model.py
  hydroopticnet_main.py
  hydroopticnet_inf.py
  veilnet_model.py
  attennet_model.py
  losses.py
  aocm_algorithm.py
  sef.py
  depth_generation.py
  data_load_and_logger.py
  transmap_generate.py
  runtime_utils.py
  rgb.py
  req.txt
  IlluminateNet_Weights/
  Diver_checkpoints/
tests/
README.md
LICENSE
DIVER_BlockDaigram.png
Processing Workflow - Training and Inference.docx
tested_environment.json
RELEASE_MANIFEST.json
```

Generated images, local datasets, training runs, caches, experimental variants, and backups are excluded from this upload folder.

## License

See `LICENSE` for Apache License 2.0.
