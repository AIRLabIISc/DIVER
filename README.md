# DIVER

[![License](https://img.shields.io/badge/License-Research%20Only-informational.svg)](#license)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

# DIVER: Domain-Invariant Visual Enhancement and Restoration

Underwater images degrade due to wavelength-dependent attenuation, scattering, and non-uniform illumination. DIVER addresses these issues using a modular, physics-aware pipeline that generalizes across water types, depths, and lighting conditions.

DIVER is an **unsupervised, domain-invariant underwater image enhancement framework** that combines empirical enhancement with physics-guided modeling to handle diverse underwater conditions. DIVER provides **robust, physics-aware, and domain-invariant enhancement** for real-world underwater imaging and robotic perception. It is evaluated on **8 underwater datasets** (shallow, deep, turbid, low-light, and artificial illumination) and outperforms or matches state-of-the-art methods (WaterNet, UDNet, Phaseformer) across domains

## Architecture
- **IlluminateNet**: for adaptive luminance enhancement OR **Spectral Equalization Filter (SEF)**: normalizes spectral imbalance.
- **Adaptive Optical Correction Module (AOCM)**: Refines hue and contrast via channel-adaptive filtering.
- **Hydro-OpticNet (HON)**: Physics-constrained network that compensates for backscatter and wavelength-dependent attenuation.

All modules are trained **unsupervised** using a composite loss function.


<div align="center">
  <img src="DIVER_BlockDaigram.png" width="75%"/>
</div>

## Highlights
- **No preprocessing required**
- **Depth-aware enhancement** 
- Effective in all underwater lighting conditions

---

## Repository Structure (Quick Tour)
- `training+inference.ipynb` : **Training + Inference** notebook (main entrypoint)
-  `IlluminateNet_Checkpoints/` : pretrained weights for Seethru and Fishtrac datasets for preprocessing
- `Diver_Checkpoints/` : pretrained weights for all benchmark Datasets
- `req.txt` : python dependencies

---

## Requirements
- **Python 3.12** (recommended / tested)
- Works best on **Linux + NVIDIA GPU** (CUDA) for training
- Install dependencies from `req.txt`

---

## Setup & Installation

### 1) Clone the repository
```bash
https://github.com/AIRLabIISc/DIVER.git
```

### 2) Create and activate a virtual environment
```bash
python3.12 -m venv diver
source diver/bin/activate
```

### 3) Install requirements
```bash
pip install --upgrade pip
pip install -r req.txt
```

## Usage

To run DIVER use the provided notebook:
training+inference.ipynb




