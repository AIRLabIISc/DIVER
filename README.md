# DIVER

[![License](https://img.shields.io/badge/License-Research%20Only-informational.svg)](#license)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

DIVER, an unsupervised domain-invariant restoration architecture that unifies empirical enhancement and physics-guided modeling . 

Underwater imaging is severely affected by wavelength-dependent attenuation, scattering, and non-uniform illumination, all of which vary considerably across water types and depths. To address these challenges, we introduce an unsupervised Domain-Invariant Visual Enhancement and Restoration (DIVER) framework that integrates empirical corrections with physics-guided modeling to achieve robust image enhancement. The architecture begins with either IlluminateNet, which performs adaptive luminance enhancement, or the Spectral Equalization Filter, which normalizes spectral distributions. This is followed by the Adaptive Optical Correction Module (AOCM), which refines hue and contrast through channel-adaptive filtering, and the Hydro-OpticNet (HON) block, which leverages physics-constrained learning to compensate for backscatter and wavelength-dependent attenuation. Tunable parameters in IlluminateNet and Hydro-OpticNet are optimized via unsupervised training with a composite loss function.
We evaluate DIVER across eight diverse datasets spanning shallow, deep, and highly turbid environments, including naturally low-lit and artificially illuminated scenarios, using both reference and non-reference metrics. While state-of-the-art methods such as WaterNet, UDNet, and Phaseformer perform reasonably well in shallow-water conditions, they degrade significantly in deeper, unevenly illuminated, or artificially lit environments. In contrast, DIVER consistently achieves best or near-best results across all datasets, underscoring its domain-invariant capability. Notably, on the low-lit SeaThru dataset, where color-palette references enable quantitative evaluation of color restoration accuracy, DIVER achieves at least a 4.9% improvement over existing methods. Beyond visual quality metrics, DIVER also enhances downstream robotic perception tasks, improving keypoint repeatability and feature matching performance using ORB descriptors. Collectively, these findings demonstrate that DIVER is both robust and domain-invariant across a wide range of underwater imaging conditions.



<div align="center">
  <img src="DIVER_BlockDaigram_final.png" width="75%"/>
</div>

## Highlights
- **No preprocessing required**
- **Depth-aware enhancement** using transmission maps
- **Lightweight CNN** architecture
- Effective in **deep-sea** and **low-light** conditions

---

## Repository Structure (Quick Tour)
- `train+inf.ipynb` : **Training + Inference** notebook (main entrypoint)
- `weights/` : pretrained weights FOR SeeThru and OceanDark Datasets
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
git clone https://github.com/AIRLabIISc/REVERSEA.git
cd Reversea
```

### 2) Create and activate a virtual environment
```bash
python3.12 -m venv reversea
source reversea/bin/activate
```

### 3) Install requirements
```bash
pip install --upgrade pip
pip install -r req.txt
```

## Running

The easiest way to run ReverSea is using the provided notebook:
train+inf.ipynb


  organization={IEEE}
}
