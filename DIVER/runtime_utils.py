"""Checkpoint utilities shared by training and inference entry points."""

import gzip
from pathlib import Path
import re
import torch


def checkpoint_path(path):
    path = Path(path)
    if path.is_file():
        return path
    files = [p for p in path.glob("*.pth") if p.is_file()]
    if not files:
        raise ValueError(f"No .pth checkpoint in {path}")
    def natural_key(p):
        return [(0, int(part)) if part.isdigit() else (1, part)
                for part in re.split(r"(\d+)", p.name)]
    return sorted(files, key=natural_key)[-1]


def read_checkpoint(path, device):
    path = checkpoint_path(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as stream:
            return torch.load(stream, map_location=device, weights_only=True)
    return torch.load(path, map_location=device, weights_only=True)


def load_illuminate_checkpoint(path, model, device):
    checkpoint = read_checkpoint(path, device)
    state = checkpoint.get("rs_model_state_dict", checkpoint.get("bs_model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint has no IlluminateNet state dictionary.")
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_hon_checkpoint(path, veil_model, atten_model, device):
    checkpoint = read_checkpoint(path, device)
    state = checkpoint.get("veil_model_state_dict", checkpoint.get("bs_model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint has no VeilNet state dictionary.")
    aliases = {"beta_b_coef": "alp_b_coef", "beta_d_coef": "alp_d_coef",
               "backscatter_conv.weight": "veil_conv.weight"}
    state = {aliases.get(key, key): value for key, value in state.items()}
    veil_model.load_state_dict(state, strict=True)
    state = checkpoint.get("atten_net_model_state_dict", checkpoint.get("da_model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint has no AttenNet state dictionary.")
    atten_model.load_state_dict(state, strict=True)
    return checkpoint
