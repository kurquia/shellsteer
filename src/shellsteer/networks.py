"""Load a network by name from networks.yaml.

The checkpoint comes from Hugging Face, or from a local file. The model code comes from the ShellFlow package.
The returned network uses the EMA weights, as in the paper, in eval mode and with no gradients on the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule


class NetworkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hf_repo: str
    file: str
    width: int


@dataclass
class Network:
    name: str
    model: torch.nn.Module          # the EMA weights; this is what generates and what we steer
    module: RiemannianDiTsModule    # the Lightning module, for the sampler settings stored with it
    checkpoint: Path


def read_networks(path: str | Path) -> dict[str, NetworkEntry]:
    """Read networks.yaml into one entry per network name."""
    raw = yaml.safe_load(Path(path).read_text())
    return {name: NetworkEntry.model_validate(entry) for name, entry in raw.items()}


def load_network(name: str, networks_file: str | Path, *, checkpoint: str | Path | None = None,
                 device: str | torch.device = "cpu") -> Network:
    """Load network `name`. `checkpoint` replaces the Hugging Face download with a local file."""
    entries = read_networks(networks_file)
    if name not in entries:
        raise ValueError(f"network {name!r} is not in {networks_file}; it has {sorted(entries)}")
    entry = entries[name]

    path = Path(checkpoint) if checkpoint is not None else Path(hf_hub_download(entry.hf_repo, entry.file))
    module = RiemannianDiTsModule.load_from_checkpoint(path, map_location=device)
    if module.ema_model is None:
        raise ValueError(f"{path} has no EMA weights")
    module.eval().requires_grad_(False)
    model = module.ema_model

    if model.d_model != entry.width:
        raise ValueError(f"{name}: networks.yaml says width {entry.width}, the checkpoint has {model.d_model}")
    return Network(name=name, model=model, module=module, checkpoint=path)
