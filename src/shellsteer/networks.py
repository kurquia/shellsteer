"""Load a network from its entry in networks.yaml.

The checkpoint comes from Hugging Face, or from a local file. The model code comes from the ShellFlow package.
The returned network uses the EMA weights, as in the paper, in eval mode and with no gradients on the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

from shellsteer.config import NetworkEntry


@dataclass
class Network:
    model: torch.nn.Module          # the EMA weights; this is what generates and what we steer
    module: RiemannianDiTsModule    # the Lightning module, for the sampler settings stored with it
    checkpoint: Path                # the file the weights came from


def load_network(entry: NetworkEntry, device: str | torch.device, checkpoint: str | Path | None = None) -> Network:
    """Load the network of `entry` onto `device`. `checkpoint` replaces the Hugging Face download with a local file."""
    path = Path(checkpoint) if checkpoint is not None else Path(hf_hub_download(entry.hf_repo, entry.file))
    module = RiemannianDiTsModule.load_from_checkpoint(path, map_location=device)
    if module.ema_model is None:
        raise ValueError(f"{path} has no EMA weights")
    module.eval().requires_grad_(False)
    model = module.ema_model

    if model.d_model != entry.width:
        raise ValueError(f"{path}: networks.yaml says width {entry.width}, the checkpoint has {model.d_model}")
    return Network(model=model, module=module, checkpoint=path)
