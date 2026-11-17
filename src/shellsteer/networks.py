"""Load a network from its entry in networks.yaml.

The checkpoint comes from Hugging Face. The model code comes from the ShellFlow package.
The checkpoint must name the dataloader config of the entry, if it names one.
The returned network uses the EMA weights, as in the paper, in eval mode and with no gradients on the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

import torch
from huggingface_hub import hf_hub_download
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

from shellsteer.config import NetworkEntry


@dataclass
class Network:
    model: torch.nn.Module          # the EMA weights; this is what generates and what we steer
    module: RiemannianDiTsModule    # the Lightning module, for the sampler settings stored with it


def load_network(entry: NetworkEntry, device: str | torch.device) -> Network:
    """Download the network of `entry` from Hugging Face (or read the local cache) and load it onto `device`."""
    path = hf_hub_download(entry.hf_repo, entry.file, revision=entry.revision)
    module = RiemannianDiTsModule.load_from_checkpoint(path, map_location=device)
    if module.ema_model is None:
        raise ValueError(f"{path} has no EMA weights")
    module.eval().requires_grad_(False)
    model = module.ema_model

    if model.d_model != entry.width:
        raise ValueError(f"{path}: networks.yaml says width {entry.width}, the checkpoint has {model.d_model}")
    stored = (module.hparams.get("model_config", {}).get("aux_head", {}).get("distribution_matching", {})
              .get("dataloader_config"))
    if stored is not None and PurePath(stored).name != entry.dataloader_config:
        raise ValueError(f"{path}: networks.yaml says dataloader_config {entry.dataloader_config}, the checkpoint "
                         f"was trained with {PurePath(stored).name}")
    return Network(model=model, module=module)
