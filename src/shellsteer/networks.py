"""Load a network from its entry in networks.yaml.

The checkpoint comes from Hugging Face. The model code comes from the ShellFlow package.
The checkpoint must name the dataloader config of the entry, if it names one.
The returned network uses the EMA weights, as in the paper, in eval mode and with no gradients on the weights.

ShellGATr (backbone lgatr_slim) needs fp32 matmuls without TF32: its vector stream holds pair masses as small inner
products of large vectors, and TF32 rounds them away. PyTorch keeps this setting for the whole process, so
`load_network` does not change it. `precision(network)` turns TF32 off inside a `with` block and restores the
previous setting when the last such block of the process ends; `Workers.map` runs every batch inside it. For the
other backbones it changes nothing.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from pathlib import PurePath

import torch
from huggingface_hub import hf_hub_download
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule

from shellsteer.config import NetworkEntry


@dataclass
class Network:
    model: torch.nn.Module          # the EMA weights; this is what generates and what we steer
    module: RiemannianDiTsModule    # the Lightning module, for the sampler settings stored with it
    backbone: str | None            # the checkpoint's model_config backbone: None (attention blocks) or "lgatr_slim"

    @property
    def needs_fp32(self) -> bool:
        """True if the network must run with fp32 matmuls and no TF32 (ShellGATr)."""
        return self.backbone == "lgatr_slim"


_precision_lock = threading.Lock()
_precision_users = 0
_precision_saved: tuple[str, bool] | None = None


@contextmanager
def precision(network: Network) -> Iterator[None]:
    """Inside the `with` block, fp32 matmuls run at precision "highest" and cuDNN uses no TF32, if the network needs
    it (`Network.needs_fp32`), as in the paper's runs of ShellGATr; otherwise nothing changes.

    The setting is global to the process, so the blocks of all threads share it: the first block to enter sets it,
    and the last one to exit restores the values from before the first.
    """
    global _precision_users, _precision_saved
    if not network.needs_fp32:
        yield
        return
    with _precision_lock:
        if _precision_users == 0:
            _precision_saved = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
        _precision_users += 1
    try:
        yield
    finally:
        with _precision_lock:
            _precision_users -= 1
            if _precision_users == 0:
                torch.set_float32_matmul_precision(_precision_saved[0])
                torch.backends.cudnn.allow_tf32 = _precision_saved[1]
                _precision_saved = None


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
    return Network(model=model, module=module, backbone=module.hparams.get("model_config", {}).get("backbone"))
