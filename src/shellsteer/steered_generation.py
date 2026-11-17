"""Steered generation (paper Sec. 5): generate the events of the steering sample with the edit of Eq. 8, adding
alpha xi(t_k) to the two muon tokens at the output of block L and at every Euler step k, and keep the mass and the
four-momenta of each pair.

The vector comes from `steering_vectors.load_steering_vector`, the events from
`events.load_events(entry, "steering", n)`.
Without an edit, `generate_pairs` attaches no hook and generates the baseline. Batch i starts from the noise of
`sampler.batch_seed(seed, i)` for every alpha, so a steered pair has the noise of the same baseline pair.

The class of each pair comes from the baseline mass of the first seed (2041 in the paper). `events.select` keeps the
1,000,000 pairs with a class and the lowest priority once, and `save_kept` stores them; the same pairs count for every
seed and alpha. Each generation is one file:

    <run>/generated/seed_<seed>/alpha_<alpha:+g>.npz    from generated_path(run, seed, alpha); the baseline is alpha_+0
    <run>/generated/kept.npz                            from kept_path(run): the class of each kept pair

The mean ablation (paper Sec. 5.3) is the edit `Ablation`; its c_k come from `activation_means.mean_projection`.
It runs on the sample of the config's `ablation.sample` (the directions sample in the paper), so its baseline is
generated again for that sample and each seed:

    <run>/ablated/seed_<seed>/block_<L>.npz             from ablated_path(run, seed, block)
    <run>/ablated/seed_<seed>/baseline.npz              from ablated_path(run, seed, None)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from shellsteer.events import Events, batches, edited_tokens, pair_mass
from shellsteer.hooks import Ablate, Steer, attach, on_step
from shellsteer.networks import Network
from shellsteer.sampler import batch_seed, generate, prior

# Every generated file records these settings. The caller can add others, such as the network and dataset revisions.
METADATA_KEYS = ("alpha", "block", "target", "method", "seed", "steps", "edited steps", "tokens", "norm_restore",
                 "steering vector file")
# Every file of the ablation records these settings; "block" is None for its baseline.
ABLATION_METADATA_KEYS = ("block", "target", "method", "seed", "steps", "sample", "steering vector file",
                          "activation means file")


@dataclass(frozen=True)
class Edit:
    """The edit of Eq. 8: alpha xi^-(t_k) on the negative edited tokens and alpha xi^+(t_k) on the positive ones."""
    xi: torch.Tensor                                         # (steps, 2, width) steering vector, negative token first
    alpha: float
    block: int                                               # the edit acts on the output of this block
    tokens: Literal["pair", "minus", "plus", "all_of_type"]  # the edited tokens, see events.edited_tokens
    pair: Literal["muon", "electron", "tau"]                 # particle type of the pair, from the dataset entry
    steps: tuple[int, int] | None                            # edit at Euler steps first..last only; None: every step
    norm_restore: bool                                       # rescale each edited token to its norm before the edit


@dataclass(frozen=True)
class Ablation:
    """The mean ablation of Eq. 24 on the two tokens of the pair at the output of `block`, at every Euler step."""
    xi: torch.Tensor                                         # (steps, 2, width) unit steering vector, negative first
    mean: torch.Tensor                                       # (steps,) c_k of Eq. 25, from mean_projection
    block: int


@dataclass(frozen=True)
class GeneratedPairs:
    mass: np.ndarray          # (E,) float64 invariant mass of the pair in GeV
    four_momenta: np.ndarray  # (E, 2, 4) float32 four-momenta (px, py, pz, E) of the pair in GeV, negative first
    event: np.ndarray         # (E,) index of the event in the combined dataset


def check_edit(edit: Edit | Ablation, steps: int, width: int) -> None:
    """Stop unless xi has shape (steps, 2, width), the edited steps lie in 0..steps - 1, and an ablation has one c_k
    per step."""
    if edit.xi.dim() != 3 or edit.xi.shape[1] != 2:
        raise ValueError(f"xi has shape {tuple(edit.xi.shape)}, expected (steps, 2, width)")
    if edit.xi.shape[0] != steps:
        raise ValueError(f"xi has {edit.xi.shape[0]} steps, the sampler runs {steps}")
    if edit.xi.shape[2] != width:
        raise ValueError(f"xi has width {edit.xi.shape[2]}, the network has width {width}")
    if isinstance(edit, Ablation):
        if edit.mean.shape != (steps,):
            raise ValueError(f"mean has shape {tuple(edit.mean.shape)}, expected ({steps},), one c_k per step")
        return
    if edit.steps is not None and not 0 <= edit.steps[0] <= edit.steps[1] < steps:
        raise ValueError(f"edit steps: need 0 <= first <= last < {steps}, got {edit.steps}")


def generate_pairs(network: Network, events: Events, edit: Edit | Ablation | None, *, steps: int, batch_size: int,
                   seed: int, device: str | torch.device) -> GeneratedPairs:
    """Generate every event of `events` with `edit` (steering or ablation), or without edits for edit=None.

    Batch i holds events i x batch_size to (i + 1) x batch_size - 1 in file order and starts from the noise of
    batch_seed(seed, i). The mass is `events.pair_mass` of the float32 four-momenta, then converted to float64.
    """
    if edit is not None:
        check_edit(edit, steps, network.model.d_model)
        edit = replace(edit, xi=edit.xi.to(device))
        if isinstance(edit, Ablation):
            edit = replace(edit, mean=edit.mean.to(device))
    masses, four_momenta = [], []
    for index, batch, minus, plus in batches(events, batch_size, device):
        state = prior(network, batch, batch_seed(seed, index))
        if edit is None:
            p4 = generate(network, batch, state, steps)
        else:
            if isinstance(edit, Ablation):
                hook = Ablate(edit.xi, edit.mean, minus, plus)
            else:
                hook = Steer(edit.xi, edit.alpha, *edited_tokens(batch, minus, plus, edit.pair, edit.tokens),
                             steps=edit.steps, norm_restore=edit.norm_restore)
            with attach(network, edit.block, hook):
                p4 = generate(network, batch, state, steps, on_step=on_step(hook))
        rows = torch.arange(len(minus), device=p4.device)
        masses.append(pair_mass(p4, minus, plus).double().cpu().numpy())
        four_momenta.append(torch.stack([p4[rows, minus], p4[rows, plus]], dim=1).cpu().numpy())
    return GeneratedPairs(mass=np.concatenate(masses), four_momenta=np.concatenate(four_momenta), event=events.event)


def generated_path(run: str | Path, seed: int, alpha: float) -> Path:
    """<run>/generated/seed_<seed>/alpha_<alpha:+g>.npz, for example alpha_-1.5.npz; the baseline is alpha_+0.npz.

    Stops if alpha needs more than the 6 significant digits of the file name, because two such alphas would share
    one file.
    """
    if float(f"{alpha:g}") != alpha:
        raise ValueError(f"alpha: {alpha!r} is written as {alpha:g} in the file name; use at most 6 significant digits")
    return Path(run) / "generated" / f"seed_{seed}" / f"alpha_{alpha + 0.0:+g}.npz"   # + 0.0 turns -0.0 into +0


def ablated_path(run: str | Path, seed: int, block: int | None) -> Path:
    """<run>/ablated/seed_<seed>/block_<block>.npz, or baseline.npz for block=None: the baseline of the ablation."""
    name = "baseline" if block is None else f"block_{block}"
    return Path(run) / "ablated" / f"seed_{seed}" / f"{name}.npz"


def kept_path(run: str | Path) -> Path:
    """<run>/generated/kept.npz: the class of each kept pair, from the baseline of the first seed."""
    return Path(run) / "generated" / "kept.npz"


def save_generated(path: str | Path, generated: GeneratedPairs, metadata: dict,
                   required: tuple[str, ...] = METADATA_KEYS) -> None:
    """Write the mass, the four-momenta, the event index, and the settings that made them to one .npz.

    metadata must hold every key of `required` and can hold more: METADATA_KEYS for steering, ABLATION_METADATA_KEYS
    for the ablation. Creates the folder of `path` if needed.
    """
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"metadata has no {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{"mass": generated.mass, "four-momenta": generated.four_momenta, "event": generated.event,
                      "metadata": np.array(json.dumps(metadata, indent=1))})


def load_generated(path: str | Path) -> tuple[GeneratedPairs, dict]:
    with np.load(path) as data:
        generated = GeneratedPairs(mass=data["mass"], four_momenta=data["four-momenta"], event=data["event"])
        metadata = json.loads(str(data["metadata"]))
    return generated, metadata


def save_kept(path: str | Path, kept: np.ndarray, event: np.ndarray, metadata: dict) -> None:
    """Write the class of each pair from `events.select` ("" for pairs left out), the event index, and the settings.

    kept, event: (E,), in the order of the steering sample. Creates the folder of `path` if needed.
    """
    if len(kept) != len(event):
        raise ValueError(f"kept has {len(kept)} pairs, event has {len(event)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{"class": np.asarray(kept, dtype=str), "event": event,
                      "metadata": np.array(json.dumps(metadata, indent=1))})


def load_kept(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """The class of each pair (an object array, as `events.select` returns it), the event index, and the settings."""
    with np.load(path) as data:
        kept, event = data["class"].astype(object), data["event"]
        metadata = json.loads(str(data["metadata"]))
    return kept, event, metadata
