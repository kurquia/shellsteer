"""Steered generation (paper Sec. 5): generate the events of the steering sample with the edit of Eq. 8, adding
alpha xi(t_k) to the two muon tokens at the output of block L and at every Euler step k, and keep the mass and the
four-momenta of each pair.

The vector comes from `steering_vectors.load_steering_vector`, the events from
`events.load_events(entry, "steering", n, network)`.
Without an edit, `generate_pairs` attaches no hook and generates the baseline. Batch i starts from the noise of
`sampler.batch_seed(seed, i)` for every alpha, so a steered pair has the noise of the same baseline pair.

The class of each pair comes from the baseline mass of the first seed (2041 in the paper). `events.select` keeps the
1,000,000 pairs with a class and the lowest priority once, and `save_kept` stores them; the same pairs count for every
seed and alpha. Each generation is one file:

    <run>/generated/seed_<seed>/alpha_<alpha:+g>.npz    from generated_path(run, seed, alpha); the baseline is alpha_+0
    <run>/generated/kept.npz                            from kept_path(run): the class of each kept pair

A steering sample of every final state (paper Fig. 4) has no pairs and no class selection. `generate_events` edits
every particle of the edited type (tokens all_of_type) and keeps the four-momenta, types, charges, and mask of every
particle of every event, and the decoder's mass of each jet, large-R jet, and hadronic tau (Fig. 14).
`save_generated_events` writes them to the same generated_path, one file per seed and alpha; there is no kept.npz.

The mean ablation (paper Sec. 5.3) is the edit `Ablation`; its c_k come from `activation_means.mean_projection`.
It runs on the sample of the config's `ablation.sample` (the directions sample in the paper), so its baseline is
generated again for that sample and each seed:

    <run>/ablated/seed_<seed>/block_<L>.npz             from ablated_path(run, seed, block)
    <run>/ablated/seed_<seed>/baseline.npz              from ablated_path(run, seed, None)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from shellsteer.devices import Workers
from shellsteer.events import (Events, FinalStates, batch_at, batch_count, edited_tokens, final_state_batch_at,
                               pair_mass, tokens_of_type)
from shellsteer.hooks import Ablate, Steer, attach, on_step
from shellsteer.networks import Network
from shellsteer.sampler import NanCounts, OnNan, batch_seed, generate, prior

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
    pair: Literal["muon", "electron", "tau"]                 # the edited type: the pair type of the directions sample
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
    nans: NanCounts | None = None   # the NaN counts of the generation; None for pairs read from a file, whose
                                    # metadata holds them under "NaN counts"


@dataclass(frozen=True)
class GeneratedEvents:
    """Every particle of every event of a sample without pairs (paper Fig. 4), in file order."""
    four_momenta: np.ndarray     # (E, N, 4) float32 four-momenta (px, py, pz, E) in GeV, zero on padding
    particle_type: np.ndarray    # (E, N) int16 ShellFlow type code of each slot, as in the dataset
    particle_charge: np.ndarray  # (E, N) int16 charge + 1 of each slot
    particle_mask: np.ndarray    # (E, N) bool, True on real particles
    event: np.ndarray            # (E,) index of the event in the combined dataset
    nans: NanCounts | None = None   # as in GeneratedPairs, without the pair-mass column
    mass: np.ndarray | None = None  # (E, N) float32 sampler.decoded_mass in GeV, NaN on massless types and padding;
                                    # None for a file written before shellsteer stored it


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


def generate_pairs(workers: Workers, events: Events, edit: Edit | Ablation | None, *, steps: int, batch_size: int,
                   seed: int, on_nan: OnNan = "set_to_zero") -> GeneratedPairs:
    """Generate every event of `events` with `edit` (steering or ablation), or without edits for edit=None.

    Batch i holds events i x batch_size to (i + 1) x batch_size - 1 in file order and starts from the noise of
    batch_seed(seed, i); `workers.map` runs it on one of its devices and returns the batches in order. The mass is
    `events.pair_mass` of the float32 four-momenta, then converted to float64.
    on_nan: what `sampler.generate` does with a NaN; the counts are in the `nans` of the result.
    """
    if edit is not None:
        check_edit(edit, steps, workers.width)

    def one_batch(index: int, network: Network, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch, minus, plus = batch_at(events, index, batch_size, device)
        state = prior(network, batch, batch_seed(seed, index))
        settings = dict(on_nan=on_nan, label=f"batch {index} of seed {seed}")
        if edit is None:
            generated = generate(network, batch, state, steps, **settings)
        else:
            if isinstance(edit, Ablation):
                hook = Ablate(edit.xi.to(device), edit.mean.to(device), minus, plus)
            else:
                hook = Steer(edit.xi.to(device), edit.alpha, *edited_tokens(batch, minus, plus, edit.pair, edit.tokens),
                             steps=edit.steps, norm_restore=edit.norm_restore)
            with attach(network, edit.block, hook):
                generated = generate(network, batch, state, steps, on_step=on_step(hook), **settings)
        p4 = generated.four_momenta
        rows = torch.arange(len(minus), device=p4.device)
        mass = pair_mass(p4, minus, plus)
        return (mass.double().cpu().numpy(), torch.stack([p4[rows, minus], p4[rows, plus]], dim=1).cpu().numpy(),
                NanCounts.of_batch(generated, mass))

    masses, four_momenta, nans = [], [], []
    for mass, pair, counts in workers.map(batch_count(events, batch_size), one_batch):
        masses.append(mass)
        four_momenta.append(pair)
        nans.append(counts)
    return GeneratedPairs(mass=np.concatenate(masses), four_momenta=np.concatenate(four_momenta), event=events.event,
                          nans=NanCounts(on_nan, events.event, np.concatenate(nans)))


def generate_events(workers: Workers, events: FinalStates, edit: Edit | None, *, steps: int, batch_size: int,
                    seed: int, on_nan: OnNan = "set_to_zero") -> GeneratedEvents:
    """Generate every event of a sample without pairs with `edit`, or without edits for edit=None (paper Fig. 4).

    The edit adds alpha xi^- to every negative and alpha xi^+ to every positive particle of type `edit.pair`, so it
    needs tokens all_of_type; an event without such a particle is generated as without the edit. Batch i holds events
    i x batch_size to (i + 1) x batch_size - 1 in file order and starts from the noise of batch_seed(seed, i), as in
    `generate_pairs`; `workers.map` runs it on one of its devices and returns the batches in order, so the four-momenta,
    the masses, and the NaN counts per event are in file order and equal those of one device of the same type.
    """
    if edit is not None:
        if not isinstance(edit, Edit) or edit.tokens != "all_of_type":
            raise ValueError(f"edit: events without pairs take only a steering edit with tokens all_of_type, got "
                             f"{getattr(edit, 'tokens', type(edit).__name__)!r}")
        check_edit(edit, steps, workers.width)

    def one_batch(index: int, network: Network, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = final_state_batch_at(events, index, batch_size, device)
        state = prior(network, batch, batch_seed(seed, index))
        settings = dict(on_nan=on_nan, label=f"batch {index} of seed {seed}")
        if edit is None:
            generated = generate(network, batch, state, steps, **settings)
        else:
            hook = Steer(edit.xi.to(device), edit.alpha, *tokens_of_type(batch, edit.pair), steps=edit.steps,
                         norm_restore=edit.norm_restore)
            with attach(network, edit.block, hook):
                generated = generate(network, batch, state, steps, on_step=on_step(hook), **settings)
        return (generated.four_momenta.cpu().numpy(), generated.mass.float().cpu().numpy(),
                NanCounts.of_batch(generated, None))

    four_momenta, masses, nans = [], [], []
    for batch_four_momenta, mass, counts in workers.map(batch_count(events, batch_size), one_batch):
        four_momenta.append(batch_four_momenta)
        masses.append(mass)
        nans.append(counts)
    return GeneratedEvents(four_momenta=np.concatenate(four_momenta), mass=np.concatenate(masses),
                           particle_type=events.batch["particle_type"].numpy().astype(np.int16),
                           particle_charge=events.batch["particle_charge"].numpy().astype(np.int16),
                           particle_mask=events.batch["particle_mask"].numpy() > 0.5, event=events.event,
                           nans=NanCounts(on_nan, events.event, np.concatenate(nans)))


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
    for the ablation. The NaN counts of `generated.nans`, if any, go in as "NaN counts". Creates the folder of `path`
    if needed.
    """
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"metadata has no {missing}")
    if generated.nans is not None:
        metadata = {**metadata, "NaN counts": generated.nans.metadata()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{"mass": generated.mass, "four-momenta": generated.four_momenta, "event": generated.event,
                      "metadata": np.array(json.dumps(metadata, indent=1))})


def load_generated(path: str | Path) -> tuple[GeneratedPairs, dict]:
    with np.load(path) as data:
        if "mass" not in data.files:
            raise ValueError(f"{path} holds every particle of each event, not pairs; read it with "
                             f"load_generated_events")
        generated = GeneratedPairs(mass=data["mass"], four_momenta=data["four-momenta"], event=data["event"])
        metadata = json.loads(str(data["metadata"]))
    return generated, metadata


def save_generated_events(path: str | Path, generated: GeneratedEvents, metadata: dict) -> None:
    """Write every particle of `generated`, its decoded mass if any, and the settings that made it to one .npz, as
    `save_generated` does for pairs; metadata must hold every key of METADATA_KEYS."""
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata has no {missing}")
    if generated.nans is not None:
        metadata = {**metadata, "NaN counts": generated.nans.metadata()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mass = {} if generated.mass is None else {"decoded_mass": generated.mass}   # "mass": pair mass, load_generated
    np.savez(path, **{"four-momenta": generated.four_momenta, "particle_type": generated.particle_type,
                      "particle_charge": generated.particle_charge, "particle_mask": generated.particle_mask,
                      "event": generated.event, **mass, "metadata": np.array(json.dumps(metadata, indent=1))})


def load_generated_events(path: str | Path) -> tuple[GeneratedEvents, dict]:
    """The particles and the settings from `save_generated_events`; stops on a file of pairs. `mass` is None for a file
    without decoded masses."""
    with np.load(path) as data:
        if "particle_type" not in data.files:
            raise ValueError(f"{path} holds pairs, not every particle of each event; read it with load_generated")
        generated = GeneratedEvents(four_momenta=data["four-momenta"], particle_type=data["particle_type"],
                                    particle_charge=data["particle_charge"], particle_mask=data["particle_mask"],
                                    event=data["event"],
                                    mass=data["decoded_mass"] if "decoded_mass" in data.files else None)
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
