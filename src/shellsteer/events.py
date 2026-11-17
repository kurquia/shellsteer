"""Which events the pipeline uses, and which tokens of each event it edits.

An event here is its conditioning: the type, charge, extra features, and mask of each particle, and the missing
momentum. The network generates the kinematics; `pair_mass` reads the mass of the pair from them. A dataset of pairs
in datasets.yaml holds two samples of events with exactly one pair of the dataset's type, one negative and one positive
particle:

    directions: the events for the gradients and the steering vectors.
    steering:   the events we steer and generate.

`load_events` reads them. Each file holds the conditioning (BATCH_KEYS), `event` (the index of the event in the
combined dataset), `priority` (its position in a seeded random order of the events), and `m_real` (the real-data mass
of the pair in GeV, float32, the grey line of paper Figs. 3 and 7).

A dataset of every final state (pair: null in datasets.yaml, sample C of paper Fig. 4) holds only a steering sample.
Its events have zero, one, or several particles of each type and no pair. `load_final_states` reads them. The file
holds the conditioning, `event`, `p4` (the real four-momenta, the grey lines of Fig. 4), and `m_real`, which is NaN
except in events with exactly one muon pair. It has no priority: the events load in file order.

Every file comes from Hugging Face at a pinned revision. The events are stored in the order the paper ran them, so the
batches and their noise seeds match the paper's runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from shellsteer.config import Classes, DatasetEntry

# A batch holds the event skeleton: types, charges, extra features, mask, and missing momentum.
# The network generates the kinematics; these keys only condition it.
BATCH_KEYS = ("particle_type", "particle_charge", "particle_extra", "particle_mask", "met_vector")

PAIR_TYPES = {"electron": 11, "muon": 13, "tau": 15}   # ShellFlow's PARTICLE_TYPE_CODES
# Every particle type of ShellFlow, code -> name: the rows of paper Fig. 14, edited type first in Fig. 4.
PARTICLE_TYPES = {13: "muon", 11: "electron", 22: "photon", 1: "jet", 10: "large-R jet", 15: "hadronic tau"}
NEGATIVE, POSITIVE = 0, 2                               # ShellFlow stores charge + 1


@dataclass(frozen=True)
class Events:
    batch: dict[str, torch.Tensor]   # BATCH_KEYS, every event of the sample, on the CPU
    minus: torch.Tensor              # (E,) slot of the negative particle of the pair
    plus: torch.Tensor               # (E,) slot of the positive particle of the pair
    event: np.ndarray                # (E,) index of the event in the combined dataset
    priority: np.ndarray             # (E,) position of the event in the seeded random order
    m_real: np.ndarray               # (E,) float32 real-data mass of the pair in GeV, as stored

    def __len__(self) -> int:
        return len(self.event)


@dataclass(frozen=True)
class FinalStates:
    """The events of a dataset of every final state: no pair, no priority."""
    batch: dict[str, torch.Tensor]   # BATCH_KEYS, every event of the sample, on the CPU
    event: np.ndarray                # (E,) index of the event in the combined dataset
    p4: np.ndarray                   # (E, N, 4) float32 real four-momenta (px, py, pz, E) in GeV, zero on padding
    m_real: np.ndarray               # (E,) float32 real-data mass of the muon pair in GeV; NaN without one muon pair

    def __len__(self) -> int:
        return len(self.event)


def download(entry: DatasetEntry, sample: Literal["directions", "steering"]) -> str:
    """Local path of the file of `sample`, from Hugging Face or its cache. Stops if the entry has no such file."""
    name = getattr(entry.files, sample)
    if name is None:
        raise ValueError(f"the dataset of {entry.files.steering!r} has no {sample} sample")
    return hf_hub_download(entry.hf_repo, name, revision=entry.revision, repo_type="dataset")


def check_file(path: str, data, keys: set[str], entry: DatasetEntry, n: int) -> int:
    """The number of events in the open file `data`; stops if it lacks a key of `keys`, if the number differs from
    datasets.yaml, or if it is below n."""
    missing = keys - set(data.files)
    if missing:
        raise ValueError(f"{path} has no {sorted(missing)}")
    size = len(data["event"])
    if size != entry.events:
        raise ValueError(f"{path} has {size} events, datasets.yaml says {entry.events}")
    if n > size:
        raise ValueError(f"{path} has {size} events, the config asks for {n}")
    return size


def as_conditioning(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The batch with types and charges as int64, the dtype the network takes; the files store them as int16."""
    return {**batch, "particle_type": batch["particle_type"].long(), "particle_charge": batch["particle_charge"].long()}


def load_events(entry: DatasetEntry, sample: Literal["directions", "steering"], n: int) -> Events:
    """The n events of `sample` with the lowest priority, kept in file order.

    The files are sorted by physics sample, so the first n events are not a random choice; the lowest priorities are.
    With n equal to the size of the file, every event loads in the order of the file.
    """
    if entry.pair is None:
        raise ValueError(f"the dataset {entry.files.steering!r} has no pairs (pair: null); read it with "
                         f"load_final_states")
    keys = {*BATCH_KEYS, "event", "priority", "m_real"}
    path = download(entry, sample)
    with np.load(path) as data:
        check_file(path, data, keys, entry, n)
        keep = np.sort(np.argsort(data["priority"], kind="stable")[:n])
        batch = {key: torch.from_numpy(data[key][keep]) for key in BATCH_KEYS}
        event, priority, m_real = data["event"][keep], data["priority"][keep], data["m_real"][keep]
    if m_real.dtype != np.float32:
        raise ValueError(f"{path}: m_real is {m_real.dtype}, expected float32")
    batch = as_conditioning(batch)
    minus, plus = pair_slots(batch, entry.pair)
    return Events(batch=batch, minus=minus, plus=plus, event=event, priority=priority, m_real=m_real)


def load_final_states(entry: DatasetEntry, n: int) -> FinalStates:
    """The first n events of a dataset of every final state (pair: null), in file order.

    The file has no priority, so the first n events are the first n of the file, which is sorted by physics sample;
    only n equal to the size of the file gives the paper's sample.
    """
    if entry.pair is not None:
        raise ValueError(f"the dataset {entry.files.steering!r} holds {entry.pair} pairs; read it with load_events")
    keys = {*BATCH_KEYS, "event", "p4", "m_real"}
    path = download(entry, "steering")
    with np.load(path) as data:
        check_file(path, data, keys, entry, n)
        batch = {key: torch.from_numpy(data[key][:n]) for key in BATCH_KEYS}
        event, p4, m_real = data["event"][:n], data["p4"][:n], data["m_real"][:n]
    for name, array in (("p4", p4), ("m_real", m_real)):
        if array.dtype != np.float32:
            raise ValueError(f"{path}: {name} is {array.dtype}, expected float32")
    if p4.shape != (*batch["particle_type"].shape, 4):
        raise ValueError(f"{path}: p4 has shape {p4.shape}, expected {(*batch['particle_type'].shape, 4)}")
    return FinalStates(batch=as_conditioning(batch), event=event, p4=p4, m_real=m_real)


def pair_slots(batch: dict[str, torch.Tensor], pair: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(E,) slots of the negative and of the positive particle. Stops if an event has not exactly one of each."""
    of_type = (batch["particle_mask"] > 0.5) & (batch["particle_type"] == PAIR_TYPES[pair])
    negative = of_type & (batch["particle_charge"] == NEGATIVE)
    positive = of_type & (batch["particle_charge"] == POSITIVE)
    bad = (negative.sum(1) != 1) | (positive.sum(1) != 1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} events have not exactly one {pair} pair, "
                         f"the first at row {int(bad.nonzero()[0, 0])}")
    return negative.int().argmax(1), positive.int().argmax(1)


def tokens_of_type(batch: dict[str, torch.Tensor], particle: str) -> tuple[torch.Tensor, torch.Tensor]:
    """negative, positive: (B, N), True on every real particle of type `particle` (a key of PAIR_TYPES) with charge
    -1 and with charge +1. An event without such a particle has no True entry."""
    of_type = (batch["particle_mask"] > 0.5) & (batch["particle_type"] == PAIR_TYPES[particle])
    return of_type & (batch["particle_charge"] == NEGATIVE), of_type & (batch["particle_charge"] == POSITIVE)


def edited_tokens(batch: dict[str, torch.Tensor], minus: torch.Tensor, plus: torch.Tensor, pair: str,
                  tokens: Literal["pair", "minus", "plus", "all_of_type"]) -> tuple[torch.Tensor, torch.Tensor]:
    """edit_minus, edit_plus: (B, N), True on the tokens that Steer edits with xi^- and with xi^+.

    pair: the two particles of the pair. minus, plus: one of them. all_of_type: every particle of the pair's type,
    negative ones with xi^-, positive ones with xi^+.
    """
    slots = torch.arange(batch["particle_type"].shape[1], device=minus.device)
    at_minus, at_plus = slots == minus[:, None], slots == plus[:, None]
    none = torch.zeros_like(at_minus)
    if tokens == "pair":
        return at_minus, at_plus
    if tokens == "minus":
        return at_minus, none
    if tokens == "plus":
        return none, at_plus
    if tokens == "all_of_type":
        return tokens_of_type(batch, pair)
    raise ValueError(f"tokens: {tokens!r} is not pair, minus, plus, or all_of_type")


def pair_mass_squared(p4: torch.Tensor, minus: torch.Tensor, plus: torch.Tensor) -> torch.Tensor:
    """(B,) m^2 = (E^- + E^+)^2 - |p^- + p^+|^2 of the pair in GeV^2, from the four-momenta (px, py, pz, E) that
    `generate` returns. It can be negative or zero for a badly generated pair."""
    rows = torch.arange(len(minus), device=p4.device)
    total = p4[rows, minus] + p4[rows, plus]
    return total[:, 3] ** 2 - (total[:, :3] ** 2).sum(1)


def pair_mass(p4: torch.Tensor, minus: torch.Tensor, plus: torch.Tensor) -> torch.Tensor:
    """(B,) invariant mass of the pair in GeV, with m^2 floored at 1e-9 GeV^2, as the paper's generated masses."""
    return pair_mass_squared(p4, minus, plus).clamp(min=1e-9).sqrt()


def log_pair_mass(p4: torch.Tensor, minus: torch.Tensor, plus: torch.Tensor) -> torch.Tensor:
    """(B,) log m of the pair, the quantity the gradients are taken of: 0.5 log m^2, with m^2 floored at 1e-6 GeV^2
    as in the paper's gradient pass. No square root, so the gradient is finite everywhere."""
    return 0.5 * torch.log(pair_mass_squared(p4, minus, plus).clamp(min=1e-6))


def batches(events: Events, batch_size: int, device: str | torch.device
            ) -> Iterator[tuple[int, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
    """(batch index, batch, minus, plus) on `device`, in file order. The batch index enters the noise seed."""
    for index in range(batch_count(events, batch_size)):
        yield index, *batch_at(events, index, batch_size, device)


def batch_count(events: Events, batch_size: int) -> int:
    return -(-len(events) // batch_size)


def batch_at(events: Events, index: int, batch_size: int, device: str | torch.device
             ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Batch `index` of `batches` (events index x batch_size onwards, in file order), minus, and plus on `device`."""
    part = slice(index * batch_size, (index + 1) * batch_size)
    batch = {key: value[part].to(device) for key, value in events.batch.items()}
    return batch, events.minus[part].to(device), events.plus[part].to(device)


def final_state_batches(events: FinalStates, batch_size: int, device: str | torch.device
                        ) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """(batch index, batch) on `device`, in file order, as `batches` for a sample without pairs."""
    for index, start in enumerate(range(0, len(events), batch_size)):
        yield index, {key: value[start:start + batch_size].to(device) for key, value in events.batch.items()}


def class_of(mass: np.ndarray, classes: Classes) -> np.ndarray:
    """Class of each pair from its generated mass without edits, in GeV.

    A window name if the mass is inside the window (open interval), "rest" above min_mass outside every window, and
    "" at or below min_mass or when the mass is not finite. A pair with "" leaves the sample.
    """
    mass = np.asarray(mass, dtype=np.float64)
    names = np.full(mass.shape, "", dtype=object)
    names[np.isfinite(mass) & (mass > classes.min_mass)] = "rest"
    for name, (low, high) in classes.windows.items():
        names[(mass > low) & (mass < high)] = name
    return names


def class_names(classes: Classes) -> list[str]:
    """The classes in the order of the moment arrays: the windows as in classes.yaml, then "rest"."""
    return [*classes.windows, "rest"]


def select(mass: np.ndarray, priority: np.ndarray, classes: Classes, keep: int) -> np.ndarray:
    """Which pairs of a sample enter the classes: the `keep` pairs with a class of lowest priority.

    mass: generated mass without edits, per event. Returns the class of each pair as `class_of` does, with "" for
    the pairs left out. Stops if fewer than `keep` pairs have a class.
    """
    names = class_of(mass, classes)
    has_class = np.flatnonzero(names != "")
    if len(has_class) < keep:
        raise ValueError(f"{len(has_class)} pairs have a class, events.keep asks for {keep}")
    chosen = has_class[np.argsort(priority[has_class], kind="stable")[:keep]]
    out = np.full(len(names), "", dtype=object)
    out[chosen] = names[chosen]
    return out
