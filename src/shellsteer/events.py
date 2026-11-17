"""Which events the pipeline uses, and which tokens of each event it edits.

An event here is its conditioning: the type, charge, extra features, and mask of each particle, and the missing
momentum. The network generates the kinematics; `pair_mass` reads the mass of the pair from them. A dataset of pairs
in datasets.yaml holds two samples of events with exactly one pair of the dataset's type, one negative and one positive
particle:

    directions: the events for the gradients and the steering vectors.
    steering:   the events we steer and generate.

`load_events` reads them. Each file holds `rows` (the ShellFlow row of each event, see below), the conditioning
without the extra features (STORED_KEYS), `event` (the index of the event in the combined dataset), `priority` (its
position in a seeded random order of the events), and `m_real` (the real-data mass of the pair in GeV, float32, the
grey line of paper Figs. 3 and 7).

A dataset of every final state (pair: null in datasets.yaml, sample C of paper Fig. 4) holds only a steering sample.
Its events have zero, one, or several particles of each type and no pair. `load_final_states` reads them. The file
holds `rows`, STORED_KEYS, `event`, `p4` (the real four-momenta, the grey lines of Fig. 4), and `m_real`, which is NaN
except in events with exactly one muon pair. It has no priority: the events load in file order. `final_state_rows`
streams the rows of the same events, for the real-data mass of Fig. 14.

`rows` holds the bytes ShellFlow's ParticleDataset reads for each event: float32 (E, 2, L), the raw feature values
and the packed particle and feature indices. L is the row length of the source dataset: 266 for the combined dataset,
161 for 2to4lep. Each network normalises the extra features with the constants of its own
dataloader config (networks.yaml), so the files hold no `particle_extra`: `particle_extra` builds it per network with
ShellFlow's normaliser and collator.

Every file comes from Hugging Face at a pinned revision. The events are stored in the order the paper ran them, so the
batches and their noise seeds match the paper's runs. A datasets.yaml entry whose files are not uploaded yet has the
revision NOT_UPLOADED; `download` stops on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Iterator, Literal

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from shellflow.data.particle_dataset import ParticleDataset
from shellflow.data.particle_vector_collator import ParticleVectorCollator

from shellsteer.config import Classes, DatasetEntry, NetworkEntry

# A batch holds the event skeleton: types, charges, extra features, mask, and missing momentum.
# The network generates the kinematics; these keys only condition it.
BATCH_KEYS = ("particle_type", "particle_charge", "particle_extra", "particle_mask", "met_vector")
# The keys a file stores as they are; particle_extra comes from `rows`.
STORED_KEYS = tuple(key for key in BATCH_KEYS if key != "particle_extra")
PARTICLES = 10                                          # particle slots per event, ShellFlow's max_particles

PAIR_TYPES = {"electron": 11, "muon": 13, "tau": 15}   # ShellFlow's PARTICLE_TYPE_CODES
# Every particle type of ShellFlow, code -> name: the rows of paper Fig. 14, edited type first in Fig. 4.
PARTICLE_TYPES = {13: "muon", 11: "electron", 22: "photon", 1: "jet", 10: "large-R jet", 15: "hadronic tau"}
NEGATIVE, POSITIVE = 0, 2                               # ShellFlow stores charge + 1
NOT_UPLOADED = "TODO-after-upload"                      # datasets.yaml revision of files not on Hugging Face yet


class NotUploadedError(ValueError):
    """A datasets.yaml entry names files that are not on Hugging Face yet (revision NOT_UPLOADED)."""


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
    """Local path of the file of `sample`, from Hugging Face or its cache. Stops if the entry has no such file, or if
    its files are not uploaded yet (revision NOT_UPLOADED)."""
    name = getattr(entry.files, sample)
    if name is None:
        raise ValueError(f"the dataset of {entry.files.steering!r} has no {sample} sample")
    if entry.revision == NOT_UPLOADED:
        raise NotUploadedError(f"{name} is not on Hugging Face yet: its datasets.yaml entry has revision "
                               f"{NOT_UPLOADED!r}. Upload the file to {entry.hf_repo} and set revision to the commit "
                               f"of the upload")
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


def particle_extra(rows: np.ndarray, dataloader_config: str, batch_size: int = 16384) -> np.ndarray:
    """(E, 10, 13) float32 extra features of the events in `rows` (float32 (E, 2, L) ShellFlow rows), normalised
    with the constants of `dataloader_config`, a file name in shellflow/config/.

    This is ShellFlow's training path: the row split of ParticleDataset.__getitem__, its normaliser, and its collator,
    `batch_size` events at a time.
    """
    if rows.dtype != np.float32 or rows.ndim != 3 or rows.shape[1] != 2 or rows.shape[2] < 3:
        raise ValueError(f"rows: {rows.dtype} {rows.shape}, expected float32 (E, 2, L) ShellFlow rows")
    path = resources.files("shellflow") / "config" / dataloader_config
    if not path.is_file():
        raise ValueError(f"dataloader config {dataloader_config!r} is not in shellflow/config/")
    normalization = yaml.safe_load(path.read_text())["normalization"]
    dataset = ParticleDataset.__new__(ParticleDataset)    # __init__ needs the data files; the normaliser does not
    dataset._setup_normalization({stats["id"]: stats for stats in normalization})
    collate = ParticleVectorCollator(max_particles=PARTICLES)
    extra = np.empty((len(rows), PARTICLES, 13), np.float32)
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        features = torch.from_numpy(part[:, 0, 2:].copy())
        packed = torch.from_numpy(part[:, 1, 2:].copy()).long()
        particle_idxs, feature_idxs = packed // 1000, packed % 1000
        features = dataset._normalize_features(features, feature_idxs)
        items = [{"features": f, "particle_idxs": p, "feature_idxs": i}
                 for f, p, i in zip(features, particle_idxs, feature_idxs)]
        extra[start:start + len(part)] = collate(items)["particle_extra"].numpy()
    return extra


def row_blocks(data, path: str, keep: np.ndarray, batch_size: int = 16384
               ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(index, rows) of the events `keep` (sorted row numbers) of the open file `data`, block by block: index holds
    row numbers of `keep`, rows their float32 (len(index), 2, L) ShellFlow rows. The blocks cover `keep` in order.

    A compressed file cannot be memory-mapped, so this streams `rows` in blocks of `batch_size` events and yields
    only the rows of the needed events. The rows of the whole file (2.2 GB) are never in memory at once.
    """
    with data.zip.open("rows.npy") as file:
        version = np.lib.format.read_magic(file)
        read_header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        shape, fortran, dtype = read_header[version](file)
        if dtype != np.float32 or fortran or len(shape) != 3 or shape[1] != 2 or shape[2] < 3:
            raise ValueError(f"{path}: rows is {dtype} {shape}, expected float32 (E, 2, L) in C order")
        if len(keep) and keep[-1] >= shape[0]:
            raise ValueError(f"{path}: rows holds {shape[0]} events, the file has more")
        done, row_bytes = 0, 2 * shape[2] * 4
        for start in range(0, int(keep[-1]) + 1 if len(keep) else 0, batch_size):
            count = min(batch_size, shape[0] - start)
            block = np.frombuffer(file.read(count * row_bytes), np.float32).reshape(count, 2, shape[2])
            wanted = keep[done:np.searchsorted(keep, start + count)]
            yield wanted, block[wanted - start]
            done += len(wanted)
            del block


def read_extra(data, path: str, keep: np.ndarray, dataloader_config: str, batch_size: int = 16384) -> np.ndarray:
    """(len(keep), 10, 13) particle_extra of the events `keep` (sorted row numbers) of the open file `data`, from
    the rows of `row_blocks`."""
    extra = np.empty((len(keep), PARTICLES, 13), np.float32)
    done = 0
    for index, rows in row_blocks(data, path, keep, batch_size):
        extra[done:done + len(index)] = particle_extra(rows, dataloader_config, batch_size)
        done += len(index)
    return extra


def load_events(entry: DatasetEntry, sample: Literal["directions", "steering"], n: int,
                network: NetworkEntry) -> Events:
    """The n events of `sample` with the lowest priority, kept in file order, with the extra features of `network`.

    The files are sorted by physics sample, so the first n events are not a random choice; the lowest priorities are.
    With n equal to the size of the file, every event loads in the order of the file.
    """
    if entry.pair is None:
        raise ValueError(f"the dataset {entry.files.steering!r} has no pairs (pair: null); read it with "
                         f"load_final_states")
    keys = {*STORED_KEYS, "rows", "event", "priority", "m_real"}
    path = download(entry, sample)
    with np.load(path) as data:
        check_file(path, data, keys, entry, n)
        keep = np.sort(np.argsort(data["priority"], kind="stable")[:n])
        extra = read_extra(data, path, keep, network.dataloader_config)
        batch = {key: torch.from_numpy(extra if key == "particle_extra" else data[key][keep]) for key in BATCH_KEYS}
        event, priority, m_real = data["event"][keep], data["priority"][keep], data["m_real"][keep]
    if m_real.dtype != np.float32:
        raise ValueError(f"{path}: m_real is {m_real.dtype}, expected float32")
    batch = as_conditioning(batch)
    minus, plus = pair_slots(batch, entry.pair)
    return Events(batch=batch, minus=minus, plus=plus, event=event, priority=priority, m_real=m_real)


def load_final_states(entry: DatasetEntry, n: int, network: NetworkEntry) -> FinalStates:
    """The first n events of a dataset of every final state (pair: null), in file order, with the extra features of
    `network`.

    The file has no priority, so the first n events are the first n of the file, which is sorted by physics sample;
    only n equal to the size of the file gives the paper's sample.
    """
    if entry.pair is not None:
        raise ValueError(f"the dataset {entry.files.steering!r} holds {entry.pair} pairs; read it with load_events")
    keys = {*STORED_KEYS, "rows", "event", "p4", "m_real"}
    path = download(entry, "steering")
    with np.load(path) as data:
        check_file(path, data, keys, entry, n)
        extra = read_extra(data, path, np.arange(n), network.dataloader_config)
        batch = {key: torch.from_numpy(extra if key == "particle_extra" else data[key][:n]) for key in BATCH_KEYS}
        event, p4, m_real = data["event"][:n], data["p4"][:n], data["m_real"][:n]
    for name, array in (("p4", p4), ("m_real", m_real)):
        if array.dtype != np.float32:
            raise ValueError(f"{path}: {name} is {array.dtype}, expected float32")
    if p4.shape != (*batch["particle_type"].shape, 4):
        raise ValueError(f"{path}: p4 has shape {p4.shape}, expected {(*batch['particle_type'].shape, 4)}")
    return FinalStates(batch=as_conditioning(batch), event=event, p4=p4, m_real=m_real)


def final_state_rows(entry: DatasetEntry, n: int, batch_size: int = 16384) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(index, rows) of the first n events of a dataset of every final state, the events of `load_final_states`,
    block by block as `row_blocks` yields them. The real-data mass of paper Fig. 14 comes from these rows."""
    if entry.pair is not None:
        raise ValueError(f"the dataset {entry.files.steering!r} holds {entry.pair} pairs; it has no final states")
    path = download(entry, "steering")
    with np.load(path) as data:
        check_file(path, data, {"rows", "event"}, entry, n)
        yield from row_blocks(data, path, np.arange(n), batch_size)


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


def batch_count(events: Events | FinalStates, batch_size: int) -> int:
    """Number of batches of `batch_size` events, the last one possibly shorter."""
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
    for index in range(batch_count(events, batch_size)):
        yield index, final_state_batch_at(events, index, batch_size, device)


def final_state_batch_at(events: FinalStates, index: int, batch_size: int, device: str | torch.device
                         ) -> dict[str, torch.Tensor]:
    """Batch `index` of `final_state_batches` (events index x batch_size onwards, in file order) on `device`."""
    part = slice(index * batch_size, (index + 1) * batch_size)
    return {key: value[part].to(device) for key, value in events.batch.items()}


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
