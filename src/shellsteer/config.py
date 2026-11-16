"""Read a run config (configs/paper/*.yaml) with pydantic.

pydantic checks the keys, the types, and the options from the type hints. An unknown key, a missing key, a wrong
type, or a value outside its options stops with a pydantic ValidationError that names the key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NetworkEntry(Section):
    hf_repo: str
    revision: str
    file: str
    width: int


class DatasetFiles(Section):
    directions: str
    steering: str


class DatasetEntry(Section):
    hf_repo: str
    revision: str
    files: DatasetFiles
    pair: Literal["muon", "electron", "tau"]
    events: int


class Classes(Section):
    min_mass: float
    windows: dict[str, list[float]]


class Events(Section):
    dataset: str
    directions: int
    steering: int
    keep: int


class Sampler(Section):
    steps: int
    batch_size: int
    precision: Literal["float32"]


class Gradients(Section):
    blocks: list[int]
    seed: int


class SteeringVectors(Section):
    method: Literal["fkt", "fisher", "meandiff"]
    in_time: Literal["per_step", "constant"]
    balance: Literal["equal", "by_count"]


class Steering(Section):
    block: int
    alphas: list[float]
    tokens: Literal["pair", "minus", "plus", "all_of_type"]
    steps: Literal["all"] | list[int]
    norm_restore: bool
    seeds: list[int]


class Readout(Section):
    spectrum_range: list[float]
    spectrum_bins: int
    r60_max_mass: float
    r60_min_baseline_pairs: int


class Lineshape(Section):
    half_width: float
    bins: int
    min_window_pairs: int


class Plots(Section):
    zooms: dict[str, list[float]]
    zoom_bins: int
    baseline_label: str
    pair_label: str
    target_label: str


class Config(Section):
    name: str
    network: Literal["shellflow"]
    networks: dict[str, NetworkEntry]
    datasets: dict[str, DatasetEntry]
    classes: Classes
    target: str
    events: Events
    sampler: Sampler
    gradients: Gradients
    steering_vectors: SteeringVectors
    steering: Steering
    readout: Readout
    lineshape: Lineshape
    plots: Plots

    @model_validator(mode="after")
    def check(self):
        if self.network not in self.networks:
            raise ValueError(f"network: {self.network!r} has no entry in the networks file")
        dataset = self.datasets.get(self.events.dataset)
        if dataset is None:
            raise ValueError(f"events.dataset: {self.events.dataset!r} has no entry in the datasets file")
        for sample in ("directions", "steering"):
            n = getattr(self.events, sample)
            if not 0 < n <= dataset.events:
                raise ValueError(f"events.{sample}: need 0 < n <= {dataset.events} (events in the file), got {n}")
            if self.events.keep > n:
                raise ValueError(f"events.keep: {self.events.keep} is more than the {n} events of events.{sample}")
        windows = sorted(self.classes.windows.values())
        for lo, hi in windows:
            if not self.classes.min_mass <= lo < hi:
                raise ValueError(f"classes.windows: need min_mass <= low < high, got [{lo}, {hi}]")
        if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
            raise ValueError("classes.windows: windows overlap")
        if self.target not in self.classes.windows:
            raise ValueError(f"target: {self.target!r} is not one of {sorted(self.classes.windows)}")
        if self.steering.block not in self.gradients.blocks:
            raise ValueError(f"steering.block: block {self.steering.block} is not in gradients.blocks")
        if 0 in self.steering.alphas:
            raise ValueError("steering.alphas: leave out 0; the baseline always runs")
        steps = self.steering.steps
        if steps != "all" and (len(steps) != 2 or not 0 <= steps[0] <= steps[1] < self.sampler.steps):
            raise ValueError(f"steering.steps: need 0 <= first <= last < {self.sampler.steps}, got {list(steps)}")
        # The checks below guard settings that the run reads only after the generation.
        spectrum = self.readout.spectrum_range
        if len(spectrum) != 2 or not 0 < spectrum[0] < spectrum[1]:
            raise ValueError(f"readout.spectrum_range: need [low, high] with 0 < low < high, got {spectrum}")
        if not spectrum[0] < self.readout.r60_max_mass <= spectrum[1]:
            raise ValueError(f"readout.r60_max_mass: need it inside readout.spectrum_range {spectrum}, "
                             f"got {self.readout.r60_max_mass}")
        if self.lineshape.half_width <= 0 or self.lineshape.bins <= 5:
            raise ValueError(f"lineshape: need half_width > 0 and more bins than the 5 fitted parameters, got "
                             f"half_width {self.lineshape.half_width} and bins {self.lineshape.bins}")
        for label, zoom in self.plots.zooms.items():
            if len(zoom) != 2 or not 0 < zoom[0] < zoom[1]:
                raise ValueError(f"plots.zooms[{label!r}]: need [low, high] with 0 < low < high, got {zoom}")
        return self


def load_config(path: str | Path) -> Config:
    """Read the run config at `path` and the networks, datasets, and classes files it names.

    The paths of those files are relative to the run config.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw["networks"] = yaml.safe_load((path.parent / raw["networks"]).read_text())
    raw["datasets"] = yaml.safe_load((path.parent / raw["datasets"]).read_text())
    raw["classes"] = yaml.safe_load((path.parent / raw["classes"]).read_text())
    return Config.model_validate(raw)


def read_networks(path: str | Path) -> dict[str, NetworkEntry]:
    """Read a networks file on its own, for example in a notebook: network name -> entry."""
    raw = yaml.safe_load(Path(path).read_text())
    return {name: NetworkEntry.model_validate(entry) for name, entry in raw.items()}


def read_datasets(path: str | Path) -> dict[str, DatasetEntry]:
    """Read a datasets file on its own, for example in a notebook: dataset name -> entry."""
    raw = yaml.safe_load(Path(path).read_text())
    return {name: DatasetEntry.model_validate(entry) for name, entry in raw.items()}
