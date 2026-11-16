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


class Classes(Section):
    min_mass: float
    windows: dict[str, list[float]]


class Events(Section):
    split: Literal["validation"]
    pair: Literal["muon", "electron", "tau", "photon", "jet", "largeRjet"]
    order_seed: int
    n_A: int
    n_B: int


class Sampler(Section):
    steps: int
    batch_size: int
    precision: Literal["float32"]


class Gradients(Section):
    blocks: list[int]
    seed: int


class Directions(Section):
    method: Literal["fkt", "lda", "meandiff"]
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
    r60_max_mass: float
    spectrum_bins: int


class Config(Section):
    name: str
    network: Literal["shellflow", "shellflow-b", "shellgatr"]
    classes: Classes
    target: str
    events: Events
    sampler: Sampler
    gradients: Gradients
    directions: Directions
    steering: Steering
    readout: Readout

    @model_validator(mode="after")
    def check(self):
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
        if self.steering.tokens in ("minus", "plus") and self.events.pair not in ("muon", "electron", "tau"):
            raise ValueError(f"steering.tokens: {self.steering.tokens!r} needs a charged pair")
        return self


def load_config(path: str | Path) -> Config:
    """Read the run config at `path` and the classes file it names (path relative to the run config)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw["classes"] = yaml.safe_load((path.parent / raw["classes"]).read_text())
    return Config.model_validate(raw)
