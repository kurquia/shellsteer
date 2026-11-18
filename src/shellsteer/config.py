"""Read a run config (configs/paper/*.yaml) with pydantic.

pydantic checks the keys, the types, and the options from the type hints. An unknown key, a missing key, a wrong
type, or a value outside its options stops with a pydantic ValidationError that names the key.

A config runs one of three chains (cli.py), and `Config.chain` says which:

    steering:   steering, readout, lineshape, and plots; the steering sample holds muon pairs.
    kinematics: steering and kinematics; the steering sample holds every final state (events.steering_dataset).
    ablation:   ablation and readout; the steering sample holds muon pairs.

Each chain needs its own sections and stops on the sections of the others.
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
    dataloader_config: str      # a file name in shellflow/config/, the constants of the extra features


class DatasetFiles(Section):
    directions: str | None = None    # only a dataset of pairs has a directions sample
    steering: str


class DatasetEntry(Section):
    hf_repo: str
    revision: str
    files: DatasetFiles
    # The pair type of every event, or None (null in YAML) for events of every final state, which have no pair.
    pair: Literal["muon", "electron", "tau"] | None
    events: int

    @model_validator(mode="after")
    def check(self):
        if self.pair is not None and self.files.directions is None:
            raise ValueError(f"files.directions: a dataset of {self.pair} pairs needs a directions sample")
        if self.pair is None and self.files.directions is not None:
            raise ValueError("files.directions: a dataset without pairs (pair: null) has no directions sample; the "
                             "gradients are those of the pair mass")
        return self


class Classes(Section):
    min_mass: float
    windows: dict[str, list[float]]


class Events(Section):
    dataset: str                     # the directions sample, and the steering sample unless steering_dataset is set
    # The dataset of the steering sample if it is not events.dataset, such as the events of every final state of
    # paper Fig. 4. None: events.dataset.
    steering_dataset: str | None = None
    directions: int
    steering: int
    keep: int


class Sampler(Section):
    steps: int
    batch_size: int
    precision: Literal["float32"]
    # What happens to a NaN in the final state or the four-momenta (sampler.generate). Every run counts the NaNs and
    # writes the counts to its log and to the metadata of its files. set_to_zero: set NaN coordinates of the final
    # state to 0, as ShellFlow's sampler and the paper; stop: stop with an error that names the batch and the count.
    on_nan: Literal["set_to_zero", "stop"] = "set_to_zero"


class Gradients(Section):
    blocks: list[int]
    seed: int


class SteeringVectors(Section):
    method: Literal["fkt", "fisher", "meandiff"]
    in_time: Literal["per_step", "constant"]
    balance: Literal["equal", "by_count"]
    # A steering_vectors.npz written by `steer steering-vectors`, such as vectors/main_block1.npz; a relative path is
    # relative to the working directory, as --runs. With it, `steer steering-vectors` checks the file against this
    # config and copies it into the run folder, and the run has no gradient pass (cli.py). None: the vectors come
    # from the gradient pass of the run.
    from_file: str | None = None


class Steering(Section):
    block: int
    alphas: list[float]
    tokens: Literal["pair", "minus", "plus", "all_of_type"]
    steps: Literal["all"] | list[int]
    norm_restore: bool
    seeds: list[int]


class AblationFigure(Section):
    """Settings of paper Fig. 7 that the readout does not use."""
    inset_range: list[float]         # GeV: the mass range the inset shows, inside peak_range
    ratio_range: list[float]         # y range of the ratio panel, linear
    baseline_label: str
    pair_label: str
    target_label: str


class Ablation(Section):
    """The mean ablation of paper Sec. 5.3 (Eq. 24) of the steering sample: the target's steering vector at `block`,
    xi and c_k (Eq. 25) from the directions sample."""
    block: int
    seeds: list[int]
    peak_range: list[float]          # GeV: the histogram of the peak height, and the bins of the inset of paper Fig. 7
    peak_bins: int
    figure: AblationFigure


class Readout(Section):
    spectrum_range: list[float]
    spectrum_bins: int
    r60_max_mass: float
    r60_min_baseline_pairs: int


class Lineshape(Section):
    half_width: float
    bins: int
    min_window_pairs: int


class Kinematics(Section):
    """Paper Fig. 4 and Fig. 14 (App. D.1): the kinematics of every particle type of a steering sample of every final
    state."""
    bins: int                        # equal bins per panel: in log for p_T and E, linear for eta and phi
    readout_min_baseline: int        # the largest change per bin counts bins with at least this many baseline particles
    ratio_min_baseline: int          # ratio panels: drawn in bins with at least this many baseline particles
    other_types_ratio_range: list[float]   # y range of every ratio panel of Fig. 14; Fig. 4 scales each panel
    baseline_label: str


class Plots(Section):
    zooms: dict[str, list[float]]
    zoom_bins: int
    baseline_label: str
    pair_label: str
    target_label: str


class Config(Section):
    name: str
    network: str                     # a network of the networks file: shellflow, shellflow_b, or shellgatr
    networks: dict[str, NetworkEntry]
    datasets: dict[str, DatasetEntry]
    classes: Classes
    target: str
    events: Events
    sampler: Sampler
    gradients: Gradients
    steering_vectors: SteeringVectors
    steering: Steering | None = None
    readout: Readout | None = None
    lineshape: Lineshape | None = None
    plots: Plots | None = None
    kinematics: Kinematics | None = None
    ablation: Ablation | None = None

    def dataset(self, sample: Literal["directions", "steering"]) -> DatasetEntry:
        """The datasets.yaml entry of the directions or the steering sample."""
        if sample == "steering" and self.events.steering_dataset is not None:
            return self.datasets[self.events.steering_dataset]
        return self.datasets[self.events.dataset]

    @property
    def chain(self) -> Literal["steering", "kinematics", "ablation"]:
        """The run of this config: ablation with an ablation section, else kinematics if the steering sample has no
        pairs, else steering."""
        if self.ablation is not None:
            return "ablation"
        return "kinematics" if self.dataset("steering").pair is None else "steering"

    @model_validator(mode="after")
    def check(self):
        if self.network not in self.networks:
            raise ValueError(f"network: {self.network!r} has no entry in the networks file")
        for key in ("dataset", "steering_dataset"):
            name = getattr(self.events, key)
            if name is not None and name not in self.datasets:
                raise ValueError(f"events.{key}: {name!r} has no entry in the datasets file")
        if self.dataset("directions").pair is None:
            raise ValueError(f"events.dataset: {self.events.dataset!r} has no pairs (pair: null), so it has no "
                             f"directions sample; name it in events.steering_dataset")
        steering_pair = self.dataset("steering").pair
        if steering_pair is not None and steering_pair != self.dataset("directions").pair:
            raise ValueError(f"events.steering_dataset: {self.events.steering_dataset!r} holds {steering_pair} pairs, "
                             f"the steering vectors come from {self.dataset('directions').pair} pairs")
        for sample in ("directions", "steering"):
            n, size = getattr(self.events, sample), self.dataset(sample).events
            if not 0 < n <= size:
                raise ValueError(f"events.{sample}: need 0 < n <= {size} (events in the file), got {n}")
            if self.events.keep > n and self.dataset(sample).pair is not None:
                raise ValueError(f"events.keep: {self.events.keep} is more than the {n} events of events.{sample}")
        windows = sorted(self.classes.windows.values())
        for lo, hi in windows:
            if not self.classes.min_mass <= lo < hi:
                raise ValueError(f"classes.windows: need min_mass <= low < high, got [{lo}, {hi}]")
        if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
            raise ValueError("classes.windows: windows overlap")
        if self.target not in self.classes.windows:
            raise ValueError(f"target: {self.target!r} is not one of {sorted(self.classes.windows)}")
        sections = {"steering": self.steering, "readout": self.readout, "lineshape": self.lineshape,
                    "plots": self.plots, "kinematics": self.kinematics}
        needed = {"steering": ("steering", "readout", "lineshape", "plots"), "kinematics": ("steering", "kinematics"),
                  "ablation": ("readout",)}[self.chain]
        which = {"steering": "a config without an ablation section and a steering sample of pairs",
                 "kinematics": "a config whose steering sample has no pairs (events.steering_dataset)",
                 "ablation": "a config with an ablation section"}[self.chain]
        missing = [name for name in needed if sections[name] is None]
        if missing:
            raise ValueError(f"{', '.join(missing)}: required by the {self.chain} run ({which})")
        present = [name for name, section in sections.items() if section is not None and name not in needed]
        if present:
            raise ValueError(f"{', '.join(present)}: the {self.chain} run does not read these; leave them out of "
                             f"{which}")
        if self.chain == "kinematics":
            if self.steering.tokens != "all_of_type":
                raise ValueError(f"steering.tokens: {self.steering.tokens!r} needs a pair; the steering sample "
                                 f"{self.events.steering_dataset!r} has none, so use all_of_type")
            if len(self.steering.seeds) != 1:
                raise ValueError(f"steering.seeds: the kinematics run draws one generation per alpha, so it takes "
                                 f"one seed, got {self.steering.seeds}")
            kinematics = self.kinematics
            if kinematics.bins < 1 or kinematics.readout_min_baseline < 1 or kinematics.ratio_min_baseline < 1:
                raise ValueError(f"kinematics: need bins, readout_min_baseline, and ratio_min_baseline >= 1, got "
                                 f"{kinematics.bins}, {kinematics.readout_min_baseline}, and "
                                 f"{kinematics.ratio_min_baseline}")
            ratio = kinematics.other_types_ratio_range
            if len(ratio) != 2 or not 0 <= ratio[0] < 1 < ratio[1]:
                raise ValueError(f"kinematics.other_types_ratio_range: need [low, high] with 0 <= low < 1 < high, "
                                 f"got {ratio}")
        # With from_file the vectors come from the file, and `steer steering-vectors` checks the block there.
        from_gradients = self.steering_vectors.from_file is None
        if self.steering is not None:
            if from_gradients and self.steering.block not in self.gradients.blocks:
                raise ValueError(f"steering.block: block {self.steering.block} is not in gradients.blocks")
            if 0 in self.steering.alphas:
                raise ValueError("steering.alphas: leave out 0; the baseline always runs")
            steps = self.steering.steps
            if steps != "all" and (len(steps) != 2 or not 0 <= steps[0] <= steps[1] < self.sampler.steps):
                raise ValueError(f"steering.steps: need 0 <= first <= last < {self.sampler.steps}, got {list(steps)}")
        if self.ablation is not None:
            if self.dataset("steering").pair is None:
                raise ValueError(f"ablation: the steering sample {self.events.steering_dataset!r} has no "
                                 f"pairs, and the ablation edits the pair's tokens")
            if from_gradients and self.ablation.block not in self.gradients.blocks:
                raise ValueError(f"ablation.block: block {self.ablation.block} is not in gradients.blocks")
            if not self.ablation.seeds or len(set(self.ablation.seeds)) != len(self.ablation.seeds):
                raise ValueError(f"ablation.seeds: need at least one seed and no repeats, got {self.ablation.seeds}")
            peak = self.ablation.peak_range
            if len(peak) != 2 or not 0 < peak[0] < peak[1] or self.ablation.peak_bins < 1:
                raise ValueError(f"ablation.peak_range, peak_bins: need [low, high] with 0 < low < high and at least "
                                 f"1 bin, got {peak} and {self.ablation.peak_bins}")
            inset, ratio = self.ablation.figure.inset_range, self.ablation.figure.ratio_range
            if len(inset) != 2 or not peak[0] <= inset[0] < inset[1] <= peak[1]:
                raise ValueError(f"ablation.figure.inset_range: need [low, high] with low < high inside "
                                 f"ablation.peak_range {peak}, got {inset}")
            if len(ratio) != 2 or not 0 <= ratio[0] < ratio[1]:
                raise ValueError(f"ablation.figure.ratio_range: need [low, high] with 0 <= low < high, got {ratio}")
        # The checks below guard settings that the run reads only after the generation.
        if self.readout is not None:
            spectrum = self.readout.spectrum_range
            if len(spectrum) != 2 or not 0 < spectrum[0] < spectrum[1]:
                raise ValueError(f"readout.spectrum_range: need [low, high] with 0 < low < high, got {spectrum}")
            if not spectrum[0] < self.readout.r60_max_mass <= spectrum[1]:
                raise ValueError(f"readout.r60_max_mass: need it inside readout.spectrum_range {spectrum}, "
                                 f"got {self.readout.r60_max_mass}")
        if self.lineshape is not None and (self.lineshape.half_width <= 0 or self.lineshape.bins <= 5):
            raise ValueError(f"lineshape: need half_width > 0 and more bins than the 5 fitted parameters, got "
                             f"half_width {self.lineshape.half_width} and bins {self.lineshape.bins}")
        for label, zoom in (self.plots.zooms if self.plots is not None else {}).items():
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
