"""Read a run config (configs/paper/*.yaml) into dataclasses.

Every key of the YAML file must be a field of the matching dataclass, and every field without a default must be in
the file. An unknown key, a missing key, or a value outside its listed options stops with a ConfigError.
The classes file is read too; its path is relative to the directory of the run config.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Classes:
    min_mass: float
    windows: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Events:
    split: Literal["validation"]
    pair: Literal["muon", "electron", "tau", "photon", "jet", "largeRjet"]
    order_seed: int
    n_A: int
    n_B: int


@dataclass(frozen=True)
class Sampler:
    steps: int
    batch_size: int
    precision: Literal["float32"]


@dataclass(frozen=True)
class Gradients:
    blocks: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class Directions:
    method: Literal["fkt", "lda", "meandiff"]
    in_time: Literal["per_step", "constant"]
    balance: Literal["equal", "by_count"]


@dataclass(frozen=True)
class Steering:
    block: int
    alphas: tuple[float, ...]
    tokens: Literal["pair", "minus", "plus", "all_of_type"]
    steps: Literal["all"] | tuple[int, int]
    norm_restore: bool
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Readout:
    r60_max_mass: float
    spectrum_bins: int


@dataclass(frozen=True)
class Config:
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


def load_config(path: str | Path) -> Config:
    """Read the run config at `path` and the classes file it names."""
    path = Path(path)
    raw = _read_yaml(path)
    if not isinstance(raw.get("classes"), str):
        raise ConfigError(f"{path}: 'classes' must be the path of a classes file")
    raw["classes"] = _read_yaml(path.parent / raw["classes"])
    config = _build(Config, raw, "config")
    _check(config)
    return config


def to_dict(config: Config) -> dict[str, Any]:
    """Plain dict of the config, to save next to the outputs of a run."""
    return _plain(dataclasses.asdict(config))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"no such file: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    missing = fields - set(data)
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    return cls(**{k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()})


def _convert(hint: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value
    if args and type(None) not in args and origin not in (tuple, dict):  # a union such as Literal["all"] | tuple
        for option in args:
            try:
                return _convert(option, value, where)
            except ConfigError:
                pass
        raise ConfigError(f"{where}: {value!r} matches none of the allowed forms")
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} entries, got {len(value)}")
        return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return {_convert(args[0], k, where): _convert(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {hint!r}")


def _check(config: Config) -> None:
    """Checks that involve more than one key."""
    for name, (lo, hi) in config.classes.windows.items():
        if not config.classes.min_mass <= lo < hi:
            raise ConfigError(f"classes.windows.{name}: need min_mass <= low < high, got [{lo}, {hi}]")
    edges = sorted(config.classes.windows.values())
    for (_, hi), (lo, _) in zip(edges, edges[1:]):
        if lo < hi:
            raise ConfigError("classes.windows: windows overlap")
    if "rest" in config.classes.windows:
        raise ConfigError("classes.windows: 'rest' is the name of the class outside every window")
    if config.target not in config.classes.windows:
        raise ConfigError(f"target: {config.target!r} is not a class window {sorted(config.classes.windows)}")
    for key in ("order_seed", "n_A", "n_B"):
        if getattr(config.events, key) < 0:
            raise ConfigError(f"events.{key}: must not be negative")
    if config.sampler.steps < 1 or config.sampler.batch_size < 1:
        raise ConfigError("sampler: steps and batch_size must be positive")
    if not config.gradients.blocks or len(set(config.gradients.blocks)) != len(config.gradients.blocks):
        raise ConfigError("gradients.blocks: need at least one block, each once")
    if config.steering.block not in config.gradients.blocks:
        raise ConfigError(f"steering.block: block {config.steering.block} is not in gradients.blocks")
    if 0.0 in config.steering.alphas:
        raise ConfigError("steering.alphas: leave out 0; the baseline always runs")
    if not config.steering.seeds:
        raise ConfigError("steering.seeds: need at least one seed")
    steps = config.steering.steps
    if steps != "all" and not 0 <= steps[0] <= steps[1] < config.sampler.steps:
        raise ConfigError(f"steering.steps: need 0 <= first <= last < {config.sampler.steps}, got {list(steps)}")
    if config.events.pair in ("photon", "jet", "largeRjet") and config.steering.tokens in ("minus", "plus"):
        raise ConfigError(f"steering.tokens: {config.steering.tokens!r} needs a charged pair")


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x
