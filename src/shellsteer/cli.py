"""Command-line runner: `steer <stage> CONFIG [--set key=value]...`, or `steer all CONFIG` for the run of the config.

A config runs one of three chains of stages (STAGES, `Config.chain`): the steering run (Sec. 5); the kinematics run
(Figs. 4 and 14), whose steering sample holds events of every final state; or, with an `ablation` section, the mean
ablation (Sec. 5.3) of the steering sample, with xi and c_k (Eq. 25) from the directions sample. A run writes to
<runs>/<config name>/ (README, "Run the full chain"). Each stage reads only the files of the stages it needs and
stops, naming the missing file and the stage that writes it, if one is absent. A stage refuses to replace its own
output unless --overwrite is given, and refuses to run on outputs that were made with other settings. The resolved
config, overrides applied, goes to config.yaml in the run folder; the log to stdout.

With steering_vectors.from_file, the vectors come from a file of an earlier `steer steering-vectors` (stages_of):
`steer steering-vectors` checks the file against the config and copies it into the run folder, the steering and
kinematics runs have no gradients stage, and in the ablation run `steer gradients` makes only the baseline and the kept
pairs of the directions sample, which c_k needs. S (Eq. 21) then comes from the file, or the readout reports it as
unavailable.

The readouts write JSON: a few numbers per seed, which a person can read and any language can load. The lineshape
fit is a model of the Z (App. E) and runs only when the target is Z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import types
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Union, get_args, get_origin

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ValidationError

from shellsteer.activation_means import (activation_means, load_activation_means, mean_projection,
                                         save_activation_means)
from shellsteer.config import Config, load_config
from shellsteer.devices import (DeviceError, Devices, Workers, describe_devices, devices_metadata, load_workers,
                                parse_devices)
from shellsteer.events import (PAIR_TYPES, PARTICLE_TYPES, class_names, class_of, load_events, load_final_states,
                               select, tokens_of_type)
from shellsteer.gradients import gradient_moments, load_moments, save_moments
from shellsteer.kinematics import VARIABLES, largest_changes, load_panels, save_panels, type_masks, type_panels
from shellsteer.lineshape import LineshapeFit, fit_masses
from shellsteer.metrics import (below_mass_deviation, log_bin_edges, peak_height_change, pole_response_fit, r60,
                                selectivity_curve, target_shift, window_log_spread)
from shellsteer.plots import (ablation_figure, all_particles_figure, kinematics_figure, pole_and_width_figure, save,
                              selectivity_figure, steered_spectrum_figure)
from shellsteer.sampler import NanError
from shellsteer.steered_generation import (ABLATION_METADATA_KEYS, Ablation, Edit, ablated_path, generate_events,
                                           generate_pairs, generated_path, kept_path, load_generated,
                                           load_generated_events, load_kept, save_generated, save_generated_events,
                                           save_kept)
from shellsteer.steering_vectors import (check_steering_vectors_file, gradient_response,
                                        gradient_responses_per_step, load_gradient_response_per_step,
                                        load_steering_vector, save_steering_vectors, steering_vectors)

CONFIG_FILE = "config.yaml"
MOMENTS_FILE = "gradient_moments.npz"
DIRECTIONS_KEPT_FILE = "directions_kept.npz"
MEANS_FILE = "activation_means.npz"
ABLATED_DIR = "ablated"
ABLATED_KEPT_FILE = "ablated/kept.npz"   # the kept pairs of the steering sample on the first seed's baseline
ABLATION_READOUT_FILE = "ablation_readout.json"
VECTORS_FILE = "steering_vectors.npz"
GENERATED_DIR = "generated"
READOUT_FILE = "readout.json"
PLOTS_DIR = "plots"
ABLATION_FIGURE_FILE = "plots/ablation.pdf"
KINEMATICS_FILE = "kinematics.npz"
KINEMATICS_READOUT_FILE = "kinematics_readout.json"
KINEMATICS_FIGURE_FILE = "plots/kinematics.pdf"           # paper Fig. 4
ALL_PARTICLES_FIGURE_FILE = "plots/all_particles.pdf"     # paper Fig. 14
LINESHAPE_TARGET = "Z"            # the lineshape model of App. E describes the Z only


class StageError(Exception):
    """A run cannot go on; the message says what to do."""


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


# ---------------------------------------------------------------- config and overrides

def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """The config with each `key=value` set at its dotted path, validated again by the pydantic model.

    The value is read as YAML, so `1024` is an int, `[2041, 2042]` a list, and `all` a string. Every key but the last
    of the path must exist.
    """
    raw = config.model_dump()
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep or not key:
            raise StageError(f"--set {override!r}: expected key=value, such as sampler.batch_size=1024")
        *parents, leaf = key.split(".")
        node: Any = raw
        for depth, part in enumerate(parents):
            if node is None:
                raise StageError(f"--set {key}: {'.'.join(parents[:depth])} is not in this config")
            if not isinstance(node, dict) or part not in node:
                where = ".".join(parents[:depth]) or "the top level"
                options = sorted(node) if isinstance(node, dict) else []
                raise StageError(f"--set {key}: {where} has no key {part!r}; keys there: {options}")
            node = node[part]
        if node is None:
            raise StageError(f"--set {key}: {'.'.join(parents)} is not in this config")
        if not isinstance(node, dict):
            raise StageError(f"--set {key}: {'.'.join(parents)} is a value, not a section")
        node[leaf] = yaml.safe_load(value)
    return Config.model_validate(raw)


def read_config(path: str | Path, overrides: list[str]) -> Config:
    """The run config at `path` with the overrides applied; stops if the run name is not a plain folder name."""
    config = apply_overrides(load_config(path), overrides)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", config.name) or config.name in (".", ".."):
        raise StageError(f"name: {config.name!r} is not a plain folder name (letters, digits, _ . + -)")
    return config


def with_defaults(raw: Any, model: type[BaseModel]) -> Any:
    """`raw`, a dump of `model`, with each key it lacks set to the key's default in `model`, in nested sections too.

    A config.yaml written before a key with a default existed then compares equal to a config that keeps the default.
    So a key added later must default to what the code did before it, as sampler.on_nan does.
    """
    if not isinstance(raw, dict):
        return raw
    filled = dict(raw)
    for name, field in model.model_fields.items():
        if name not in filled:
            if not field.is_required():
                filled[name] = field.get_default(call_default_factory=True)
            continue
        options = get_args(field.annotation) if get_origin(field.annotation) in (Union, types.UnionType) else ()
        section = [t for t in (field.annotation, *options) if isinstance(t, type) and issubclass(t, BaseModel)]
        if section:
            filled[name] = with_defaults(filled[name], section[0])
    return filled


def differing_keys(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Dotted paths at which two config dumps differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        keys = []
        for key in sorted(set(old) | set(new), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            keys += differing_keys(old.get(key), new.get(key), path) if key in old and key in new else [path]
        return keys
    return [] if old == new else [prefix]


# ---------------------------------------------------------------- stages

def require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise StageError(f"{path} is missing; run `steer {stage}` first")
    return path


def require_generated(config: Config, run: Path) -> None:
    require(kept_path(run), "generate")
    for seed in config.steering.seeds:
        for alpha in (0.0, *config.steering.alphas):
            require(generated_path(run, seed, alpha), "generate")


def start_workers(stage: str, config: Config, devices: Devices) -> Workers:
    """Log the devices of the stage and load the network once on each."""
    for line in describe_devices(devices):
        log(stage, line)
    return load_workers(config.networks[config.network], devices)


def class_counts(kept: np.ndarray, names: list[str]) -> str:
    return ", ".join(f"{name} {int(np.sum(kept == name)):,}" for name in names)


def run_gradients(config: Config, run: Path, devices: Devices) -> None:
    """Gradient pass on the directions sample (paper Sec. 4.1): the per-class sums of gradients.py."""
    stage = "gradients"
    entry = config.datasets[config.events.dataset]
    workers = start_workers(stage, config, devices)
    directions = load_events(entry, "directions", config.events.directions)
    log(stage, f"{len(directions):,} events of the directions sample")
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, on_nan=config.sampler.on_nan)
    start = time.perf_counter()
    baseline = generate_pairs(workers, directions, None, seed=config.gradients.seed, **sizes)
    kept = select(baseline.mass, directions.priority, config.classes, config.events.keep)
    names = class_names(config.classes)
    log(stage, f"baseline of seed {config.gradients.seed} in {time.perf_counter() - start:.1f} s; kept "
               f"{config.events.keep:,} pairs: {class_counts(kept, names)}")
    log(stage, f"baseline of seed {config.gradients.seed}: {baseline.nans.text()}")
    kept_metadata = {"seed": config.gradients.seed, "keep": config.events.keep, "sample": "directions",
                     "device": devices_metadata(devices), "NaN counts": baseline.nans.metadata()}
    if config.steering_vectors.from_file is not None:
        save_kept(run / DIRECTIONS_KEPT_FILE, kept, directions.event, kept_metadata)
        log(stage, f"wrote {run / DIRECTIONS_KEPT_FILE}; no gradient pass: the vectors come from "
                   f"steering_vectors.from_file {config.steering_vectors.from_file}")
        return
    start = time.perf_counter()
    moments, nans = gradient_moments(workers, directions, kept, config.classes, blocks=list(config.gradients.blocks),
                                     seed=config.gradients.seed, **sizes)
    counts = ", ".join(f"{name} {int(n):,}" for name, n in zip(moments["classes"], moments["count"]))
    log(stage, f"gradient pass in {time.perf_counter() - start:.1f} s; pairs per class: {counts}; "
               f"non-finite pairs {int(moments['non-finite pairs']):,}")
    log(stage, f"gradient pass: {nans.text()}")
    metadata = {"steps": config.sampler.steps, "batch_size": config.sampler.batch_size,
                "seed": config.gradients.seed, "blocks": list(config.gradients.blocks), "keep": config.events.keep,
                "events": config.events.directions, "device": devices_metadata(devices),
                "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump(),
                "NaN counts": nans.metadata(), "NaN counts of the baseline": baseline.nans.metadata()}
    save_moments(run / MOMENTS_FILE, moments, metadata)
    log(stage, f"wrote {run / MOMENTS_FILE}")
    save_kept(run / DIRECTIONS_KEPT_FILE, kept, directions.event, kept_metadata)
    log(stage, f"wrote {run / DIRECTIONS_KEPT_FILE}")


def edited_blocks(config: Config) -> list[int]:
    """The blocks the run edits: steering.block, or ablation.block for the mean ablation."""
    return [config.steering.block if config.ablation is None else config.ablation.block]


def run_steering_vectors(config: Config, run: Path, devices: Devices) -> None:
    """Steering vectors of every class and block (paper Sec. 4.2): from the class sums, or copied from
    steering_vectors.from_file after a check against the config."""
    stage = "steering-vectors"
    method = config.steering_vectors
    settings = {"method": method.method, "in_time": method.in_time, "balance": method.balance}
    if method.from_file is not None:
        source = Path(method.from_file)
        try:
            unrecorded = check_steering_vectors_file(
                source, **settings, classes=config.classes.model_dump(), blocks=edited_blocks(config),
                steps=config.sampler.steps, network_name=config.network,
                network=config.networks[config.network].model_dump())
        except ValueError as error:
            raise StageError(f"steering_vectors.from_file: {error}") from error
        for what in unrecorded:
            log(stage, f"{source} does not record {what}; not checked")
        shutil.copyfile(source, run / VECTORS_FILE)
        digest = hashlib.md5((run / VECTORS_FILE).read_bytes()).hexdigest()
        log(stage, f"{method.method}, {method.in_time}, balance {method.balance}: copied {source} (md5 {digest}); "
                   f"no gradient pass in this run")
        log(stage, f"wrote {run / VECTORS_FILE}")
        return
    moments, moments_metadata = load_moments(require(run / MOMENTS_FILE, "gradients"))
    vectors = steering_vectors(moments, method.method, method.in_time, method.balance)
    # S as the edit applies xi, in float32 (load_steering_vector).
    responses = gradient_responses_per_step(moments, vectors.astype(np.float32))
    gradient_pass = {key: moments_metadata[key] for key in ("dataset", "events", "keep", "seed", "blocks",
                                                            "batch_size", "device")}
    save_steering_vectors(run / VECTORS_FILE, vectors, moments["classes"], moments["blocks"], {
        **settings, "gradient moments file": MOMENTS_FILE, "steps": moments_metadata["steps"],
        "classes": config.classes.model_dump(), "network": moments_metadata["network"],
        "gradient pass": gradient_pass}, response_per_step=responses)
    log(stage, f"{method.method}, {method.in_time}, balance {method.balance}: vectors of shape {vectors.shape} "
               f"(classes, blocks, steps, 2, width)")
    target = [str(name) for name in moments["classes"]].index(config.target)
    log(stage, f"gradient response S of the {config.target} vector (Eq. 21): " + ", ".join(
        f"block {int(block)} {responses[target, b].sum():.4f}" for b, block in enumerate(moments["blocks"])))
    log(stage, f"wrote {run / VECTORS_FILE}")


def run_generate(config: Config, run: Path, devices: Devices) -> None:
    """Baseline and steered generation of the steering sample (paper Sec. 5, Eq. 8), every seed and alpha.

    A steering sample of pairs keeps the mass and four-momenta of each pair, and the kept pairs of the first seed's
    baseline. A steering sample of every final state (Fig. 4) keeps every particle of every event and has no kept
    pairs.
    """
    stage = "generate"
    alphas = (0.0, *config.steering.alphas)
    # Every output path first: generated_path stops on an alpha whose file name would clash, before any generation.
    paths = {(seed, alpha): generated_path(run, seed, alpha) for seed in config.steering.seeds for alpha in alphas}
    xi = load_steering_vector(require(run / VECTORS_FILE, "steering-vectors"), config.target, config.steering.block)
    entry, edited = config.dataset("steering"), config.dataset("directions").pair
    pairs = entry.pair is not None
    workers = start_workers(stage, config, devices)
    if pairs:
        events = load_events(entry, "steering", config.events.steering)
    else:
        events = load_final_states(entry, config.events.steering)
        negative, positive = (mask.sum(1) for mask in tokens_of_type(events.batch, edited))
        per_event = (negative + positive).bincount()
        log(stage, f"edited {edited}s per event: " + ", ".join(f"{k}: {int(n):,}" for k, n in enumerate(per_event))
                   + f"; {int(negative.sum()):,} negative, {int(positive.sum()):,} positive")
    log(stage, f"{len(events):,} events of the steering sample; {config.target} vector at block "
               f"{config.steering.block}; seeds {list(config.steering.seeds)}, alphas {list(alphas)}")
    edit_steps = None if config.steering.steps == "all" else tuple(config.steering.steps)
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, on_nan=config.sampler.on_nan)
    for seed in config.steering.seeds:
        for alpha in alphas:
            edit = None if alpha == 0 else Edit(xi=xi, alpha=alpha, block=config.steering.block,
                                                tokens=config.steering.tokens, pair=edited, steps=edit_steps,
                                                norm_restore=config.steering.norm_restore)
            start = time.perf_counter()
            generated = (generate_pairs if pairs else generate_events)(workers, events, edit, seed=seed, **sizes)
            seconds = time.perf_counter() - start
            log(stage, f"seed {seed}, alpha {alpha:+g}: {generated.nans.text()}")
            if pairs and seed == config.steering.seeds[0] and alpha == 0:
                kept = select(generated.mass, events.priority, config.classes, config.events.keep)
                save_kept(kept_path(run), kept, generated.event, {"seed": seed, "keep": config.events.keep})
                log(stage, f"kept {config.events.keep:,} pairs on the baseline of seed {seed}: "
                           f"{class_counts(kept, class_names(config.classes))}; wrote {kept_path(run)}")
            path = paths[(seed, alpha)]
            (save_generated if pairs else save_generated_events)(path, generated, {
                "alpha": alpha, "block": config.steering.block, "target": config.target,
                "method": config.steering_vectors.method, "seed": seed, "steps": config.sampler.steps,
                "edited steps": None if edit_steps is None else list(edit_steps), "tokens": config.steering.tokens,
                "norm_restore": config.steering.norm_restore, "steering vector file": VECTORS_FILE,
                "batch_size": config.sampler.batch_size, "device": devices_metadata(devices),
                "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump()})
            log(stage, f"seed {seed}, alpha {alpha:+g}: {len(generated.event):,} {'pairs' if pairs else 'events'} "
                       f"in {seconds:.1f} s; wrote {path}")


def fit_summary(fit: LineshapeFit) -> dict[str, Any]:
    """The fitted parameters and fit quality of one lineshape fit, without its histogram."""
    summary = {key: value for key, value in asdict(fit).items() if key not in ("edges", "counts", "expected")}
    summary["at_bound"] = list(fit.at_bound)
    return summary


def counted_pairs(run: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The class of each pair and the mask of the kept pairs; stops if the files of a seed hold other events."""
    kept, event, _ = load_kept(kept_path(run))
    baseline, _ = load_generated(generated_path(run, seed, 0.0))
    if not np.array_equal(baseline.event, event):
        raise StageError(f"{generated_path(run, seed, 0.0)} holds other events than {kept_path(run)}; rerun "
                         f"`steer generate --overwrite`")
    return kept, kept != ""


def first_order_slope(config: Config, run: Path) -> tuple[float | None, str]:
    """S (Eq. 21) of the target's vector at steering.block and the file it comes from; or None and the reason.

    S is the sum over the Euler steps of the response per step that `steer steering-vectors` stores in the vectors
    file. For a vectors file without it, S comes from the run's gradient_moments.npz if the run made its own
    vectors; a run started from a file has no moments.
    """
    vectors_file = require(run / VECTORS_FILE, "steering-vectors")
    per_step = load_gradient_response_per_step(vectors_file, config.target, config.steering.block)
    if per_step is not None:
        return float(per_step.sum()), VECTORS_FILE
    from_file = config.steering_vectors.from_file
    if from_file is None:
        moments, _ = load_moments(require(run / MOMENTS_FILE, "gradients"))
        xi = load_steering_vector(vectors_file, config.target, config.steering.block)
        return gradient_response(moments, xi, config.target, config.steering.block), MOMENTS_FILE
    return None, (f"steering_vectors.from_file {from_file} stores no gradient response, and the run has no "
                  f"gradient pass")


def run_readout(config: Config, run: Path, devices: Devices) -> None:
    """R_60, the target shift, the selectivity curve, the response fits, and the Z lineshape fits (Sec. 5, App. E)."""
    stage = "readout"
    s, s_source = first_order_slope(config, run)
    require_generated(config, run)
    seeds, alphas = list(config.steering.seeds), list(config.steering.alphas)
    readout, classes = config.readout, config.classes
    edges = log_bin_edges(*readout.spectrum_range, readout.spectrum_bins)
    fit_lineshape = config.target == LINESHAPE_TARGET
    if not fit_lineshape:
        log(stage, f"no lineshape fit: the model of App. E is the {LINESHAPE_TARGET}'s, the target is {config.target}")
    fit_settings = dict(peak_window=tuple(classes.windows[config.target]), half_width=config.lineshape.half_width,
                        bins=config.lineshape.bins, min_window_pairs=config.lineshape.min_window_pairs)

    rows, fits = [], {}
    for seed in seeds:
        kept, counted = counted_pairs(run, seed)
        base = load_generated(generated_path(run, seed, 0.0))[0].mass[counted]
        in_target = class_of(base, classes) == config.target
        log(stage, f"seed {seed}: {int(counted.sum()):,} kept pairs, {int(in_target.sum()):,} in {config.target} "
                   f"on this seed's baseline")
        if fit_lineshape:
            fits[(seed, 0.0)] = fit_masses(base, base, None, **fit_settings)
            rows.append({"seed": seed, "alpha": 0.0, "target shift": None, "R_60": None,
                         "lineshape": fit_summary(fits[(seed, 0.0)])})
        for alpha in alphas:
            steered = load_generated(generated_path(run, seed, alpha))[0].mass[counted]
            row = {"seed": seed, "alpha": alpha, "target shift": target_shift(steered, base, in_target),
                   "R_60": r60(steered, base, edges=edges, max_mass=readout.r60_max_mass,
                               min_baseline_pairs=readout.r60_min_baseline_pairs), "lineshape": None}
            if fit_lineshape:
                fits[(seed, alpha)] = fit_masses(steered, base, fits[(seed, 0.0)], **fit_settings)
                row["lineshape"] = fit_summary(fits[(seed, alpha)])
            rows.append(row)
            log(stage, f"seed {seed}, alpha {alpha:+g}: target shift {row['target shift']:+.4f} in log m, "
                       f"R_60 {row['R_60']:.4f}")
    for (seed, alpha), fit in fits.items():
        log(stage, f"seed {seed}, alpha {alpha:+g}: pole {fit.pole:.3f} GeV, width {fit.width:.3f} GeV, "
                   f"{fit.pairs_in_window:,.0f} pairs in the window" + (f"; AT BOUND {list(fit.at_bound)}"
                                                                      if fit.at_bound else ""))

    steered_rows = [row for row in rows if row["alpha"] != 0]
    curve = selectivity_curve(*(np.array([row[key] for row in steered_rows])
                                for key in ("seed", "alpha", "target shift", "R_60")))
    result: dict[str, Any] = {
        "target": config.target, "block": config.steering.block, "method": config.steering_vectors.method,
        "seeds": seeds, "alphas": [0.0, *alphas],
        "kept pairs": {name: int(np.sum(kept == name)) for name in class_names(classes)},
        "gradient response": s, "gradient response source": s_source if s is not None else None,
        "gradient response unavailable": s_source if s is None else None,
        "selectivity curve": {key: value.tolist() for key, value in curve.items()},
        "per seed and alpha": rows, "lineshape over seeds": None, "pole response fit": None}
    if s is None:
        log(stage, f"gradient response S (Eq. 21) UNAVAILABLE: {s_source}; readout.json has null")
    else:
        log(stage, f"gradient response S = {s:.4f} (Eq. 21, first order), from {s_source}")
    if fit_lineshape:
        grid = sorted({0.0, *alphas})
        poles = np.array([[fits[(seed, alpha)].pole for seed in seeds] for alpha in grid])
        widths = np.array([[fits[(seed, alpha)].width for seed in seeds] for alpha in grid])
        result["lineshape over seeds"] = {"alpha": grid, "pole": poles.mean(1).tolist(),
                                          "pole std": poles.std(1).tolist(), "width": widths.mean(1).tolist(),
                                          "width std": widths.std(1).tolist()}
        if len(alphas) >= 2:
            s_fit, r2 = pole_response_fit(np.array(grid), poles.mean(1))
            result["pole response fit"] = {"S_fit": s_fit, "R^2": r2}
            log(stage, f"pole response fit of the seed-mean poles: S_fit = {s_fit:.4f}, R^2 = {r2:.4f}")
        else:
            log(stage, "no pole response fit: it needs at least two alphas other than 0")
    (run / READOUT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    log(stage, f"wrote {run / READOUT_FILE}")


def real_masses(config: Config, sample: Literal["directions", "steering"], event: np.ndarray, kept_file: Path,
                rerun: str) -> np.ndarray:
    """(E,) m_real of the events of `sample`, in the order of `event`, the events of `kept_file`. Stops if the
    sample holds other events; `rerun` names the stage that wrote `kept_file`."""
    events = load_events(config.dataset(sample), sample, getattr(config.events, sample))
    if not np.array_equal(events.event, event):
        raise StageError(f"the {sample} sample holds other events than {kept_file}; rerun `steer {rerun} "
                         f"--overwrite`")
    return events.m_real


def run_plots(config: Config, run: Path, devices: Devices) -> None:
    """The steered spectrum with real data (Fig. 3) and the selectivity curve, from the first seed; pole and width
    (Fig. 5) for Z."""
    stage = "plots"
    result = json.loads(require(run / READOUT_FILE, "readout").read_text())
    require_generated(config, run)
    seed = config.steering.seeds[0]
    _, counted = counted_pairs(run, seed)
    masses = {alpha: load_generated(generated_path(run, seed, alpha))[0].mass[counted]
              for alpha in (0.0, *config.steering.alphas)}
    real = real_masses(config, "steering", load_kept(kept_path(run))[1], kept_path(run), "generate")[counted]
    plots, readout = config.plots, config.readout
    folder = run / PLOTS_DIR
    folder.mkdir()
    figures = {
        "steered_spectrum.pdf": steered_spectrum_figure(
            masses.pop(0.0), masses, real, edges=log_bin_edges(*readout.spectrum_range, readout.spectrum_bins),
            zooms={label: tuple(zoom) for label, zoom in plots.zooms.items()}, zoom_bins=plots.zoom_bins,
            baseline_label=plots.baseline_label, pair_label=plots.pair_label),
        "selectivity.pdf": selectivity_figure({config.steering_vectors.method: result["selectivity curve"]},
                                              target_label=plots.target_label, pair_label=plots.pair_label,
                                              max_mass=readout.r60_max_mass)}
    over_seeds = result["lineshape over seeds"]
    if over_seeds is not None and result["gradient response"] is None:
        log(stage, f"pole_and_width.pdf has no first-order line: no gradient response S "
                   f"({result['gradient response unavailable']})")
    if over_seeds is None:
        log(stage, f"no pole and width figure: no lineshape fit for target {config.target}")
    else:
        figures["pole_and_width.pdf"] = pole_and_width_figure(
            over_seeds["alpha"], over_seeds["pole"], over_seeds["pole std"], over_seeds["width"],
            over_seeds["width std"], first_order_log_slope=result["gradient response"])
    log(stage, f"spectrum from seed {seed}, {int(counted.sum()):,} kept pairs")
    for name, figure in figures.items():
        save(figure, folder / name)
        log(stage, f"wrote {folder / name}")


def run_activation_means(config: Config, run: Path, devices: Devices) -> None:
    """Mean of the residual stream on the pair's tokens over the kept pairs of the directions sample, the input of
    c_k (paper Eq. 25): one forward pass without edits with the noise of gradients.seed."""
    stage = "activation-means"
    kept_file = require(run / DIRECTIONS_KEPT_FILE, "gradients")
    kept, event, _ = load_kept(kept_file)
    entry = config.datasets[config.events.dataset]
    workers = start_workers(stage, config, devices)
    events = load_events(entry, "directions", config.events.directions)
    if not np.array_equal(events.event, event):
        raise StageError(f"the directions sample holds other events than {kept_file}; rerun `steer gradients "
                         f"--overwrite`")
    log(stage, f"{len(events):,} events of the directions sample; block {config.ablation.block}, "
               f"seed {config.gradients.seed}")
    start = time.perf_counter()
    means, nans = activation_means(workers, events, kept, config.classes, blocks=[config.ablation.block],
                                   steps=config.sampler.steps, batch_size=config.sampler.batch_size,
                                   seed=config.gradients.seed, on_nan=config.sampler.on_nan)
    counts = ", ".join(f"{name} {int(n):,}" for name, n in zip(means["classes"], means["count"]))
    log(stage, f"forward pass in {time.perf_counter() - start:.1f} s; pairs per class: {counts} (total "
               f"{int(np.sum(means['count'])):,}); non-finite pairs {int(means['non-finite pairs']):,}")
    log(stage, nans.text())
    save_activation_means(run / MEANS_FILE, means, {
        "steps": config.sampler.steps, "batch_size": config.sampler.batch_size, "seed": config.gradients.seed,
        "blocks": [config.ablation.block], "kept pairs file": DIRECTIONS_KEPT_FILE,
        "device": devices_metadata(devices),
        "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump(),
        "NaN counts": nans.metadata()})
    log(stage, f"wrote {run / MEANS_FILE}")


def run_ablate(config: Config, run: Path, devices: Devices) -> None:
    """Baseline and mean-ablated generation (paper Eq. 24) of the steering sample for every seed, with xi and c_k
    (Eq. 25) from the directions sample."""
    stage = "ablate"
    ablation = config.ablation
    blocks = (None, ablation.block)                                    # None: the baseline
    paths = {(seed, block): ablated_path(run, seed, block) for seed in ablation.seeds for block in blocks}
    xi = load_steering_vector(require(run / VECTORS_FILE, "steering-vectors"), config.target, ablation.block)
    means, _ = load_activation_means(require(run / MEANS_FILE, "activation-means"))
    c = mean_projection(means, xi, ablation.block)
    log(stage, f"c_k from {int(np.sum(means['count'])):,} pairs: {c.min():+.4f} to {c.max():+.4f} over "
               f"{len(c)} steps")
    entry = config.dataset("steering")
    events = load_events(entry, "steering", config.events.steering)
    workers = start_workers(stage, config, devices)
    log(stage, f"{len(events):,} events of the steering sample; {config.target} vector at "
               f"block {ablation.block}; seeds {list(ablation.seeds)}")
    edit = Ablation(xi=xi, mean=torch.from_numpy(c), block=ablation.block)
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, on_nan=config.sampler.on_nan)
    for seed in ablation.seeds:
        for block in blocks:
            start = time.perf_counter()
            generated = generate_pairs(workers, events, None if block is None else edit, seed=seed, **sizes)
            seconds = time.perf_counter() - start
            log(stage, f"seed {seed}, {'baseline' if block is None else f'block {block}'}: {generated.nans.text()}")
            if seed == ablation.seeds[0] and block is None:
                kept = select(generated.mass, events.priority, config.classes, config.events.keep)
                save_kept(run / ABLATED_KEPT_FILE, kept, generated.event, {"seed": seed, "keep": config.events.keep,
                                                                            "sample": "steering"})
                log(stage, f"kept {config.events.keep:,} pairs on the baseline of seed {seed}: "
                           f"{class_counts(kept, class_names(config.classes))}; wrote {run / ABLATED_KEPT_FILE}")
            path = paths[(seed, block)]
            save_generated(path, generated, {
                "block": block, "target": config.target, "method": config.steering_vectors.method, "seed": seed,
                "steps": config.sampler.steps, "sample": "steering", "steering vector file": VECTORS_FILE,
                "activation means file": MEANS_FILE, "c_k": c.tolist(), "batch_size": config.sampler.batch_size,
                "device": devices_metadata(devices), "network": config.networks[config.network].model_dump(),
                "dataset": entry.model_dump()}, required=ABLATION_METADATA_KEYS)
            what = "baseline" if block is None else f"ablated at block {block}"
            log(stage, f"seed {seed}, {what}: {len(generated.mass):,} pairs in {seconds:.1f} s; wrote {path}")


def ablated_masses(config: Config, run: Path, seed: int, kept: np.ndarray, kept_event: np.ndarray,
                   kept_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Baseline and ablated masses of one seed, of the kept pairs with a class on this seed's own baseline mass, as
    in the paper. Stops if a file holds other events than `kept_file`."""
    block = config.ablation.block
    (base, _), (ablated, _) = (load_generated(ablated_path(run, seed, b)) for b in (None, block))
    for generated, b in ((base, None), (ablated, block)):
        if not np.array_equal(generated.event, kept_event):
            raise StageError(f"{ablated_path(run, seed, b)} holds other events than {kept_file}; rerun "
                             f"`steer ablate --overwrite`")
    counted = (kept != "") & (class_of(base.mass, config.classes) != "")
    return base.mass[counted], ablated.mass[counted]


def run_ablation_readout(config: Config, run: Path, devices: Devices) -> None:
    """The numbers of the mean ablation (paper Sec. 5.3, Fig. 7): spread of the target peak, change of its height,
    and the deviation below readout.r60_max_mass, per seed and over seeds (metrics.py)."""
    stage = "ablation-readout"
    ablation, readout, classes = config.ablation, config.readout, config.classes
    kept_file = run / ABLATED_KEPT_FILE
    kept, kept_event, _ = load_kept(require(kept_file, "ablate"))
    for seed in ablation.seeds:
        for block in (None, ablation.block):
            require(ablated_path(run, seed, block), "ablate")
    window = tuple(classes.windows[config.target])
    edges = log_bin_edges(*readout.spectrum_range, readout.spectrum_bins)
    peak_edges = np.linspace(*ablation.peak_range, ablation.peak_bins + 1)
    deviation_settings = dict(max_mass=readout.r60_max_mass, min_baseline_pairs=readout.r60_min_baseline_pairs)
    rows, histograms = [], {"spectrum": ([], []), "peak": ([], [])}       # (baseline, ablated) per seed
    for seed in ablation.seeds:
        b, a = ablated_masses(config, run, seed, kept, kept_event, kept_file)
        spread = (window_log_spread(b, window), window_log_spread(a, window))
        spectrum = (np.histogram(b, edges)[0], np.histogram(a, edges)[0])
        peak = (np.histogram(b, peak_edges)[0], np.histogram(a, peak_edges)[0])
        for key, pair in (("spectrum", spectrum), ("peak", peak)):
            for per_seed, counts in zip(histograms[key], pair):
                per_seed.append(counts)
        row = {"seed": seed, "pairs": len(b),
               "pairs in the target window": {"baseline": int(np.sum((b > window[0]) & (b < window[1]))),
                                              "ablated": int(np.sum((a > window[0]) & (a < window[1])))},
               "spread of log m in the target window": {"baseline": spread[0], "ablated": spread[1],
                                                        "ratio": spread[1] / spread[0]},
               "peak height change": peak_height_change(peak[1], peak[0]),
               "deviation below max mass": below_mass_deviation(spectrum[1], spectrum[0], edges=edges,
                                                                 **deviation_settings),
               "R_60": r60(a, b, edges=edges, max_mass=readout.r60_max_mass,
                           min_baseline_pairs=readout.r60_min_baseline_pairs)}
        rows.append(row)
        ratio = row["spread of log m in the target window"]["ratio"]
        log(stage, f"seed {seed}: {row['pairs']:,} pairs; spread x{ratio:.3f}, peak "
                   f"{100 * row['peak height change']:+.1f} %, below {readout.r60_max_mass:g} GeV "
                   f"{100 * row['deviation below max mass']:.2f} %, R_60 {row['R_60']:.4f}")
    mean = {key: (np.mean(base_list, axis=0), np.mean(ablated_list, axis=0))
            for key, (base_list, ablated_list) in histograms.items()}
    result = {
        "target": config.target, "block": ablation.block, "sample": "steering", "seeds": list(ablation.seeds),
        "target window": list(window), "kept pairs file": kept_file.name,
        "spread ratio, mean over seeds": float(np.mean([r["spread of log m in the target window"]["ratio"]
                                                        for r in rows])),
        "peak height change, seed-average histograms": peak_height_change(mean["peak"][1], mean["peak"][0]),
        "deviation below max mass, seed-average histograms": below_mass_deviation(
            mean["spectrum"][1], mean["spectrum"][0], edges=edges, **deviation_settings),
        "R_60, mean over seeds": float(np.mean([r["R_60"] for r in rows])),
        "per seed": rows}
    log(stage, f"over seeds: spread x{result['spread ratio, mean over seeds']:.3f}, peak "
               f"{100 * result['peak height change, seed-average histograms']:+.1f} %, below "
               f"{readout.r60_max_mass:g} GeV "
               f"{100 * result['deviation below max mass, seed-average histograms']:.2f} %, "
               f"R_60 {result['R_60, mean over seeds']:.4f}")
    (run / ABLATION_READOUT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    log(stage, f"wrote {run / ABLATION_READOUT_FILE}")


def run_ablation_plots(config: Config, run: Path, devices: Devices) -> None:
    """Paper Fig. 7: the spectrum of the baseline, the mean ablation, and real data, counts averaged over seeds."""
    stage = "ablation-plots"
    ablation, readout = config.ablation, config.readout
    kept_file = run / ABLATED_KEPT_FILE
    kept, kept_event, _ = load_kept(require(kept_file, "ablate"))
    pairs = [ablated_masses(config, run, seed, kept, kept_event, kept_file) for seed in ablation.seeds]
    real = real_masses(config, "steering", kept_event, kept_file, "ablate")[kept != ""]
    log(stage, f"seeds {list(ablation.seeds)}: {', '.join(f'{len(b):,}' for b, _ in pairs)} pairs; real data of the "
               f"{len(real):,} kept pairs")
    settings = ablation.figure
    figure = ablation_figure(
        [b for b, _ in pairs], [a for _, a in pairs], real,
        edges=log_bin_edges(*readout.spectrum_range, readout.spectrum_bins),
        inset_edges=np.linspace(*ablation.peak_range, ablation.peak_bins + 1),
        inset_range=tuple(settings.inset_range), ratio_range=tuple(settings.ratio_range),
        baseline_label=settings.baseline_label, pair_label=settings.pair_label, target_label=settings.target_label)
    (run / ABLATION_FIGURE_FILE).parent.mkdir()
    save(figure, run / ABLATION_FIGURE_FILE)
    log(stage, f"wrote {run / ABLATION_FIGURE_FILE}")


def run_kinematics_readout(config: Config, run: Path, devices: Devices) -> None:
    """Counts of p_T, eta, phi, and E of every particle type for real data, the baseline, and each alpha (Figs. 4 and
    14), and the largest change per bin that Sec. 5.1 and App. D.1 quote."""
    stage = "kinematics-readout"
    seed, alphas = config.steering.seeds[0], [0.0, *config.steering.alphas]
    paths = {alpha: require(generated_path(run, seed, alpha), "generate") for alpha in alphas}
    real = load_final_states(config.dataset("steering"), config.events.steering)
    generated = {alpha: load_generated_events(path)[0] for alpha, path in paths.items()}
    particle_type, particle_mask = real.batch["particle_type"].numpy(), real.batch["particle_mask"].numpy() > 0.5
    for alpha, events in generated.items():
        if not (np.array_equal(events.event, real.event) and np.array_equal(events.particle_type, particle_type)
                and np.array_equal(events.particle_mask, particle_mask)):
            raise StageError(f"{paths[alpha]} holds other events or particles than the steering sample; rerun "
                             f"`steer generate --overwrite`")
    masks = type_masks(particle_type, particle_mask, list(PARTICLE_TYPES))
    edited = config.dataset("directions").pair
    if not masks[PAIR_TYPES[edited]].any():
        raise StageError(f"the steering sample has no {edited}, so the edit changes nothing and Fig. 4 has no panel")
    absent = [PARTICLE_TYPES[code] for code, mask in masks.items() if not mask.any()]
    if absent:
        log(stage, f"no {', '.join(absent)} in the {len(real):,} events: left out of the counts and the figures")
    masks = {code: mask for code, mask in masks.items() if mask.any()}
    kinematics = config.kinematics
    panels = type_panels(real.p4, generated.pop(0.0).four_momenta,
                         {alpha: events.four_momenta for alpha, events in generated.items()}, masks,
                         bins=kinematics.bins)
    particles = {PARTICLE_TYPES[code]: {"particles": int(mask.sum()), "events": int(mask.any(1).sum())}
                 for code, mask in masks.items()} | {name: {"particles": 0, "events": 0} for name in absent}
    save_panels(run / KINEMATICS_FILE, panels, {"seed": seed, "alphas": alphas, "bins": kinematics.bins,
                                                "particle types": {str(c): n for c, n in PARTICLE_TYPES.items()},
                                                "particles": particles})
    log(stage, f"wrote {run / KINEMATICS_FILE}")

    minimum = kinematics.readout_min_baseline
    changes = {PARTICLE_TYPES[code]: {variable: {f"{alpha:+g}": change for alpha, change in
                                                 largest_changes(variable, panels[code][variable], minimum).items()}
                                      for variable in VARIABLES} for code in masks}
    others = {}
    for alpha in config.steering.alphas:
        found = [(change["largest change"], name, variable, change["bin centre"])
                 for name, by_variable in changes.items() if name != edited
                 for variable, by_alpha in by_variable.items() if (change := by_alpha[f"{alpha:+g}"]) is not None]
        largest = max(found) if found else None
        others[f"{alpha:+g}"] = None if largest is None else dict(zip(
            ("largest change", "particle type", "variable", "bin centre"), largest))
    result = {"seed": seed, "alphas": alphas, "bins": kinematics.bins, "edited type": edited,
              "min baseline particles in a bin": minimum, "particles": particles,
              "largest change per bin": changes, f"largest change per bin of the types other than the {edited}": others}
    (run / KINEMATICS_READOUT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    for variable in ("pt", "eta", "phi"):
        parts = [f"alpha {key}: " + ("no bin" if change is None else
                                     f"{change['largest change']:.4f} at {change['bin centre']:.3g}")
                 for key, change in changes[edited][variable].items()]
        log(stage, f"{edited} {variable}, largest |steered / baseline - 1| in bins with >= {minimum:,} baseline "
                   f"particles: " + "; ".join(parts))
    for key, change in others.items():
        log(stage, f"alpha {key}, every type but the {edited}: " + ("no bin" if change is None else
            f"{change['largest change']:.4f} ({change['particle type']} {change['variable']} at "
            f"{change['bin centre']:.3g})"))
    log(stage, f"wrote {run / KINEMATICS_READOUT_FILE}")


def run_kinematics_plots(config: Config, run: Path, devices: Devices) -> None:
    """Fig. 4, the edited type's p_T, eta, and phi, and Fig. 14, every other type, from the counts of
    kinematics-readout."""
    stage = "kinematics-plots"
    panels, _ = load_panels(require(run / KINEMATICS_FILE, "kinematics-readout"))
    kinematics, edited = config.kinematics, PAIR_TYPES[config.dataset("directions").pair]
    figures = {
        KINEMATICS_FIGURE_FILE: kinematics_figure(panels[edited], particle=edited,
                                                  baseline_label=kinematics.baseline_label,
                                                  ratio_min_baseline=kinematics.ratio_min_baseline),
        ALL_PARTICLES_FIGURE_FILE: all_particles_figure(
            {code: panels[code] for code in PARTICLE_TYPES if code != edited and code in panels},
            baseline_label=kinematics.baseline_label, ratio_min_baseline=kinematics.ratio_min_baseline,
            ratio_range=tuple(kinematics.other_types_ratio_range))}
    (run / PLOTS_DIR).mkdir()
    for name, figure in figures.items():
        save(figure, run / name)
        log(stage, f"wrote {run / name}")


Chain = Literal["steering", "kinematics", "ablation"]
CHAINS: tuple[Chain, ...] = ("steering", "kinematics", "ablation")


@dataclass(frozen=True)
class Stage:
    name: str
    chains: tuple[Chain, ...]          # the runs the stage belongs to
    needs: tuple[str, ...]             # stages whose outputs this stage reads
    outputs: tuple[str, ...]           # paths in the run folder that the stage writes
    sections: tuple[str, ...]          # config keys, dotted, that the stage reads beyond those of the stages it needs
    gpu: bool                          # runs the network; takes --devices
    run: Callable[[Config, Path, Devices], None]
    help: str


STAGES = (
    Stage("gradients", CHAINS, (), (MOMENTS_FILE, DIRECTIONS_KEPT_FILE),
          ("network", "networks", "datasets", "classes", "events", "sampler", "gradients"), True, run_gradients,
          "gradient pass on the directions sample, per-class sums, kept pairs (Sec. 4.1); GPU"),
    Stage("steering-vectors", CHAINS, ("gradients",), (VECTORS_FILE,), ("target", "steering_vectors"), False,
          run_steering_vectors, "steering vectors of every class and block from the sums (Sec. 4.2); CPU, seconds"),
    Stage("generate", ("steering", "kinematics"), ("steering-vectors",), (GENERATED_DIR,), ("steering",), True,
          run_generate, "baseline and steered generation of the steering sample, every seed and alpha (Sec. 5); GPU"),
    Stage("readout", ("steering",), ("generate",), (READOUT_FILE,), ("readout", "lineshape"), False, run_readout,
          "R_60, target shift, selectivity curve, response fits; lineshape fit only for target Z (App. E); CPU"),
    Stage("plots", ("steering",), ("readout",), (PLOTS_DIR,), ("plots",), False, run_plots,
          "steered spectrum, selectivity curve, pole and width (Z only) as PDF; CPU"),
    Stage("activation-means", ("ablation",), ("gradients",), (MEANS_FILE,), ("ablation.block",), True,
          run_activation_means, "mean residual stream of the kept pairs of the directions sample, for c_k "
                                "(Eq. 25); GPU"),
    Stage("ablate", ("ablation",), ("steering-vectors", "activation-means"), (ABLATED_DIR,),
          ("ablation.seeds",), True, run_ablate,
          "baseline and mean-ablated generation (Eq. 24) of the steering sample, every seed (Sec. 5.3); GPU"),
    Stage("ablation-readout", ("ablation",), ("ablate",), (ABLATION_READOUT_FILE,),
          ("readout", "ablation.peak_range", "ablation.peak_bins"), False, run_ablation_readout,
          "spread and height of the target peak and the deviation below 60 GeV under the ablation (Sec. 5.3); CPU"),
    Stage("ablation-plots", ("ablation",), ("ablation-readout",), (PLOTS_DIR,), ("ablation.figure",), False,
          run_ablation_plots, "spectrum of the baseline, the mean ablation, and real data (Fig. 7) as PDF; CPU"),
    Stage("kinematics-readout", ("kinematics",), ("generate",), (KINEMATICS_FILE, KINEMATICS_READOUT_FILE),
          ("kinematics.bins", "kinematics.readout_min_baseline"), False, run_kinematics_readout,
          "counts of p_T, eta, phi, and E of every particle type, largest change per bin (Figs. 4 and 14); CPU"),
    Stage("kinematics-plots", ("kinematics",), ("kinematics-readout",), (PLOTS_DIR,),
          ("kinematics.ratio_min_baseline", "kinematics.other_types_ratio_range", "kinematics.baseline_label"),
          False, run_kinematics_plots, "kinematics of the edited type (Fig. 4) and of every other type (Fig. 14) as "
                                       "PDF; CPU"),
)
STAGE_NAMES = [stage.name for stage in STAGES]
BY_NAME = {stage.name: stage for stage in STAGES}


def stages_of(config: Config) -> tuple[Stage, ...]:
    """The stages of the config's run (`Config.chain`), in order.

    With steering_vectors.from_file, steering-vectors needs no earlier stage and takes over the config keys of the
    gradients stage (network, events, sampler, ...), which the later stages rely on. The steering and kinematics runs
    then have no gradients stage. The ablation run keeps it, writing only directions_kept.npz: c_k needs those kept
    pairs.
    """
    chain = [s for s in STAGES if config.chain in s.chains]
    if config.steering_vectors.from_file is None:
        return tuple(chain)
    gradients = BY_NAME["gradients"]
    adjusted = []
    for stage in chain:
        if stage.name == "gradients":
            if config.chain != "ablation":
                continue
            stage = replace(stage, outputs=(DIRECTIONS_KEPT_FILE,),
                            help="baseline and kept pairs of the directions sample, no gradient pass "
                                 "(steering_vectors.from_file); GPU")
        elif stage.name == "steering-vectors":
            sections = tuple(key for key in gradients.sections if key != "gradients")
            stage = replace(stage, needs=(), sections=(*sections, *stage.sections),
                            help="check steering_vectors.from_file against the config and copy it; CPU, seconds")
        adjusted.append(stage)
    return tuple(adjusted)


def stage_of(name: str, config: Config) -> Stage:
    """The stage `name` of the config's run; stops if the run has no such stage."""
    stages = {stage.name: stage for stage in stages_of(config)}
    if name in stages:
        return stages[name]
    stage = BY_NAME[name]
    if config.chain in stage.chains:                         # only gradients, left out by steering_vectors.from_file
        raise StageError(f"`steer {name}`: config {config.name} takes its vectors from steering_vectors.from_file "
                         f"{config.steering_vectors.from_file}, so the {config.chain} run has no gradient pass; set "
                         f"steering_vectors.from_file=null to run it")
    raise StageError(f"`steer {name}` is a stage of the {' and '.join(stage.chains)} runs; config {config.name} runs "
                     f"the {config.chain} stages {', '.join(s.name for s in stages_of(config))}")


def ancestors(stage: Stage, stages: tuple[Stage, ...] = STAGES) -> list[Stage]:
    """Every stage of `stages` whose output `stage` depends on, directly or through other stages, in order."""
    by_name = {s.name: s for s in stages}
    names, todo = set(), list(stage.needs)
    while todo:
        name = todo.pop()
        if name not in names:
            names.add(name)
            todo.extend(by_name[name].needs)
    return [s for s in stages if s.name in names]


def used_by(stage: Stage, stages: tuple[Stage, ...] = STAGES) -> set[str]:
    """Config keys that the output of `stage` depends on: its own and those of the stages it depends on."""
    return {key for s in (*ancestors(stage, stages), stage) for key in s.sections}


def touches(key: str, sections: set[str]) -> bool:
    """True if the dotted config key lies inside one of `sections`, or one of them lies inside the key."""
    return any(key == s or key.startswith(s + ".") or s.startswith(key + ".") for s in sections)


def prepare(stage: Stage, config: Config, run: Path, overwrite: bool, warn_later: bool) -> None:
    """Check the run folder, then write config.yaml and remove the old output of the stage (--overwrite).

    `stage` comes from stage_of. Stops if the output of a stage it needs is missing or was made with other settings,
    if its own output exists without --overwrite, or if new settings of this stage would leave the output of a later
    stage made with the old ones.
    """
    stages = stages_of(config)
    new = config.model_dump()
    config_path = run / CONFIG_FILE
    needed = ancestors(stage, stages)
    for done in needed:
        for out in done.outputs:
            require(run / out, done.name)
    changed = (differing_keys(with_defaults(yaml.safe_load(config_path.read_text()), Config), new)
               if config_path.exists() else [])
    if needed and not config_path.exists():
        raise StageError(f"{run} has the output of `steer {needed[0].name}` but no {CONFIG_FILE}; start a new run "
                         f"with --set name=<new name>")
    for done in needed:
        clash = [key for key in changed if touches(key, used_by(done, stages))]
        if clash:
            raise StageError(f"{config_path} differs at {clash}, settings that `steer {done.name}` used for its "
                             f"output; run with the settings of {config_path}, or start a new run with "
                             f"--set name=<new name>")
    existing = [run / out for out in stage.outputs if (run / out).exists()]
    if existing and not overwrite:
        raise StageError(f"{existing[0]} exists; pass --overwrite to replace it, or --set name=<new name> for a new "
                         f"run")
    later = [(s, run / out) for s in stages if stage in ancestors(s, stages) for out in s.outputs
             if (run / out).exists()]
    clash = [key for key in changed if touches(key, used_by(stage, stages))]
    if clash and later:
        raise StageError(f"the settings {clash} differ from {config_path}, and {later[0][1]} was made with the old "
                         f"ones; remove the outputs of the later stages first, or start a new run with "
                         f"--set name=<new name>")
    run.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
    for path in existing:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        log(stage.name, f"removed {path} (--overwrite)")
    for later_stage, path in later if warn_later else ():
        log(stage.name, f"{path} came from the earlier output of this stage; rerun `steer {later_stage.name} "
                        f"--overwrite`")


def run_stage(stage: Stage, config: Config, run: Path, devices: Devices, overwrite: bool,
              warn_later: bool = True) -> None:
    prepare(stage, config, run, overwrite, warn_later)
    log(stage.name, f"start, run folder {run}")
    start = time.perf_counter()
    stage.run(config, run, devices)
    log(stage.name, f"done in {time.perf_counter() - start:.1f} s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steer", description="Run the stages of a steering run or of a mean ablation (README, \"Run the full "
                                  "chain\"); outputs go to RUNS/<name>/.")
    commands = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    chains = {chain: ", ".join(s.name for s in STAGES if chain in s.chains) for chain in CHAINS}
    every = (f"every stage of the config's run in order: {chains['steering']} for a steering run; "
             f"{chains['kinematics']} for a steering sample of every final state; {chains['ablation']} for a config "
             f"with an ablation section")
    for name, text, gpu in [*((s.name, s.help, s.gpu) for s in STAGES), ("all", every, True)]:
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument("config", metavar="CONFIG", help="run config, such as configs/paper/main_z_block1.yaml")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                             help="override one config key by its dotted path, such as sampler.batch_size=1024; the "
                                  "value is read as YAML; repeat --set for more keys")
        command.add_argument("--runs", default="runs", help="folder of the run folders (default: runs)")
        command.add_argument("--overwrite", action="store_true", help="replace the output of the stage if it exists")
        if gpu:
            command.add_argument("--devices", default=None, metavar="LIST",
                                 help="devices of the stages that run the network, comma-separated, such as "
                                      "cuda:0,cuda:1, cpu, or cpu,cpu (two CPU workers); batch i runs on device i mod "
                                      "n, and the files equal those of one device of the same type (default: every "
                                      "visible GPU, the CPU if none)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of `steer`; returns 0 on success and 1, with the reason on stderr, when a stage cannot run."""
    args = build_parser().parse_args(argv)
    try:
        config = read_config(args.config, args.overrides)
        run = Path(args.runs) / config.name
        stages = stages_of(config) if args.stage == "all" else (stage_of(args.stage, config),)
        devices = parse_devices(args.devices) if any(stage.gpu for stage in stages) else (torch.device("cpu"),)
        for stage in stages:
            run_stage(stage, config, run, devices, args.overwrite, warn_later=args.stage != "all")
    except (StageError, ValidationError, NanError, DeviceError) as error:
        print(f"steer {args.stage}: {error}", file=sys.stderr)
        return 1
    return 0
