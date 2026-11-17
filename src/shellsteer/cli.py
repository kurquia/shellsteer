"""Command-line runner: `steer <stage> CONFIG [--set key=value]...`, or `steer all CONFIG` for the run of the config.

A config runs one of two chains of stages (STAGES): the steering run (Sec. 5) or, with an `ablation` section, the
mean ablation (Sec. 5.3). A run writes to <runs>/<config name>/ (STRUCTURE.md, "Outputs"). Each stage reads only the
files of the stages it needs and stops, naming the missing file and the stage that writes it, if one is absent. A
stage refuses to replace its own output unless --overwrite is given, and refuses to run on outputs that were made
with other settings. The resolved config, overrides applied, goes to config.yaml in the run folder; the log to stdout.

The readouts write JSON: a few numbers per seed, which a person can read and any language can load. The lineshape
fit is a model of the Z (App. E) and runs only when the target is Z.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import yaml
from pydantic import ValidationError

from shellsteer.activation_means import (activation_means, load_activation_means, mean_projection,
                                         save_activation_means)
from shellsteer.config import Config, load_config
from shellsteer.events import class_names, class_of, load_events, select
from shellsteer.gradients import gradient_moments, load_moments, save_moments
from shellsteer.lineshape import LineshapeFit, fit_masses
from shellsteer.metrics import (below_mass_deviation, log_bin_edges, peak_height_change, pole_response_fit, r60,
                                selectivity_curve, target_shift, window_log_spread)
from shellsteer.networks import load_network
from shellsteer.plots import pole_and_width_figure, save, selectivity_figure, steered_spectrum_figure
from shellsteer.steered_generation import (ABLATION_METADATA_KEYS, Ablation, Edit, ablated_path, generate_pairs,
                                           generated_path, kept_path, load_generated, load_kept, save_generated,
                                           save_kept)
from shellsteer.steering_vectors import gradient_response, load_steering_vector, save_steering_vectors, steering_vectors

CONFIG_FILE = "config.yaml"
MOMENTS_FILE = "gradient_moments.npz"
DIRECTIONS_KEPT_FILE = "directions_kept.npz"
MEANS_FILE = "activation_means.npz"
ABLATED_DIR = "ablated"
ABLATED_KEPT_FILE = "ablated/kept.npz"   # the kept pairs of the steering sample when the ablation runs on it
ABLATION_READOUT_FILE = "ablation_readout.json"
VECTORS_FILE = "steering_vectors.npz"
GENERATED_DIR = "generated"
READOUT_FILE = "readout.json"
PLOTS_DIR = "plots"
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


def check_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise StageError(f"--device {device}: CUDA is not available here; run on a machine with a CUDA GPU, or pass "
                         f"--device cpu (slow at the paper's size)")


def class_counts(kept: np.ndarray, names: list[str]) -> str:
    return ", ".join(f"{name} {int(np.sum(kept == name)):,}" for name in names)


def run_gradients(config: Config, run: Path, device: str) -> None:
    """Gradient pass on the directions sample (paper Sec. 4.1): the per-class sums of gradients.py."""
    stage = "gradients"
    check_device(device)
    entry = config.datasets[config.events.dataset]
    network = load_network(config.networks[config.network], device)
    directions = load_events(entry, "directions", config.events.directions)
    log(stage, f"{len(directions):,} events of the directions sample, device {device}")
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, device=device)
    start = time.perf_counter()
    baseline = generate_pairs(network, directions, None, seed=config.gradients.seed, **sizes).mass
    kept = select(baseline, directions.priority, config.classes, config.events.keep)
    names = class_names(config.classes)
    log(stage, f"baseline of seed {config.gradients.seed} in {time.perf_counter() - start:.1f} s; kept "
               f"{config.events.keep:,} pairs: {class_counts(kept, names)}")
    start = time.perf_counter()
    moments = gradient_moments(network, directions, kept, config.classes, blocks=list(config.gradients.blocks),
                               seed=config.gradients.seed, **sizes)
    counts = ", ".join(f"{name} {int(n):,}" for name, n in zip(moments["classes"], moments["count"]))
    log(stage, f"gradient pass in {time.perf_counter() - start:.1f} s; pairs per class: {counts}; "
               f"non-finite pairs {int(moments['non-finite pairs']):,}")
    metadata = {"steps": config.sampler.steps, "batch_size": config.sampler.batch_size,
                "seed": config.gradients.seed, "blocks": list(config.gradients.blocks), "keep": config.events.keep,
                "events": config.events.directions, "device": device,
                "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump()}
    save_moments(run / MOMENTS_FILE, moments, metadata)
    log(stage, f"wrote {run / MOMENTS_FILE}")
    save_kept(run / DIRECTIONS_KEPT_FILE, kept, directions.event,
              {"seed": config.gradients.seed, "keep": config.events.keep, "sample": "directions", "device": device})
    log(stage, f"wrote {run / DIRECTIONS_KEPT_FILE}")


def run_steering_vectors(config: Config, run: Path, device: str) -> None:
    """Steering vectors of every class and block from the class sums (paper Sec. 4.2)."""
    stage = "steering-vectors"
    moments, _ = load_moments(require(run / MOMENTS_FILE, "gradients"))
    method = config.steering_vectors
    vectors = steering_vectors(moments, method.method, method.in_time, method.balance)
    save_steering_vectors(run / VECTORS_FILE, vectors, moments["classes"], moments["blocks"],
                          {**method.model_dump(), "gradient moments file": MOMENTS_FILE})
    log(stage, f"{method.method}, {method.in_time}, balance {method.balance}: vectors of shape {vectors.shape} "
               f"(classes, blocks, steps, 2, width)")
    log(stage, f"wrote {run / VECTORS_FILE}")


def run_generate(config: Config, run: Path, device: str) -> None:
    """Baseline and steered generation of the steering sample (paper Sec. 5, Eq. 8), every seed and alpha."""
    stage = "generate"
    check_device(device)
    alphas = (0.0, *config.steering.alphas)
    # Every output path first: generated_path stops on an alpha whose file name would clash, before any generation.
    paths = {(seed, alpha): generated_path(run, seed, alpha) for seed in config.steering.seeds for alpha in alphas}
    xi = load_steering_vector(require(run / VECTORS_FILE, "steering-vectors"), config.target, config.steering.block)
    entry = config.datasets[config.events.dataset]
    network = load_network(config.networks[config.network], device)
    events = load_events(entry, "steering", config.events.steering)
    log(stage, f"{len(events):,} events of the steering sample, device {device}; {config.target} vector at block "
               f"{config.steering.block}; seeds {list(config.steering.seeds)}, alphas {list(alphas)}")
    edit_steps = None if config.steering.steps == "all" else tuple(config.steering.steps)
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, device=device)
    for seed in config.steering.seeds:
        for alpha in alphas:
            edit = None if alpha == 0 else Edit(xi=xi, alpha=alpha, block=config.steering.block,
                                                tokens=config.steering.tokens, pair=entry.pair, steps=edit_steps,
                                                norm_restore=config.steering.norm_restore)
            start = time.perf_counter()
            generated = generate_pairs(network, events, edit, seed=seed, **sizes)
            seconds = time.perf_counter() - start
            if seed == config.steering.seeds[0] and alpha == 0:
                kept = select(generated.mass, events.priority, config.classes, config.events.keep)
                save_kept(kept_path(run), kept, generated.event, {"seed": seed, "keep": config.events.keep})
                log(stage, f"kept {config.events.keep:,} pairs on the baseline of seed {seed}: "
                           f"{class_counts(kept, class_names(config.classes))}; wrote {kept_path(run)}")
            path = paths[(seed, alpha)]
            save_generated(path, generated, {
                "alpha": alpha, "block": config.steering.block, "target": config.target,
                "method": config.steering_vectors.method, "seed": seed, "steps": config.sampler.steps,
                "edited steps": None if edit_steps is None else list(edit_steps), "tokens": config.steering.tokens,
                "norm_restore": config.steering.norm_restore, "steering vector file": VECTORS_FILE,
                "batch_size": config.sampler.batch_size, "device": device,
                "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump()})
            log(stage, f"seed {seed}, alpha {alpha:+g}: {len(generated.mass):,} pairs in {seconds:.1f} s; "
                       f"wrote {path}")


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


def run_readout(config: Config, run: Path, device: str) -> None:
    """R_60, the target shift, the selectivity curve, the response fits, and the Z lineshape fits (Sec. 5, App. E)."""
    stage = "readout"
    moments, _ = load_moments(require(run / MOMENTS_FILE, "gradients"))
    xi = load_steering_vector(require(run / VECTORS_FILE, "steering-vectors"), config.target, config.steering.block)
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
        "gradient response": gradient_response(moments, xi, config.target, config.steering.block),
        "selectivity curve": {key: value.tolist() for key, value in curve.items()},
        "per seed and alpha": rows, "lineshape over seeds": None, "pole response fit": None}
    log(stage, f"gradient response S = {result['gradient response']:.4f} (Eq. 21, first order)")
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


def run_plots(config: Config, run: Path, device: str) -> None:
    """The steered spectrum (Fig. 3) and the selectivity curve, from the first seed; pole and width (Fig. 5) for Z."""
    stage = "plots"
    result = json.loads(require(run / READOUT_FILE, "readout").read_text())
    require_generated(config, run)
    seed = config.steering.seeds[0]
    _, counted = counted_pairs(run, seed)
    masses = {alpha: load_generated(generated_path(run, seed, alpha))[0].mass[counted]
              for alpha in (0.0, *config.steering.alphas)}
    plots, readout = config.plots, config.readout
    folder = run / PLOTS_DIR
    folder.mkdir()
    figures = {
        "steered_spectrum.pdf": steered_spectrum_figure(
            masses.pop(0.0), masses, None, edges=log_bin_edges(*readout.spectrum_range, readout.spectrum_bins),
            zooms={label: tuple(zoom) for label, zoom in plots.zooms.items()}, zoom_bins=plots.zoom_bins,
            baseline_label=plots.baseline_label, pair_label=plots.pair_label),
        "selectivity.pdf": selectivity_figure({config.steering_vectors.method: result["selectivity curve"]},
                                              target_label=plots.target_label, pair_label=plots.pair_label,
                                              max_mass=readout.r60_max_mass)}
    over_seeds = result["lineshape over seeds"]
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


def ablation_kept_file(config: Config, run: Path) -> tuple[Path, str]:
    """The kept pairs of the ablated sample and the stage that writes them: the directions sample's from `steer
    gradients`, the steering sample's from `steer ablate` on the baseline of its first seed."""
    if config.ablation.sample == "directions":
        return run / DIRECTIONS_KEPT_FILE, "gradients"
    return run / ABLATED_KEPT_FILE, "ablate"


def run_activation_means(config: Config, run: Path, device: str) -> None:
    """Mean of the residual stream on the pair's tokens over the kept pairs of the directions sample, the input of
    c_k (paper Eq. 25): one forward pass without edits with the noise of gradients.seed."""
    stage = "activation-means"
    check_device(device)
    kept_file = require(run / DIRECTIONS_KEPT_FILE, "gradients")
    kept, event, _ = load_kept(kept_file)
    entry = config.datasets[config.events.dataset]
    network = load_network(config.networks[config.network], device)
    events = load_events(entry, "directions", config.events.directions)
    if not np.array_equal(events.event, event):
        raise StageError(f"the directions sample holds other events than {kept_file}; rerun `steer gradients "
                         f"--overwrite`")
    log(stage, f"{len(events):,} events of the directions sample, device {device}; block {config.ablation.block}, "
               f"seed {config.gradients.seed}")
    start = time.perf_counter()
    means = activation_means(network, events, kept, config.classes, blocks=[config.ablation.block],
                             steps=config.sampler.steps, batch_size=config.sampler.batch_size,
                             seed=config.gradients.seed, device=device)
    counts = ", ".join(f"{name} {int(n):,}" for name, n in zip(means["classes"], means["count"]))
    log(stage, f"forward pass in {time.perf_counter() - start:.1f} s; pairs per class: {counts} (total "
               f"{int(np.sum(means['count'])):,}); non-finite pairs {int(means['non-finite pairs']):,}")
    save_activation_means(run / MEANS_FILE, means, {
        "steps": config.sampler.steps, "batch_size": config.sampler.batch_size, "seed": config.gradients.seed,
        "blocks": [config.ablation.block], "kept pairs file": DIRECTIONS_KEPT_FILE, "device": device,
        "network": config.networks[config.network].model_dump(), "dataset": entry.model_dump()})
    log(stage, f"wrote {run / MEANS_FILE}")


def run_ablate(config: Config, run: Path, device: str) -> None:
    """Baseline and mean-ablated generation (paper Eq. 24, c_k of Eq. 25) of ablation.sample for every seed."""
    stage = "ablate"
    check_device(device)
    ablation = config.ablation
    blocks = (None, ablation.block)                                    # None: the baseline
    paths = {(seed, block): ablated_path(run, seed, block) for seed in ablation.seeds for block in blocks}
    xi = load_steering_vector(require(run / VECTORS_FILE, "steering-vectors"), config.target, ablation.block)
    means, _ = load_activation_means(require(run / MEANS_FILE, "activation-means"))
    c = mean_projection(means, xi, ablation.block)
    log(stage, f"c_k from {int(np.sum(means['count'])):,} pairs: {c.min():+.4f} to {c.max():+.4f} over "
               f"{len(c)} steps")
    entry = config.datasets[config.events.dataset]
    events = load_events(entry, ablation.sample, getattr(config.events, ablation.sample))
    if ablation.sample == "directions":
        kept_file = require(run / DIRECTIONS_KEPT_FILE, "gradients")
        if not np.array_equal(events.event, load_kept(kept_file)[1]):
            raise StageError(f"the directions sample holds other events than {kept_file}; rerun `steer gradients "
                             f"--overwrite`")
    network = load_network(config.networks[config.network], device)
    log(stage, f"{len(events):,} events of the {ablation.sample} sample, device {device}; {config.target} vector at "
               f"block {ablation.block}; seeds {list(ablation.seeds)}")
    edit = Ablation(xi=xi, mean=torch.from_numpy(c), block=ablation.block)
    sizes = dict(steps=config.sampler.steps, batch_size=config.sampler.batch_size, device=device)
    for seed in ablation.seeds:
        for block in blocks:
            start = time.perf_counter()
            generated = generate_pairs(network, events, None if block is None else edit, seed=seed, **sizes)
            seconds = time.perf_counter() - start
            if ablation.sample == "steering" and seed == ablation.seeds[0] and block is None:
                kept = select(generated.mass, events.priority, config.classes, config.events.keep)
                save_kept(run / ABLATED_KEPT_FILE, kept, generated.event, {"seed": seed, "keep": config.events.keep,
                                                                            "sample": "steering"})
                log(stage, f"kept {config.events.keep:,} pairs on the baseline of seed {seed}: "
                           f"{class_counts(kept, class_names(config.classes))}; wrote {run / ABLATED_KEPT_FILE}")
            path = paths[(seed, block)]
            save_generated(path, generated, {
                "block": block, "target": config.target, "method": config.steering_vectors.method, "seed": seed,
                "steps": config.sampler.steps, "sample": ablation.sample, "steering vector file": VECTORS_FILE,
                "activation means file": MEANS_FILE, "c_k": c.tolist(), "batch_size": config.sampler.batch_size,
                "device": device, "network": config.networks[config.network].model_dump(),
                "dataset": entry.model_dump()}, required=ABLATION_METADATA_KEYS)
            what = "baseline" if block is None else f"ablated at block {block}"
            log(stage, f"seed {seed}, {what}: {len(generated.mass):,} pairs in {seconds:.1f} s; wrote {path}")


def run_ablation_readout(config: Config, run: Path, device: str) -> None:
    """The numbers of the mean ablation (paper Sec. 5.3, Fig. 7): spread of the target peak, change of its height,
    and the deviation below readout.r60_max_mass, per seed and over seeds (metrics.py)."""
    stage = "ablation-readout"
    ablation, readout, classes = config.ablation, config.readout, config.classes
    kept_file, kept_stage = ablation_kept_file(config, run)
    kept, kept_event, _ = load_kept(require(kept_file, kept_stage))
    for seed in ablation.seeds:
        for block in (None, ablation.block):
            require(ablated_path(run, seed, block), "ablate")
    window = tuple(classes.windows[config.target])
    edges = log_bin_edges(*readout.spectrum_range, readout.spectrum_bins)
    peak_edges = np.linspace(*ablation.peak_range, ablation.peak_bins + 1)
    deviation_settings = dict(max_mass=readout.r60_max_mass, min_baseline_pairs=readout.r60_min_baseline_pairs)
    rows, histograms = [], {"spectrum": ([], []), "peak": ([], [])}       # (baseline, ablated) per seed
    for seed in ablation.seeds:
        (base, _), (ablated, _) = (load_generated(ablated_path(run, seed, block)) for block in (None, ablation.block))
        for generated, block in ((base, None), (ablated, ablation.block)):
            if not np.array_equal(generated.event, kept_event):
                raise StageError(f"{ablated_path(run, seed, block)} holds other events than {kept_file}; rerun "
                                 f"`steer ablate --overwrite`")
        # As in the paper: the kept pairs with a class on this seed's own baseline mass.
        counted = (kept != "") & (class_of(base.mass, classes) != "")
        b, a = base.mass[counted], ablated.mass[counted]
        spread = (window_log_spread(b, window), window_log_spread(a, window))
        spectrum = (np.histogram(b, edges)[0], np.histogram(a, edges)[0])
        peak = (np.histogram(b, peak_edges)[0], np.histogram(a, peak_edges)[0])
        for key, pair in (("spectrum", spectrum), ("peak", peak)):
            for per_seed, counts in zip(histograms[key], pair):
                per_seed.append(counts)
        row = {"seed": seed, "pairs": int(counted.sum()),
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
        "target": config.target, "block": ablation.block, "sample": ablation.sample, "seeds": list(ablation.seeds),
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


@dataclass(frozen=True)
class Stage:
    name: str
    chain: Literal["both", "steering", "ablation"]   # the run the stage belongs to
    needs: tuple[str, ...]             # stages whose outputs this stage reads
    outputs: tuple[str, ...]           # paths in the run folder that the stage writes
    sections: tuple[str, ...]          # config keys, dotted, that the stage reads beyond those of the stages it needs
    gpu: bool                          # takes --device
    run: Callable[[Config, Path, str], None]
    help: str


STAGES = (
    Stage("gradients", "both", (), (MOMENTS_FILE, DIRECTIONS_KEPT_FILE),
          ("network", "networks", "datasets", "classes", "events", "sampler", "gradients"), True, run_gradients,
          "gradient pass on the directions sample, per-class sums, kept pairs (Sec. 4.1); GPU"),
    Stage("steering-vectors", "both", ("gradients",), (VECTORS_FILE,), ("target", "steering_vectors"), False,
          run_steering_vectors, "steering vectors of every class and block from the sums (Sec. 4.2); CPU, seconds"),
    Stage("generate", "steering", ("steering-vectors",), (GENERATED_DIR,), ("steering",), True, run_generate,
          "baseline and steered generation of the steering sample, every seed and alpha (Sec. 5); GPU"),
    Stage("readout", "steering", ("generate",), (READOUT_FILE,), ("readout", "lineshape"), False, run_readout,
          "R_60, target shift, selectivity curve, response fits; lineshape fit only for target Z (App. E); CPU"),
    Stage("plots", "steering", ("readout",), (PLOTS_DIR,), ("plots",), False, run_plots,
          "steered spectrum, selectivity curve, pole and width (Z only) as PDF; CPU"),
    Stage("activation-means", "ablation", ("gradients",), (MEANS_FILE,), ("ablation.block",), True,
          run_activation_means, "mean residual stream of the kept pairs of the directions sample, for c_k "
                                "(Eq. 25); GPU"),
    Stage("ablate", "ablation", ("steering-vectors", "activation-means"), (ABLATED_DIR,),
          ("ablation.sample", "ablation.seeds"), True, run_ablate,
          "baseline and mean-ablated generation (Eq. 24) of ablation.sample, every seed (Sec. 5.3); GPU"),
    Stage("ablation-readout", "ablation", ("ablate",), (ABLATION_READOUT_FILE,),
          ("readout", "ablation.peak_range", "ablation.peak_bins"), False, run_ablation_readout,
          "spread and height of the target peak and the deviation below 60 GeV under the ablation (Sec. 5.3); CPU"),
)
STAGE_NAMES = [stage.name for stage in STAGES]
BY_NAME = {stage.name: stage for stage in STAGES}


def chain_of(config: Config) -> str:
    return "steering" if config.ablation is None else "ablation"


def ancestors(stage: Stage) -> list[Stage]:
    """Every stage whose output `stage` depends on, directly or through other stages, in the order of STAGES."""
    names, todo = set(), list(stage.needs)
    while todo:
        name = todo.pop()
        if name not in names:
            names.add(name)
            todo.extend(BY_NAME[name].needs)
    return [s for s in STAGES if s.name in names]


def used_by(stage: Stage) -> set[str]:
    """Config keys that the output of `stage` depends on: its own and those of the stages it depends on."""
    return {key for s in (*ancestors(stage), stage) for key in s.sections}


def touches(key: str, sections: set[str]) -> bool:
    """True if the dotted config key lies inside one of `sections`, or one of them lies inside the key."""
    return any(key == s or key.startswith(s + ".") or s.startswith(key + ".") for s in sections)


def prepare(stage: Stage, config: Config, run: Path, overwrite: bool, warn_later: bool) -> None:
    """Check the run folder, then write config.yaml and remove the old output of the stage (--overwrite).

    Stops if the stage belongs to the other chain, if the output of a stage it needs is missing or was made with
    other settings, if its own output exists without --overwrite, or if new settings of this stage would leave the
    output of a later stage made with the old ones.
    """
    if stage.chain not in ("both", chain_of(config)):
        other = [s.name for s in STAGES if s.chain in ("both", chain_of(config))]
        raise StageError(f"`steer {stage.name}` is a stage of the {stage.chain} run; config {config.name} runs the "
                         f"{chain_of(config)} stages {', '.join(other)}")
    new = config.model_dump()
    config_path = run / CONFIG_FILE
    needed = ancestors(stage)
    for done in needed:
        for out in done.outputs:
            require(run / out, done.name)
    changed = differing_keys(yaml.safe_load(config_path.read_text()), new) if config_path.exists() else []
    if needed and not config_path.exists():
        raise StageError(f"{run} has the output of `steer {needed[0].name}` but no {CONFIG_FILE}; start a new run "
                         f"with --set name=<new name>")
    for done in needed:
        clash = [key for key in changed if touches(key, used_by(done))]
        if clash:
            raise StageError(f"{config_path} differs at {clash}, settings that `steer {done.name}` used for its "
                             f"output; run with the settings of {config_path}, or start a new run with "
                             f"--set name=<new name>")
    existing = [run / out for out in stage.outputs if (run / out).exists()]
    if existing and not overwrite:
        raise StageError(f"{existing[0]} exists; pass --overwrite to replace it, or --set name=<new name> for a new "
                         f"run")
    later = [(s, run / out) for s in STAGES if stage in ancestors(s) for out in s.outputs if (run / out).exists()]
    clash = [key for key in changed if touches(key, used_by(stage))]
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


def run_stage(stage: Stage, config: Config, run: Path, device: str, overwrite: bool, warn_later: bool = True) -> None:
    prepare(stage, config, run, overwrite, warn_later)
    log(stage.name, f"start, run folder {run}")
    start = time.perf_counter()
    stage.run(config, run, device)
    log(stage.name, f"done in {time.perf_counter() - start:.1f} s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steer", description="Run the stages of a steering run or of a mean ablation (STRUCTURE.md); outputs go "
                                  "to RUNS/<name>/.")
    commands = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    chains = {chain: ", ".join(s.name for s in STAGES if s.chain in ("both", chain)) for chain in ("steering",
                                                                                                    "ablation")}
    every = (f"every stage of the config's run in order: {chains['steering']} for a steering run; "
             f"{chains['ablation']} for a config with an ablation section")
    for name, text, gpu in [*((s.name, s.help, s.gpu) for s in STAGES), ("all", every, True)]:
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument("config", metavar="CONFIG", help="run config, such as configs/paper/main_z_block1.yaml")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                             help="override one config key by its dotted path, such as sampler.batch_size=1024; the "
                                  "value is read as YAML; repeat --set for more keys")
        command.add_argument("--runs", default="runs", help="folder of the run folders (default: runs)")
        command.add_argument("--overwrite", action="store_true", help="replace the output of the stage if it exists")
        if gpu:
            command.add_argument("--device", default="cuda",
                                 help="torch device (default: cuda); stops if CUDA is missing, use cpu to run there")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of `steer`; returns 0 on success and 1, with the reason on stderr, when a stage cannot run."""
    args = build_parser().parse_args(argv)
    try:
        config = read_config(args.config, args.overrides)
        run = Path(args.runs) / config.name
        if args.stage == "all":
            stages = [s for s in STAGES if s.chain in ("both", chain_of(config))]
        else:
            stages = [BY_NAME[args.stage]]
        device = getattr(args, "device", "cpu")
        if any(stage.gpu for stage in stages):
            check_device(device)
        for stage in stages:
            run_stage(stage, config, run, device, args.overwrite, warn_later=args.stage != "all")
    except (StageError, ValidationError) as error:
        print(f"steer {args.stage}: {error}", file=sys.stderr)
        return 1
    return 0
