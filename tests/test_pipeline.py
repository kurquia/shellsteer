"""Every stage of a run, in order, on CPU: each stage reads the files the stage before it wrote.

The settings come from the paper config, except for the size: 2048 events, 4 Euler steps, block 1 only, two seeds,
and alpha = -1 and +1. The events are the 32 of batch32.npz repeated 64 times; each batch of 256 starts from its own
noise, so the 2048 pairs differ. With 4 steps the numbers are not the paper's; the test checks that the stages fit
together. 2048 pairs are about the fewest that put 50 pairs in the Z window for the lineshape fit and fill some bins
below 60 GeV with the 20 baseline pairs that R_60 needs.
"""

from pathlib import Path

import numpy as np
import pytest

import shellsteer.events as events_module
from shellsteer.config import DatasetEntry, DatasetFiles, load_config
from shellsteer.events import BATCH_KEYS, class_of, load_events, select
from shellsteer.gradients import gradient_moments, load_moments, save_moments
from shellsteer.lineshape import fit_masses
from shellsteer.metrics import log_bin_edges, r60, selectivity_curve, target_shift
from shellsteer.networks import load_network
from shellsteer.plots import pole_and_width_figure, save, selectivity_figure, steered_spectrum_figure
from shellsteer.steered_generation import (Edit, generate_pairs, generated_path, kept_path, load_generated, load_kept,
                                           save_generated, save_kept)
from shellsteer.steering_vectors import load_steering_vector, save_steering_vectors, steering_vectors

DATA = Path(__file__).parent / "data"
MAIN = Path(__file__).parent.parent / "configs" / "paper" / "main_z_block1.yaml"
EVENTS, STEPS, BATCH_SIZE, KEEP = 2048, 4, 256, 1800
BLOCK, SEEDS, ALPHAS = 1, (2041, 2042), (-1.0, 1.0)


@pytest.fixture
def entry(tmp_path, monkeypatch):
    """A local file in place of the Hugging Face download, read for both samples."""
    with np.load(DATA / "batch32.npz") as data:
        arrays = {key: np.tile(data[key], (EVENTS // 32,) + (1,) * (data[key].ndim - 1)) for key in BATCH_KEYS}
    arrays["event"] = np.arange(EVENTS)
    arrays["priority"] = np.random.default_rng(0).permutation(EVENTS)
    np.savez(tmp_path / "sample.npz", **arrays)
    monkeypatch.setattr(events_module, "hf_hub_download", lambda *args, **kwargs: str(tmp_path / "sample.npz"))
    return DatasetEntry(hf_repo="test/repo", revision="test", pair="muon", events=EVENTS,
                        files=DatasetFiles(directions="sample.npz", steering="sample.npz"))


def test_a_run_from_the_events_to_the_figures(tmp_path, entry):
    config = load_config(MAIN)
    run = tmp_path / "run"
    run.mkdir()
    classes = config.classes
    network = load_network(config.networks[config.network], "cpu")
    sizes = dict(steps=STEPS, batch_size=BATCH_SIZE, device="cpu")

    # Gradients and steering vectors, on the directions sample.
    directions = load_events(entry, "directions", EVENTS)
    baseline = generate_pairs(network, directions, None, seed=config.gradients.seed, **sizes).mass
    kept = select(baseline, directions.priority, classes, KEEP)
    moments = gradient_moments(network, directions, kept, classes, blocks=[BLOCK], seed=config.gradients.seed,
                               **sizes)
    save_moments(run / "gradient_moments.npz", moments, {"steps": STEPS})
    moments, _ = load_moments(run / "gradient_moments.npz")
    assert moments["count"].sum() == KEEP and moments["non-finite pairs"] == 0

    method = config.steering_vectors
    vectors = steering_vectors(moments, method.method, method.in_time, method.balance)
    save_steering_vectors(run / "steering_vectors.npz", vectors, moments["classes"], moments["blocks"],
                          method.model_dump())
    xi = load_steering_vector(run / "steering_vectors.npz", config.target, BLOCK)
    assert xi.shape == (STEPS, 2, 128)
    assert np.allclose(xi.flatten(1).norm(dim=1).numpy(), 1.0)

    # Steered generation, on the steering sample: the kept pairs come from the first seed's baseline.
    steering = load_events(entry, "steering", EVENTS)
    for seed in SEEDS:
        for alpha in (0.0, *ALPHAS):
            edit = None if alpha == 0 else Edit(xi=xi, alpha=alpha, block=BLOCK, tokens=config.steering.tokens,
                                                pair=entry.pair, steps=None, norm_restore=config.steering.norm_restore)
            generated = generate_pairs(network, steering, edit, seed=seed, **sizes)
            if seed == SEEDS[0] and alpha == 0:
                save_kept(kept_path(run), select(generated.mass, steering.priority, classes, KEEP), generated.event,
                          {"seed": seed, "keep": KEEP})
            save_generated(generated_path(run, seed, alpha), generated,
                           {"alpha": alpha, "block": BLOCK, "target": config.target, "method": method.method,
                            "seed": seed, "steps": STEPS, "edited steps": None, "tokens": config.steering.tokens,
                            "norm_restore": config.steering.norm_restore,
                            "steering vector file": "steering_vectors.npz"})

    # The readout: R_60, the target shift, and the lineshape fit of each seed and alpha, over the kept pairs.
    kept, _, _ = load_kept(kept_path(run))
    counted = kept != ""
    assert counted.sum() == KEEP
    readout = config.readout
    edges = log_bin_edges(*readout.spectrum_range, readout.spectrum_bins)
    fit_settings = dict(peak_window=tuple(classes.windows[config.target]), half_width=config.lineshape.half_width,
                        bins=config.lineshape.bins, min_window_pairs=config.lineshape.min_window_pairs)
    rows, poles, widths, masses = [], {}, {}, {}
    for seed in SEEDS:
        base = load_generated(generated_path(run, seed, 0.0))[0].mass[counted]
        baseline_fit = fit_masses(base, base, None, **fit_settings)
        poles.setdefault(0.0, []).append(baseline_fit.pole)
        widths.setdefault(0.0, []).append(baseline_fit.width)
        in_target = class_of(base, classes) == config.target
        for alpha in ALPHAS:
            steered = load_generated(generated_path(run, seed, alpha))[0].mass[counted]
            assert not np.array_equal(steered, base)
            rows.append((seed, alpha, target_shift(steered, base, in_target),
                         r60(steered, base, edges=edges, max_mass=readout.r60_max_mass,
                             min_baseline_pairs=readout.r60_min_baseline_pairs)))
            fit = fit_masses(steered, base, baseline_fit, **fit_settings)
            poles.setdefault(alpha, []).append(fit.pole)
            widths.setdefault(alpha, []).append(fit.width)
            masses[alpha] = steered
        masses[0.0] = base
    seed, alpha, shifts, r60_values = (np.array(column) for column in zip(*rows))
    curve = selectivity_curve(seed, alpha, shifts, r60_values)
    assert list(curve["alpha"]) == list(ALPHAS)
    assert curve["target shift"][1] > curve["target shift"][0]      # +alpha moves the target up, -alpha down
    assert np.all(curve["R_60"] >= 0)
    np.savez(run / "selectivity_curve.npz", **curve)

    # The figures, from the last seed.
    alphas = sorted(poles)
    plots, figures = config.plots, run / "plots"
    figures.mkdir()
    save(steered_spectrum_figure(masses[0.0], {a: masses[a] for a in ALPHAS}, None, edges=edges,
                                 zooms={label: tuple(zoom) for label, zoom in plots.zooms.items()},
                                 zoom_bins=plots.zoom_bins, baseline_label=plots.baseline_label,
                                 pair_label=plots.pair_label),
         figures / "steered_spectrum.pdf")
    save(selectivity_figure({method.method: curve}, target_label=plots.target_label, pair_label=plots.pair_label,
                            max_mass=readout.r60_max_mass),
         figures / "selectivity.pdf")
    save(pole_and_width_figure(alphas, [np.mean(poles[a]) for a in alphas], [np.std(poles[a]) for a in alphas],
                               [np.mean(widths[a]) for a in alphas], [np.std(widths[a]) for a in alphas],
                               first_order_log_slope=None),
         figures / "pole_and_width.pdf")
    assert sorted(path.name for path in figures.iterdir()) == ["pole_and_width.pdf", "selectivity.pdf",
                                                               "steered_spectrum.pdf"]
