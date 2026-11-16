from pathlib import Path

import numpy as np
import pytest

from shellsteer.lineshape import (LOG_MASS, expected_counts, fit_counts, fit_masses, lineshape, peak_centre,
                                  window_edges)

DATA = Path(__file__).parent / "data"
SETTINGS = dict(peak_window=(81.19, 101.19), half_width=0.25, bins=100, min_window_pairs=50)
TRUTH = dict(pole=91.2, width=2.5, resolution=1.7, luminosity_exponent=-1.0)

# The paper's fit of the baseline generation of seed 2041 on events B: readout.json of step_balanced_Z_half.
PAPER_POLE, PAPER_WIDTH, PAPER_DEVIANCE_PER_DOF = 90.5775003091106, 2.413043175841054, 1.906174600785674


def test_paper_baseline_histogram_refits_to_the_paper_values():
    """The counts and edges of the paper's second fit window. The paper started this fit from its first fit; here it
    starts from BASELINE_START and ends at the same minimum."""
    with np.load(DATA / "lineshape_baseline_seed2041.npz") as data:
        counts, edges = data["counts"], data["edges"]
    fit = fit_counts(counts, edges, None)
    assert fit.pole == pytest.approx(PAPER_POLE, abs=1e-6)
    assert fit.width == pytest.approx(PAPER_WIDTH, abs=1e-6)
    assert fit.deviance_per_dof == pytest.approx(PAPER_DEVIANCE_PER_DOF, rel=1e-9)
    assert fit.pairs_in_window == counts.sum() == 199168
    assert fit.at_bound == ()
    assert np.array_equal(fit.edges, edges) and fit.counts.shape == fit.expected.shape == (100,)


def test_counts_of_the_model_give_its_parameters_back():
    edges = window_edges(90.0, half_width=0.25, bins=100)     # the pole is not at the centre of the window
    truth = {**TRUTH, "log_yield": 12.0}
    fit = fit_counts(expected_counts(edges, **truth), edges, None)
    for name, value in truth.items():
        assert getattr(fit, name) == pytest.approx(value, abs=1e-6), name
    assert fit.deviance_per_dof < 1e-9 and fit.at_bound == ()


def test_a_fit_from_a_start_fit_ends_at_the_same_minimum():
    edges = window_edges(90.0, half_width=0.25, bins=100)
    counts = expected_counts(edges, **TRUTH, log_yield=12.0)
    start = fit_counts(counts * 1.1, edges, None)
    fit = fit_counts(counts, edges, start)
    assert fit.pole == pytest.approx(TRUTH["pole"], abs=1e-6)
    assert fit.log_yield == pytest.approx(12.0, abs=1e-6)


def test_a_parameter_at_its_bound_is_named():
    edges = window_edges(90.0, half_width=0.25, bins=100)
    narrow = fit_counts(expected_counts(edges, **{**TRUTH, "width": 1e-5 * TRUTH["pole"]}, log_yield=12.0), edges,
                        None)
    assert narrow.at_bound == ("width",)
    outside = window_edges(70.0, half_width=0.25, bins=100)    # the peak lies above the window
    far = fit_counts(expected_counts(outside, **TRUTH, log_yield=12.0), outside, None)
    assert far.at_bound == ("pole",) and far.pole == pytest.approx(outside[-1], rel=1e-6)


def test_fit_masses_finds_the_pole_of_masses_drawn_from_the_model():
    """300,000 masses drawn on the grid points of the lineshape, so the counts follow the model exactly."""
    probability = lineshape(**TRUTH)
    rng = np.random.default_rng(0)
    mass = np.exp(LOG_MASS[rng.choice(len(LOG_MASS), size=300_000, p=probability / probability.sum())])
    fit = fit_masses(mass, mass, None, **SETTINGS)
    assert fit.pole == pytest.approx(TRUTH["pole"], abs=0.03)          # about three standard deviations
    assert fit.width == pytest.approx(TRUTH["width"], abs=0.1)
    assert fit.resolution == pytest.approx(TRUTH["resolution"], abs=0.06)
    assert fit.at_bound == ()
    # The second window is centred on the pole of the first fit, which is close to the final pole.
    assert abs(np.log(np.sqrt(fit.edges[0] * fit.edges[-1]) / fit.pole)) < 1e-3
    assert np.log(fit.edges[-1] / fit.edges[0]) == pytest.approx(0.5)
    assert fit.pairs_in_window == np.sum((mass >= fit.edges[0]) & (mass < fit.edges[-1]))


def test_peak_centre_uses_the_pairs_whose_baseline_mass_is_in_the_window():
    baseline = np.array([81.19, 85.0, 90.0, 95.0, 101.19, 120.0])      # the window is an open interval
    steered = np.array([1.0, 80.0, 88.0, 1000.0, 2.0, 3.0])
    centre = peak_centre(steered, baseline, peak_window=(81.19, 101.19), min_window_pairs=3)
    assert centre == pytest.approx(88.0, rel=1e-12)                     # exp(median of log 80, log 88, log 1000)


def test_bad_inputs_stop():
    with pytest.raises(ValueError, match="49 pairs have a baseline mass in the window 81.19 to 101.19 GeV, "
                                         "min_window_pairs is 50"):
        fit_masses(np.full(49, 91.0), np.full(49, 91.0), None, **SETTINGS)
    good, bad = np.full(100, 91.0), np.full(100, 91.0)
    bad[3] = np.nan
    bad[7] = np.inf
    with pytest.raises(ValueError, match="mass: not finite at 2 of 100 pairs, the first at index 3"):
        fit_masses(bad, good, None, **SETTINGS)
    with pytest.raises(ValueError, match="baseline_mass: not finite at 2 of 100 pairs"):
        fit_masses(good, bad, None, **SETTINGS)
    with pytest.raises(ValueError, match="expected \\(pairs,\\) for both"):
        fit_masses(good, good[:99], None, **SETTINGS)

    edges = window_edges(90.0, half_width=0.25, bins=100)
    with pytest.raises(ValueError, match="expected \\(bins,\\) and \\(bins \\+ 1,\\)"):
        fit_counts(np.ones(100), edges[:-1], None)
    with pytest.raises(ValueError, match="5 bins, need more than the 5 free parameters"):
        fit_counts(np.ones(5), edges[:6], None)
    with pytest.raises(ValueError, match="counts must be finite and not negative"):
        fit_counts(np.full(100, -1.0), edges, None)
    with pytest.raises(ValueError, match="edges must increase"):
        fit_counts(np.ones(100), edges[::-1], None)
    with pytest.raises(ValueError, match="the lineshape grid 25 to 319.9 GeV"):
        fit_counts(np.ones(100), window_edges(300.0, half_width=0.25, bins=100), None)
