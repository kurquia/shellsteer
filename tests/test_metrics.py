import numpy as np
import pytest

from shellsteer.metrics import log_bin_edges, r60, selectivity_curve, target_shift

EDGES = log_bin_edges(0.3, 300, 200)                       # the paper's binning
SETTINGS = dict(edges=EDGES, max_mass=60, min_baseline_pairs=20)
CENTRE = np.sqrt(EDGES[1:] * EDGES[:-1])
BELOW_60 = CENTRE < 60


def masses(counts: np.ndarray, total: int) -> np.ndarray:
    """counts[b] pairs at the centre of bin b, then pairs at 1000 GeV, outside every bin, up to `total` pairs."""
    inside = np.repeat(CENTRE, counts)
    return np.concatenate([inside, np.full(total - len(inside), 1000.0)])


def test_paper_binning_has_153_bins_below_60_gev():
    assert len(EDGES) == 201
    assert EDGES[0] == pytest.approx(0.3) and EDGES[-1] == pytest.approx(300)
    assert BELOW_60.sum() == 153
    assert np.flatnonzero(BELOW_60)[-1] == 152


def test_every_selected_bin_up_by_ten_percent():
    baseline = np.full(200, 100)
    steered = np.where(BELOW_60, 110, 0)                   # the bins above 60 GeV lose all pairs; they do not count
    assert r60(masses(steered, 20_000), masses(baseline, 20_000), **SETTINGS) == pytest.approx(0.1)


def test_one_bin_doubled_counts_once_in_153():
    baseline = np.full(200, 100)
    steered = baseline.copy()
    steered[152] = 200
    assert r60(masses(steered, 30_000), masses(baseline, 30_000), **SETTINGS) == pytest.approx(1 / 153)


def test_identical_spectra_give_zero():
    rng = np.random.default_rng(0)
    mass = np.exp(rng.uniform(np.log(0.3), np.log(300), 50_000))
    assert r60(mass, mass, **SETTINGS) == 0.0
    assert r60(rng.permutation(mass), mass, **SETTINGS) == 0.0     # R_60 compares the spectra, not the pairs


def test_bins_with_fewer_than_20_baseline_pairs_are_left_out():
    baseline = np.full(200, 100)
    baseline[10], baseline[11] = 19, 20
    steered = baseline.copy()
    steered[10], steered[11] = 100, 22                     # bin 10 left out, bin 11 counts with +10 %
    assert r60(masses(steered, 30_000), masses(baseline, 30_000), **SETTINGS) == pytest.approx(0.1 / 152)


def test_target_shift_is_a_difference_of_medians():
    baseline = np.array([91.0, 85.0, 95.0, 3.1, 20.0])
    in_target = np.array([True, True, True, False, False])
    steered = baseline * np.exp([0.1, 0.3, 0.2, 2.0, -1.0])
    # median of log m: log 91 before the edit, log 85 + 0.3 after it. The median of the per-pair shifts would be 0.2.
    assert target_shift(steered, baseline, in_target) == pytest.approx(0.3 + np.log(85 / 91))


def test_selectivity_curve_over_two_seeds_and_three_alphas():
    seed = np.array([12, 11, 11, 12, 11, 12])
    alpha = np.array([2.0, -1.0, 2.0, -1.0, 1.0, 1.0])
    noise = np.where(seed == 11, 1.0, -1.0)
    shifts = 0.1 * alpha + 0.01 * noise
    r60_values = 0.05 * np.abs(alpha) + 0.002 * noise
    curve = selectivity_curve(seed, alpha, shifts, r60_values)
    assert list(curve) == ["alpha", "target shift", "target shift std", "R_60", "R_60 std"]
    np.testing.assert_array_equal(curve["alpha"], [-1.0, 1.0, 2.0])
    np.testing.assert_allclose(curve["target shift"], [-0.1, 0.1, 0.2], atol=1e-15)
    np.testing.assert_allclose(curve["target shift std"], [0.01, 0.01, 0.01])
    np.testing.assert_allclose(curve["R_60"], [0.05, 0.05, 0.1])
    np.testing.assert_allclose(curve["R_60 std"], [0.002, 0.002, 0.002])


def test_r60_stops_when_no_bin_qualifies():
    mass = masses(np.full(200, 100), 20_000)
    with pytest.raises(ValueError, match="no bin with its centre below 60 GeV has 101 or more baseline pairs"):
        r60(mass, mass, edges=EDGES, max_mass=60, min_baseline_pairs=101)
    with pytest.raises(ValueError, match="no bin with its centre below 0.3 GeV"):
        r60(mass, mass, edges=EDGES, max_mass=0.3, min_baseline_pairs=20)


def test_masses_of_different_length_stop():
    mass = masses(np.full(200, 100), 20_000)
    with pytest.raises(ValueError, match="19999 steered and 20000 baseline masses"):
        r60(mass[1:], mass, **SETTINGS)
    with pytest.raises(ValueError, match="19999 steered and 20000 baseline masses"):
        target_shift(mass[1:], mass, mass > 80)
    with pytest.raises(ValueError, match="in_target: expected a boolean array of shape \\(20000,\\)"):
        target_shift(mass, mass, (mass > 80)[1:])


def test_masses_that_are_not_finite_and_positive_stop():
    mass = masses(np.full(200, 100), 20_000)
    bad = mass.copy()
    bad[[7, 9]] = [np.nan, 0.0]
    with pytest.raises(ValueError, match="steered masses: 2 are not finite and positive, the first at pair 7"):
        r60(bad, mass, **SETTINGS)


def test_selectivity_curve_stops_on_missing_alpha_or_length():
    seed = np.array([11, 11, 11, 12, 12])
    alpha = np.array([-1.0, 1.0, 2.0, -1.0, 1.0])
    values = np.zeros(5)
    with pytest.raises(ValueError, match="seed 12 has alphas \\[-1.0, 1.0\\], expected each of \\[-1.0, 1.0, 2.0\\]"):
        selectivity_curve(seed, alpha, values, values)
    with pytest.raises(ValueError, match="got lengths 5, 5, 4, 5"):
        selectivity_curve(seed, alpha, values[1:], values)
