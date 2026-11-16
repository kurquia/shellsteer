"""How selective a steering is, from the generated masses of the pairs: R_60 and the target shift (paper Sec. 5).

For each seed, the steering stage generates the kept pairs (`events.select`) without edits, which gives the baseline,
and once per alpha with the edit, all from the same noise. Each pair has one baseline mass and one steered mass per
alpha, and the functions here compare the two masses of the same pairs. Callers pass masses in GeV, of the kept pairs
only.

R_60 measures how much the steering changes the spectrum below 60 GeV, away from the Z (paper Eq. 19):

    R_60 = (1 / n_b) sum over b of |N_b / N_b^baseline - 1|,

with N_b and N_b^baseline the steered and the baseline pairs in bin b. The sum runs over the n_b bins whose geometric
centre is below `max_mass` and that hold at least `min_baseline_pairs` baseline pairs, every bin with the same weight.
The paper uses 200 bins equally spaced in log m over 0.3-300 GeV, max_mass = 60 GeV, and min_baseline_pairs = 20;
153 of these bins have their centre below 60 GeV. R_60 between the baselines of two seeds is the noise floor.

The target shift measures how far the steering moves the target peak, in log m:

    target shift = median of log m^steered - median of log m^baseline,

with both medians over the same pairs: those whose baseline mass of this seed is in the target class.

The selectivity curve gives, for each alpha, the mean and the standard deviation over seeds of the target shift and
of R_60. Steering vectors from different methods then compare at the same target shift (paper App. B.1).
"""

from __future__ import annotations

import numpy as np


def log_bin_edges(low: float, high: float, bins: int) -> np.ndarray:
    """(bins + 1,) bin edges in GeV, equally spaced in log m from `low` to `high`, as in the paper's histograms."""
    if not 0 < low < high:
        raise ValueError(f"bin edges: need 0 < low < high, got low {low} and high {high} GeV")
    if bins < 1:
        raise ValueError(f"bin edges: need at least 1 bin, got {bins}")
    return np.logspace(np.log10(low), np.log10(high), bins + 1)


def checked_masses(steered_mass: np.ndarray, baseline_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both (E,) mass arrays in float64, as in the paper's analysis of the float32 masses.

    Stops unless both have shape (E,) with the same E and every mass is finite and positive.
    """
    masses = []
    for name, mass in (("steered", steered_mass), ("baseline", baseline_mass)):
        mass = np.asarray(mass, dtype=np.float64)
        if mass.ndim != 1:
            raise ValueError(f"{name} masses: expected shape (pairs,), got {mass.shape}")
        bad = ~(np.isfinite(mass) & (mass > 0))
        if bad.any():
            raise ValueError(f"{name} masses: {int(bad.sum())} are not finite and positive, the first at pair "
                             f"{int(np.flatnonzero(bad)[0])}")
        masses.append(mass)
    if len(masses[0]) != len(masses[1]):
        raise ValueError(f"{len(masses[0])} steered and {len(masses[1])} baseline masses; both must be the masses "
                         f"of the same pairs")
    return masses[0], masses[1]


def r60(steered_mass: np.ndarray, baseline_mass: np.ndarray, *, edges: np.ndarray, max_mass: float,
        min_baseline_pairs: int) -> float:
    """R_60 of paper Eq. 19, as a fraction (0.112, not 11.2 %).

    steered_mass, baseline_mass: (E,) mass of each pair in GeV, with and without the edit, same pairs in the same
    order. edges: (bins + 1,) increasing bin edges in GeV. A mass outside the edges enters no bin.
    """
    steered_mass, baseline_mass = checked_masses(steered_mass, baseline_mass)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(f"edges: expected shape (bins + 1,) with at least 2 edges, got {edges.shape}")
    if edges[0] <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"edges: expected positive, increasing edges, got first edge {edges[0]} and "
                         f"{int(np.sum(np.diff(edges) <= 0))} that do not increase")
    if min_baseline_pairs < 1:
        raise ValueError(f"min_baseline_pairs: need at least 1, got {min_baseline_pairs}")
    steered = np.histogram(steered_mass, edges)[0]
    baseline = np.histogram(baseline_mass, edges)[0]
    centre = np.sqrt(edges[1:] * edges[:-1])
    selected = (centre < max_mass) & (baseline >= min_baseline_pairs)
    if not selected.any():
        raise ValueError(f"no bin with its centre below {max_mass} GeV has {min_baseline_pairs} or more baseline "
                         f"pairs; the most in such a bin is {baseline[centre < max_mass].max(initial=0)}")
    return float(np.mean(np.abs(steered[selected] / baseline[selected] - 1)))


def target_shift(steered_mass: np.ndarray, baseline_mass: np.ndarray, in_target: np.ndarray) -> float:
    """Median of log m^steered minus median of log m^baseline over the pairs of the target class.

    steered_mass, baseline_mass: (E,) mass of each pair in GeV, same pairs in the same order. in_target: (E,) True for
    the pairs whose baseline mass of this seed is in the target class.
    """
    steered_mass, baseline_mass = checked_masses(steered_mass, baseline_mass)
    if in_target.dtype != bool or in_target.shape != baseline_mass.shape:
        raise ValueError(f"in_target: expected a boolean array of shape {baseline_mass.shape}, got {in_target.dtype} "
                         f"of shape {in_target.shape}")
    if not in_target.any():
        raise ValueError("in_target: no pair is in the target class")
    return float(np.median(np.log(steered_mass[in_target])) - np.median(np.log(baseline_mass[in_target])))


def selectivity_curve(seed: np.ndarray, alpha: np.ndarray, target_shifts: np.ndarray,
                      r60_values: np.ndarray) -> dict[str, np.ndarray]:
    """For each alpha, the target shift and R_60 as mean and standard deviation over seeds.

    seed, alpha, target_shifts, r60_values: (runs,) one entry per steered generation; the shifts and the R_60 values
    come from `target_shift` and `r60` against the baseline of the same seed. Every seed must have every alpha exactly
    once. The standard deviation divides by the number of seeds, as in the paper.

    Returns (alphas,) arrays sorted by alpha under "alpha", "target shift", "target shift std", "R_60", "R_60 std".
    """
    runs = len(seed)
    if any(len(values) != runs for values in (alpha, target_shifts, r60_values)):
        raise ValueError(f"seed, alpha, target_shifts, r60_values: expected one entry per run each, got lengths "
                         f"{runs}, {len(alpha)}, {len(target_shifts)}, {len(r60_values)}")
    seeds, alphas = np.unique(seed), np.unique(alpha)
    for this_seed in seeds:
        have = np.sort(alpha[seed == this_seed])
        if not np.array_equal(have, alphas):
            raise ValueError(f"seed {this_seed} has alphas {have.tolist()}, expected each of {alphas.tolist()} once")
    order = np.lexsort((seed, alpha))                  # by alpha, then by seed
    shifts = target_shifts[order].reshape(len(alphas), len(seeds))
    r60_table = r60_values[order].reshape(len(alphas), len(seeds))
    return {"alpha": alphas, "target shift": shifts.mean(1), "target shift std": shifts.std(1),
            "R_60": r60_table.mean(1), "R_60 std": r60_table.std(1)}
