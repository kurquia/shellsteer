"""How selective a steering is, from the generated masses of the pairs: R_60 and the target shift (paper Sec. 5).

Each seed generates the kept pairs (`events.select`) once without edits and once per alpha, all from the same noise,
so every pair has a baseline mass and one steered mass per alpha. The functions compare the two masses of the same
pairs. Masses are in GeV, of the kept pairs only.

`r60` is R_60 (Eq. 19), `target_shift` the shift of the target peak in log m, and `selectivity_curve` their mean and
standard deviation over seeds for each alpha (App. B.1). `pole_response_fit` (Eq. 21) and `pair_response_fit`
(Eq. 20, App. D.3) fit the first-order response of the pole and of each pair; compare the slopes with
`steering_vectors.gradient_response`. The two fits differ as in the paper: log M for the pole, GeV for the pairs.

The numbers of the mean ablation (Sec. 5.3, Fig. 7) compare the ablated and the baseline generation of each seed:
`window_log_spread` (the spread of the Z peak, x1.18), `peak_height_change` (the Z peak, -28 %), and
`below_mass_deviation` (the spectrum below 60 GeV, 0.89 % on the seed-average histograms).
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

    Every bin with its geometric centre below `max_mass` and at least `min_baseline_pairs` baseline pairs has the same
    weight. The paper uses 200 bins equally spaced in log m over 0.3-300 GeV, max_mass = 60 GeV, and
    min_baseline_pairs = 20, which gives 153 bins. R_60 between the baselines of two seeds is the noise floor.
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


def checked_alphas(alpha: np.ndarray, values: int) -> tuple[np.ndarray, np.ndarray]:
    """alpha as (K,) float64 and the boolean mask of alpha != 0.

    Stops unless there are `values` alphas, all finite and different, exactly one of them 0, and at least two others.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) != values:
        raise ValueError(f"alpha: expected shape ({values},), one alpha per value, got {alpha.shape}")
    if not np.isfinite(alpha).all():
        raise ValueError(f"alpha: {int((~np.isfinite(alpha)).sum())} are not finite")
    if len(np.unique(alpha)) != len(alpha):
        raise ValueError(f"alpha: every alpha must appear once, got {sorted(alpha.tolist())}")
    steered = alpha != 0
    if steered.all():
        raise ValueError("alpha: no alpha is 0; the fit needs the baseline")
    if steered.sum() < 2:
        raise ValueError(f"alpha: the fit needs at least two alphas other than 0, got {alpha[steered].tolist()}")
    return alpha, steered


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """1 - sum (observed - predicted)^2 / sum (observed - mean observed)^2 along the first axis."""
    total = np.sum((observed - observed.mean(0)) ** 2, axis=0)
    if np.any(total == 0):
        raise ValueError(f"R^2 is undefined: {int(np.sum(total == 0))} series have the same value at every alpha")
    return 1 - np.sum((observed - predicted) ** 2, axis=0) / total


def pole_response_fit(alpha: np.ndarray, pole: np.ndarray) -> tuple[float, float]:
    """S_fit and R^2 of the fit M(alpha) = M(0) exp(alpha S) of the fitted poles (paper Sec. 5.2, Eq. 21).

    alpha: (K,) steering coefficients, one of them 0. pole: (K,) fitted pole M in GeV of each alpha; the one at
    alpha = 0 is M(0), the baseline fit, and stays fixed. S is the least-squares slope of log(M(alpha) / M(0))
    against alpha over the alphas other than 0, which has the closed form

        S_fit = sum over alpha of alpha log(M(alpha) / M(0)) / sum over alpha of alpha^2.

    R^2 compares M(alpha) with M(0) exp(alpha S_fit) in GeV over the same alphas. For the Z at block 1 the paper
    fits -3 <= alpha <= 3 in steps of 0.5 and finds S_fit = 0.1255 and R^2 = 0.9998 (App. B.2: difference of means
    0.9992, Fisher 0.979).
    """
    pole = np.asarray(pole, dtype=np.float64)
    if pole.ndim != 1:
        raise ValueError(f"pole: expected shape (alphas,), got {pole.shape}")
    alpha, steered = checked_alphas(alpha, len(pole))
    if not np.all(np.isfinite(pole) & (pole > 0)):
        raise ValueError(f"pole: every pole must be finite and positive, got {pole.tolist()}")
    a, baseline = alpha[steered], pole[~steered][0]
    s = float(np.sum(a * np.log(pole[steered] / baseline)) / np.sum(a ** 2))
    return s, float(r_squared(pole[steered], baseline * np.exp(a * s)))


def pair_response_fit(alpha: np.ndarray, masses: np.ndarray, *, tolerance: float = 1e-12,
                      max_iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """S_p and R^2_p of the fit m_p(alpha) = m_p(0) exp(alpha S_p) of every pair (paper App. D.3, Eq. 20).

    alpha: (K,) steering coefficients, one of them 0. masses: (K, pairs) mass of each pair in GeV at each alpha,
    from the same noise; the row at alpha = 0 is m_p(0), the baseline, and stays fixed. As in the paper, S_p
    minimises the sum of squares in GeV, sum over alpha of (m_p(alpha) - m_p(0) exp(alpha S_p))^2, and R^2_p is
    computed in GeV over all K alphas, alpha = 0 included. The paper fits the Z pairs over -3 <= alpha <= 3 in steps
    of 0.5 and finds a median R^2_p of 0.994.

    The fit is not linear in S_p. It starts from the least-squares slope of log(m_p(alpha) / m_p(0)) and takes Newton
    steps, or Gauss-Newton steps where the second derivative is not positive, halved while they increase the sum of
    squares by more than its rounding. It stops once no step is larger than `tolerance` (1 + |S_p|). The paper took
    30 Gauss-Newton steps, which leaves 0.04 % of its pairs short of the minimum: pairs whose mass leaves the Z at
    large |alpha|.

    Returns two (pairs,) arrays: S_p and R^2_p.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 2:
        raise ValueError(f"masses: expected shape (alphas, pairs), got {masses.shape}")
    alpha, steered = checked_alphas(alpha, len(masses))
    bad = ~(np.isfinite(masses) & (masses > 0))
    if bad.any():
        first = np.argwhere(bad)[0]
        raise ValueError(f"masses: {int(bad.sum())} are not finite and positive, the first at alpha "
                         f"{alpha[first[0]]} and pair {int(first[1])}")
    baseline = masses[~steered][0]
    a = alpha[:, None]

    def sum_of_squares(s: np.ndarray) -> np.ndarray:
        return np.sum((masses - baseline * np.exp(a * s)) ** 2, axis=0)

    s = np.sum(a * np.log(masses / baseline), axis=0) / np.sum(alpha ** 2)
    for _ in range(max_iterations):
        predicted = baseline * np.exp(a * s)
        residual = masses - predicted
        gradient = -np.sum(a * predicted * residual, axis=0)          # half the derivative of the sum of squares
        gauss_newton = np.sum((a * predicted) ** 2, axis=0)
        second = gauss_newton - np.sum(a ** 2 * predicted * residual, axis=0)
        step = -gradient / np.where(second > 0, second, gauss_newton)
        moving = np.abs(step) > tolerance * (1 + np.abs(s))
        if not moving.any():
            break
        allowed = sum_of_squares(s) * (1 + 1e-12)                       # room for the rounding of the sum
        for _ in range(60):
            worse = moving & ~(sum_of_squares(s + step) <= allowed)
            if not worse.any():
                break
            step[worse] /= 2
        s = np.where(moving, s + step, s)
    else:
        raise ValueError(f"the fit did not converge in {max_iterations} iterations for {int(moving.sum())} pairs, "
                         f"the first at pair {int(np.flatnonzero(moving)[0])}")
    return s, r_squared(masses, baseline * np.exp(a * s))


def window_log_spread(mass: np.ndarray, window: tuple[float, float]) -> float:
    """Standard deviation of log m over the pairs with mass inside the open `window` in GeV (paper Sec. 5.3).

    The spread of the Z peak: the paper's ratio x1.177 is the mean over three seeds of this spread of the ablated
    masses over that of the baseline masses, each over its own pairs inside the window. Divides by the number of
    pairs, as numpy's std.
    """
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1:
        raise ValueError(f"mass: expected shape (pairs,), got {mass.shape}")
    low, high = window
    inside = mass[(mass > low) & (mass < high)]
    if len(inside) < 2:
        raise ValueError(f"window [{low}, {high}] GeV: {len(inside)} pairs inside, the spread needs at least 2")
    return float(np.log(inside).std())


def peak_height_change(steered_counts: np.ndarray, baseline_counts: np.ndarray) -> float:
    """Highest bin of `steered_counts` over highest bin of `baseline_counts`, minus 1 (paper Sec. 5.3).

    The paper's -28 % for the Z under the mean ablation uses the Z zoom of Fig. 7 (50-140 GeV, 100 equal bins) of the
    histograms averaged over three seeds. The two highest bins need not be the same bin.
    """
    steered_counts = np.asarray(steered_counts, dtype=np.float64)
    baseline_counts = np.asarray(baseline_counts, dtype=np.float64)
    if steered_counts.ndim != 1 or steered_counts.shape != baseline_counts.shape:
        raise ValueError(f"counts: expected two histograms of the same bins, got shapes {steered_counts.shape} and "
                         f"{baseline_counts.shape}")
    if baseline_counts.max(initial=0) <= 0:
        raise ValueError("baseline_counts: no pair in any bin")
    return float(steered_counts.max() / baseline_counts.max() - 1)


def below_mass_deviation(steered_counts: np.ndarray, baseline_counts: np.ndarray, *, edges: np.ndarray,
                         max_mass: float, min_baseline_pairs: float) -> float:
    """Mean of |N_b / N_b^baseline - 1| over the bins whose lower edge is below `max_mass` and whose baseline holds at
    least `min_baseline_pairs`, as a fraction (paper Sec. 5.3).

    The statistic of Eq. 19 on histograms, which may be averages over seeds. The paper's 0.89 % for the mean ablation
    uses the seed-average histograms of three seeds, 200 bins equally spaced in log m over 0.3-300 GeV, max_mass =
    60 GeV, and min_baseline_pairs = 20, which gives 154 bins. Unlike `r60`, which selects bins by their centre (153
    bins), the bins are selected by their lower edge, as in the script that made the paper's number.
    """
    steered_counts = np.asarray(steered_counts, dtype=np.float64)
    baseline_counts = np.asarray(baseline_counts, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    shapes_agree = steered_counts.shape == baseline_counts.shape and edges.shape == (len(steered_counts) + 1,)
    if steered_counts.ndim != 1 or not shapes_agree:
        raise ValueError(f"counts and edges: expected two histograms of (bins,) and edges of (bins + 1,), got shapes "
                         f"{steered_counts.shape}, {baseline_counts.shape}, and {edges.shape}")
    if min_baseline_pairs <= 0:
        raise ValueError(f"min_baseline_pairs: need more than 0, got {min_baseline_pairs}")
    selected = (edges[:-1] < max_mass) & (baseline_counts >= min_baseline_pairs)
    if not selected.any():
        raise ValueError(f"no bin with its lower edge below {max_mass} GeV has {min_baseline_pairs} or more baseline "
                         f"pairs")
    return float(np.mean(np.abs(steered_counts[selected] / baseline_counts[selected] - 1)))
