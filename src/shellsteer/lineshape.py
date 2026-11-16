"""The lineshape fit of the Z peak (paper App. E), which turns a spectrum of dimuon masses into a pole M and a width
Gamma.

The model is the number of pairs per unit log m. Its Born part (paper Eq. 50) is

    dN_Born / dlog m = Y (M / m)^rho sum_q [L_q(m) / (L_up(M) + L_down(M))] A_0,q(m^2),

with the sum over the up-type and the down-type quarks. A_0,q is the bracket of Eq. 45: a photon, an interference,
and a Z term, with sin^2 theta_W fixed. L_q is the quark-antiquark luminosity times the acceptance of the muon
selection, from the package table data/luminosity_nnpdf40_13tev.csv (NNPDF4.0 NLO, 13 TeV), which luminosity.py
writes from the LHAPDF grid file. The Born density is folded with the radiator function F(y) of the final-state
radiation (Eq. 52), whose exponent b follows the pole (Eq. 53), and then with a Gaussian in log m of standard
deviation sigma_det / M (Eq. 51). Both folds of Eq. 54 are discrete convolutions on a grid in log m from 25
to 320 GeV with a step of 1e-3.

The fit (App. E.1) minimises the Poisson deviance D of the counts in bins of equal width in log m (Eq. 55) over five
free parameters: the pole M, the width Gamma, the resolution sigma_det, the luminosity exponent rho, and log Y. It
starts from four points and keeps the smallest D. `fit_counts` fits one histogram. `fit_masses` takes the masses of
a sample, centres the window on the peak, fits, centres the window on the fitted pole, and fits again.

The masses, in GeV, are those of the baseline generation, of a steered generation, or of the real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
from scipy.optimize import least_squares

SIN2_THETA_W = 0.23122          # sin^2 theta_W of the Particle Data Group 2018, fixed in the fit (App. E.1)
KAPPA = 1 / (4 * SIN2_THETA_W * (1 - SIN2_THETA_W))      # 1 / (4 s_W^2 c_W^2), the prefactor of chi (Eq. 44)
ALPHA_EM = 1 / 137.036          # alpha_em at zero momentum transfer, CODATA rounded, in b (Eq. 53)
MUON_MASS = 0.10566             # GeV, the Particle Data Group value rounded, in b (Eq. 53)
QUARK_TYPES = ((2 / 3, 0.5), (-1 / 3, -0.5))             # (charge, weak isospin) of up-type and of down-type quarks
MUON = (-1.0, -0.5)                                      # (charge, weak isospin) of the muon

GRID_STEP = 1e-3
LOG_MASS = np.arange(np.log(25.0), np.log(320.0) + GRID_STEP / 2, GRID_STEP)    # (grid,) log m, m in GeV
RADIATION_Y = np.arange(0.0, 0.8 + GRID_STEP / 2, GRID_STEP)                   # y from 0 to y_cut = 0.8 (Eq. 54)

LUMINOSITY_TABLE = "luminosity_nnpdf40_13tev.csv"

# The fit varies theta = (M, Gamma / M, sigma_det / M, rho, log Y) within these bounds. The fit window bounds M too.
# The names of the five, as `at_bound` gives them:
PARAMETERS = ("pole", "width", "resolution", "luminosity_exponent", "log_yield")
LOWER = np.array([25.0, 1e-4, 1e-4, -12.0, -60.0])
UPPER = np.array([200.0, 0.35, 0.35, 15.0, 60.0])
THETA_SCALE = np.array([1.0, 0.01, 0.01, 1.0, 1.0])     # x_scale of least_squares
# Gamma / M, sigma_det / M, rho, and log Y at the start of the fit of the baseline and of the real data (App. E.1).
BASELINE_START = np.array([0.019, 0.017, 2.5, 8.5])
# Three more starting points: Gamma and sigma_det of the start times these factors.
START_FACTORS = ((0.5, 1.6), (1.8, 0.6), (2.5, 2.5))


def read_luminosity_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(masses,) log m, log L_up, and log L_down of the package table, m in GeV."""
    text = files("shellsteer").joinpath("data", LUMINOSITY_TABLE).read_text()
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    if lines[0] != "mass_gev,L_up,L_down":
        raise ValueError(f"{LUMINOSITY_TABLE}: the header is {lines[0]!r}, expected 'mass_gev,L_up,L_down'")
    mass, up, down = np.loadtxt(lines[1:], delimiter=",", ndmin=2).T
    return np.log(mass), np.log(up), np.log(down)


TABLE_LOG_MASS, TABLE_LOG_UP, TABLE_LOG_DOWN = read_luminosity_table()


@dataclass(frozen=True)
class LineshapeFit:
    """One fit: the fitted parameters, the quality of the fit, and the histogram it fitted."""

    pole: float                  # M, GeV
    width: float                 # Gamma, GeV
    resolution: float            # sigma_det, GeV
    luminosity_exponent: float   # rho
    log_yield: float             # log Y
    deviance_per_dof: float      # D / (bins - 5)
    pairs_in_window: float       # sum of the counts
    at_bound: tuple[str, ...]    # names in PARAMETERS of the parameters at a bound, empty if none
    edges: np.ndarray            # (bins + 1,) bin edges in GeV
    counts: np.ndarray           # (bins,) pairs per bin
    expected: np.ndarray         # (bins,) expected count of each bin at the fitted parameters


def luminosity(log_mass: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """L_up and L_down with the acceptance at log m, in the unit of the table: log L interpolated linearly in log m."""
    return (np.exp(np.interp(log_mass, TABLE_LOG_MASS, TABLE_LOG_UP)),
            np.exp(np.interp(log_mass, TABLE_LOG_MASS, TABLE_LOG_DOWN)))


def vector_coupling(charge: float, isospin: float) -> float:
    """v_f = T^3_f - 2 Q_f sin^2 theta_W (Eq. 43)."""
    return isospin - 2 * charge * SIN2_THETA_W


def coupling_terms(charge: float, isospin: float) -> tuple[float, float, float]:
    """Photon, interference, and Z coefficient of A_0 (Eq. 45) for a quark of this charge and weak isospin.

    With s = m^2 and D = (s - M^2)^2 + M^2 Gamma^2 they multiply 1, s (s - M^2) / D, and s^2 / D.
    """
    v_quark, v_muon = vector_coupling(charge, isospin), vector_coupling(*MUON)
    a_quark, a_muon = isospin, MUON[1]
    return (charge**2 * MUON[0]**2, 2 * charge * MUON[0] * v_quark * v_muon * KAPPA,
            (v_quark**2 + a_quark**2) * (v_muon**2 + a_muon**2) * KAPPA**2)


COUPLING_TERMS = tuple(coupling_terms(charge, isospin) for charge, isospin in QUARK_TYPES)


def born_density(log_mass: np.ndarray, pole: float, width: float, luminosity_exponent: float) -> np.ndarray:
    """dN_Born / dlog m of Eq. 50 for Y = 1, at log_mass (n,), m in GeV. Pole and width in GeV. Returns (n,)."""
    s = np.exp(2 * log_mass)
    denominator = (s - pole**2) ** 2 + pole**2 * width**2
    interference, z = s * (s - pole**2) / denominator, s**2 / denominator
    bracket = sum(type_luminosity * (photon_term + interference_term * interference + z_term * z)
                  for type_luminosity, (photon_term, interference_term, z_term)
                  in zip(luminosity(log_mass), COUPLING_TERMS))
    return np.exp(-luminosity_exponent * (log_mass - np.log(pole))) * bracket / sum(luminosity(np.log(pole)))


def radiative_exponent(pole: float) -> float:
    """b = (2 alpha_em / pi) (log(M^2 / m_mu^2) - 1) of Eq. 53, at the pole M in GeV."""
    return 2 * ALPHA_EM / np.pi * (np.log(pole**2 / MUON_MASS**2) - 1)


def lineshape(pole: float, width: float, resolution: float, luminosity_exponent: float) -> np.ndarray:
    """(grid,) dN / dlog m of Eq. 54 on LOG_MASS for Y = 1. Pole, width, and resolution in GeV."""
    born = born_density(LOG_MASS, pole, width, luminosity_exponent)
    # The integral of F(y) = d/dy (1 - e^(-2y))^b over each step of y (Eq. 52).
    radiator = np.diff((1 - np.exp(-2 * RADIATION_Y)) ** radiative_exponent(pole))
    # A pair at log m after the radiation was at log m + y before it.
    radiated = np.convolve(born[::-1], radiator, "full")[:len(LOG_MASS)][::-1]
    sigma = resolution / pole
    # The Gaussian reaches one grid step beyond five standard deviations on each side.
    reach = int(5 * sigma / GRID_STEP)
    offsets = np.arange(-reach - 1, reach + 2) * GRID_STEP
    kernel = np.exp(-offsets**2 / (2 * sigma**2))
    return np.convolve(radiated, kernel / kernel.sum(), "same")


def expected_counts(edges: np.ndarray, pole: float, width: float, resolution: float, luminosity_exponent: float,
                    log_yield: float) -> np.ndarray:
    """(bins,) expected count nu_i of each bin: Y times the integral of the lineshape over the bin.

    edges: (bins + 1,) bin edges in GeV. Each edge moves up to the next grid point of LOG_MASS, as in the paper.
    """
    density = lineshape(pole, width, resolution, luminosity_exponent)
    cumulative = np.concatenate([[0.0], np.cumsum(density) * GRID_STEP])
    index = np.clip(np.searchsorted(LOG_MASS, np.log(edges)), 0, len(LOG_MASS) - 1)
    return np.exp(log_yield) * np.diff(cumulative[index])


def deviance_residuals(counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """(bins,) signed square roots of the terms of the deviance D (Eq. 55): their squares sum to D."""
    expected = np.maximum(expected, 1e-9)       # keeps n log(n / nu) finite where nu is zero
    term = expected - counts
    filled = counts > 0
    term[filled] += counts[filled] * np.log(counts[filled] / expected[filled])
    return np.sign(expected - counts) * np.sqrt(2 * np.maximum(term, 0.0))


def check_histogram(counts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """counts and edges as float64; stops unless they form a histogram of non-negative counts on the grid."""
    counts, edges = np.asarray(counts, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    if counts.ndim != 1 or edges.shape != (len(counts) + 1,):
        raise ValueError(f"counts have shape {counts.shape} and edges {edges.shape}, "
                         f"expected (bins,) and (bins + 1,)")
    if len(counts) <= len(PARAMETERS):
        raise ValueError(f"{len(counts)} bins, need more than the {len(PARAMETERS)} free parameters")
    if not (np.isfinite(counts).all() and (counts >= 0).all()):
        raise ValueError(f"counts must be finite and not negative, got the range [{counts.min()}, {counts.max()}]")
    if not (np.diff(edges) > 0).all():
        raise ValueError("edges must increase")
    grid_low, grid_high = np.exp(LOG_MASS[0]), np.exp(LOG_MASS[-1])
    if edges[0] < grid_low or edges[-1] > grid_high:
        raise ValueError(f"edges span {edges[0]:.4g} to {edges[-1]:.4g} GeV, the lineshape grid "
                         f"{grid_low:.4g} to {grid_high:.4g} GeV")
    return counts, edges


def fit_counts(counts: np.ndarray, edges: np.ndarray, start: LineshapeFit | None) -> LineshapeFit:
    """The lineshape fitted to one histogram of pair masses.

    counts: (bins,) pairs per bin. edges: (bins + 1,) bin edges in GeV, inside the grid from 25 to 320 GeV.
    start: the fit to start from, with its Gamma / M, sigma_det / M, rho, and log Y; each steered sample starts from
    the baseline fit of its seed. None starts from BASELINE_START, as the baseline and the real data do. The pole
    starts at the centre of the window in log m and stays inside the window.

    Stops with RuntimeError if the best of the four fits did not converge. A parameter at a bound is named in
    `at_bound`. The paper discards a fit with the pole at the edge of the window and keeps the others.
    """
    counts, edges = check_histogram(counts, edges)
    lower, upper = LOWER.copy(), UPPER.copy()
    lower[0], upper[0] = max(LOWER[0], edges[0]), min(UPPER[0], edges[-1])
    centre = np.sqrt(edges[0] * edges[-1])
    if start is None:
        initial = np.concatenate([[centre], BASELINE_START])
    else:
        initial = np.array([centre, start.width / start.pole, start.resolution / start.pole,
                            start.luminosity_exponent, start.log_yield])
    starts = [initial]
    for factors in START_FACTORS:
        other = initial.copy()
        other[1:3] = np.clip(initial[1:3] * factors, LOWER[1:3], UPPER[1:3])
        starts.append(other)

    def model(theta: np.ndarray) -> np.ndarray:
        pole, width_over_pole, resolution_over_pole, luminosity_exponent, log_yield = theta
        return expected_counts(edges, pole, width_over_pole * pole, resolution_over_pole * pole, luminosity_exponent,
                               log_yield)

    def residuals(theta: np.ndarray) -> np.ndarray:
        return deviance_residuals(counts, model(theta))

    fits = [least_squares(residuals, np.clip(theta, lower + 1e-9, upper - 1e-9), bounds=(lower, upper),
                          x_scale=THETA_SCALE, xtol=1e-10, ftol=1e-10) for theta in starts]
    best = min(fits, key=lambda fit: fit.cost)
    if best.status <= 0:
        raise RuntimeError(f"the lineshape fit of {counts.sum():.0f} pairs in {edges[0]:.4g} to {edges[-1]:.4g} GeV "
                           f"did not converge: {best.message}")
    theta = best.x
    # The paper's tolerances: 0.1 percent of M for the pole, 1e-6 for Gamma / M and sigma_det / M. The paper did not
    # check rho and log Y; they get 1e-6 too.
    tolerance = np.array([1e-3 * theta[0], 1e-6, 1e-6, 1e-6, 1e-6])
    near = np.minimum(theta - lower, upper - theta) < tolerance
    pole, width_over_pole, resolution_over_pole, luminosity_exponent, log_yield = (float(value) for value in theta)
    return LineshapeFit(pole=pole, width=width_over_pole * pole, resolution=resolution_over_pole * pole,
                        luminosity_exponent=luminosity_exponent, log_yield=log_yield,
                        deviance_per_dof=float(2 * best.cost / (len(counts) - len(PARAMETERS))),
                        pairs_in_window=float(counts.sum()),
                        at_bound=tuple(name for name, flag in zip(PARAMETERS, near) if flag),
                        edges=edges, counts=counts, expected=model(theta))


def window_edges(centre: float, *, half_width: float, bins: int) -> np.ndarray:
    """(bins + 1,) edges in GeV of bins of equal width in log m, from log centre - half_width to log centre +
    half_width. centre in GeV."""
    return np.exp(np.linspace(np.log(centre) - half_width, np.log(centre) + half_width, bins + 1))


def check_masses(mass: np.ndarray, baseline_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both as float64; stops unless they are finite masses of the same pairs."""
    mass, baseline_mass = np.asarray(mass, dtype=np.float64), np.asarray(baseline_mass, dtype=np.float64)
    if mass.ndim != 1 or mass.shape != baseline_mass.shape:
        raise ValueError(f"mass has shape {mass.shape} and baseline_mass {baseline_mass.shape}, expected (pairs,) "
                         f"for both")
    for name, values in (("mass", mass), ("baseline_mass", baseline_mass)):
        bad = ~np.isfinite(values)
        if bad.any():
            raise ValueError(f"{name}: not finite at {int(bad.sum())} of {len(values)} pairs, the first at index "
                             f"{int(np.flatnonzero(bad)[0])}")
    return mass, baseline_mass


def peak_centre(mass: np.ndarray, baseline_mass: np.ndarray, *, peak_window: tuple[float, float],
                min_window_pairs: int) -> float:
    """exp(median of log m) in GeV over the pairs whose baseline mass lies inside peak_window (GeV, open interval).

    mass, baseline_mass: (pairs,) in GeV, the same pairs. Stops if fewer than min_window_pairs pairs are inside.
    """
    mass, baseline_mass = check_masses(mass, baseline_mass)
    low, high = peak_window
    inside = (baseline_mass > low) & (baseline_mass < high)
    if inside.sum() < min_window_pairs:
        raise ValueError(f"{int(inside.sum())} pairs have a baseline mass in the window {low} to {high} GeV, "
                         f"min_window_pairs is {min_window_pairs}")
    return float(np.exp(np.median(np.log(mass[inside]))))


def fit_masses(mass: np.ndarray, baseline_mass: np.ndarray, start: LineshapeFit | None, *,
               peak_window: tuple[float, float], half_width: float, bins: int, min_window_pairs: int) -> LineshapeFit:
    """The lineshape fitted to a sample of pair masses in two steps, as in the paper (App. E.1).

    mass: (pairs,) masses of the sample in GeV. baseline_mass: (pairs,) masses of the same pairs in the baseline
    generation; pass `mass` itself for the baseline and for the real data. start: as in `fit_counts`.
    The first fit uses the window centred on `peak_centre`, the second the window centred on the pole of the first
    fit, starting from that fit. Every pair of the sample enters the histograms, whatever its baseline mass.
    peak_window: (low, high) in GeV, the window of the class in classes.yaml. half_width: half the width of the fit
    window in log m. bins: number of bins in the fit window.
    """
    mass, baseline_mass = check_masses(mass, baseline_mass)
    centre = peak_centre(mass, baseline_mass, peak_window=peak_window, min_window_pairs=min_window_pairs)
    edges = window_edges(centre, half_width=half_width, bins=bins)
    first = fit_counts(np.histogram(mass, edges)[0], edges, start)
    edges = window_edges(first.pole, half_width=half_width, bins=bins)
    return fit_counts(np.histogram(mass, edges)[0], edges, first)
