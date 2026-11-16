import numpy as np
import pytest

from shellsteer.lineshape import TABLE_LOG_MASS
from shellsteer.luminosity import (ETA_MAX, QUARK_TYPES, RAPIDITY_POINTS, SQRT_S, TABLE_MASSES, PartonDistributions,
                                   acceptance, luminosity, luminosity_table, read_lhagrid, write_table)

CODES = (-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5)
X = np.geomspace(1e-6, 1.0, 60)
LOWER_Q, UPPER_Q = np.geomspace(1.0, 5.0, 6), np.geomspace(5.0, 1000.0, 20)
UPPER_FACTOR = 2.0      # x f of the upper subgrid is twice the lower one, so the tests can tell which one answers


def xf(code: int, x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """A cubic in log x times a quadratic in log Q, which the cubic spline reproduces exactly."""
    lx, lq = np.log(x), np.log(q)
    return (1 + 0.1 * code ** 2) * (2.0 + 0.3 * lx + 0.05 * lx ** 2 + 0.002 * lx ** 3) * (1 + 0.2 * lq + 0.01 * lq ** 2)


def subgrid_text(q: np.ndarray, factor: float) -> str:
    rows = "\n".join(" ".join(f"{factor * xf(code, xi, qi):.17e}" for code in CODES) for xi in X for qi in q)
    return " ".join(f"{v:.17e}" for v in X) + "\n" + " ".join(f"{v:.17e}" for v in q) + "\n" + \
        " ".join(str(code) for code in CODES) + "\n" + rows + "\n"


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "test_0000.dat"
    path.write_text("PdfType: central\nFormat: lhagrid1\n---\n" + subgrid_text(LOWER_Q, 1.0) + "---\n"
                    + subgrid_text(UPPER_Q, UPPER_FACTOR) + "---\n")
    return path


def test_the_grid_file_reads_into_its_subgrids(grid):
    subgrids = read_lhagrid(grid)
    assert len(subgrids) == 2
    assert np.allclose(subgrids[1].q, UPPER_Q, rtol=1e-15) and sorted(subgrids[1].xf) == sorted(CODES)
    assert subgrids[1].xf[2].shape == (len(X), len(UPPER_Q))
    assert np.allclose(subgrids[1].xf[2][7, 3], UPPER_FACTOR * xf(2, X[7], UPPER_Q[3]), rtol=1e-15)


def test_the_spline_is_exact_for_a_cubic_and_the_upper_subgrid_holds_the_shared_edge(grid):
    partons = PartonDistributions(grid)
    x = np.geomspace(2e-6, 0.9, 37)                     # between the grid points
    for q, factor in ((2.3, 1.0), (5.0, UPPER_FACTOR), (91.1876, UPPER_FACTOR)):
        assert np.allclose(partons.xf(2, x, q), factor * xf(2, x, q), rtol=1e-10)


def test_luminosity_is_the_midpoint_rule_of_the_module_docstring(grid):
    partons = PartonDistributions(grid)
    mass = 91.1876
    y = (np.arange(RAPIDITY_POINTS) + 0.5) * ETA_MAX / RAPIDITY_POINTS
    x1, x2 = mass / SQRT_S * np.exp(y), mass / SQRT_S * np.exp(-y)
    expected = 0.0
    for q in QUARK_TYPES["up"]:
        f1, f1_bar = UPPER_FACTOR * xf(q, x1, mass) / x1, UPPER_FACTOR * xf(-q, x1, mass) / x1
        f2, f2_bar = UPPER_FACTOR * xf(q, x2, mass) / x2, UPPER_FACTOR * xf(-q, x2, mass) / x2
        expected += 2 * np.sum((f1 * f2_bar + f1_bar * f2) * acceptance(mass, y)) * ETA_MAX / RAPIDITY_POINTS
    assert luminosity(partons, QUARK_TYPES["up"], mass) == pytest.approx(expected, rel=1e-10)


def test_acceptance_limits():
    y = np.array([0.0, 1.0, 2.5, 3.0])
    assert np.allclose(acceptance(1000.0, y)[2:], 0.0)
    c = np.tanh(2.5)                                    # at y = 0 and a large mass, eta alone cuts
    assert acceptance(1e6, np.array([0.0]))[0] == pytest.approx(0.75 * (c + c ** 3 / 3), rel=1e-9)
    assert np.allclose(acceptance(13.9, y), 0.0)        # below 2 x 7 GeV no muon passes pT > 7 GeV
    assert np.all(np.diff(acceptance(91.0, np.array([0.0, 1.0, 2.0]))) < 0)


def test_values_outside_the_grid_stop(grid):
    partons = PartonDistributions(grid)
    with pytest.raises(ValueError, match="Q = 0.5 GeV is outside the grid"):
        partons.xf(2, np.array([0.1]), 0.5)
    with pytest.raises(ValueError, match="outside the grid, 1e-06 to 1.0"):
        partons.xf(2, np.array([1e-7]), 91.0)


def test_a_file_that_is_not_lhagrid1_stops(tmp_path):
    path = tmp_path / "other.dat"
    path.write_text("PdfType: central\nFormat: lhagrid2\n---\n")
    with pytest.raises(ValueError, match="not an LHAPDF lhagrid1 file"):
        read_lhagrid(path)


def test_the_written_table_reads_back_with_the_columns_the_fit_reads(grid, tmp_path):
    masses = np.geomspace(20.0, 320.0, 5)
    table = luminosity_table(PartonDistributions(grid), masses)
    path = tmp_path / "table.csv"
    write_table(path, table, grid)
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    assert lines[0] == "mass_gev,L_up,L_down"
    mass, up, down = np.loadtxt(lines[1:], delimiter=",").T
    assert np.allclose(mass, masses, rtol=1e-6)
    assert np.allclose(up, table["L_up"], rtol=1e-9) and np.allclose(down, table["L_down"], rtol=1e-9)
    assert "test_0000.dat" in path.read_text()


def test_the_package_table_has_the_masses_this_module_writes():
    assert np.allclose(np.exp(TABLE_LOG_MASS), TABLE_MASSES, rtol=1e-6)
