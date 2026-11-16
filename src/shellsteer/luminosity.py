"""The luminosity table of the lineshape fit (paper App. E): the parton luminosity of Eq. 48 times the acceptance of
the muon selection, for the up-type and the down-type quarks of Eq. 50, at 401 masses from 20 to 320 GeV.

The integral runs over the rapidity of the pair, |Y| < 2.5, by the midpoint rule. The parton distributions are those
of NNPDF4.0 NLO, central member 0, read from its LHAPDF grid file. Between the points of the grid, x f is a cubic
spline in log x and log Q, as in the paper's analysis code, not the interpolation of the LHAPDF library.

To write the table, download the set from LHAPDF, unpack it, and pass the file of member 0:

    curl -O https://lhapdfsets.web.cern.ch/current/NNPDF40_nlo_as_01180.tar.gz
    tar xzf NNPDF40_nlo_as_01180.tar.gz
    python -m shellsteer.luminosity NNPDF40_nlo_as_01180/NNPDF40_nlo_as_01180_0000.dat \\
        src/shellsteer/data/luminosity_nnpdf40_13tev.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

SQRT_S = 13000.0                # GeV
ETA_MAX, PT_MIN = 2.5, 7.0      # the muon selection: |eta| < 2.5, pT > 7 GeV
RAPIDITY_POINTS = 2500          # midpoint rule over 0 <= y <= ETA_MAX
QUARK_TYPES = {"up": (2, 4), "down": (1, 3, 5)}         # Particle Data Group codes; the antiquark is the negative
TABLE_MASSES = np.exp(np.linspace(np.log(20.0), np.log(320.0), 401))   # GeV


@dataclass(frozen=True)
class Subgrid:
    """One subgrid of an LHAPDF lhagrid1 file: x f(x, Q) of each parton at the points x and Q."""

    x: np.ndarray                   # (n_x,)
    q: np.ndarray                   # (n_q,) GeV
    xf: dict[int, np.ndarray]       # parton code -> (n_x, n_q)


def read_lhagrid(path: str | Path) -> list[Subgrid]:
    """The subgrids of an LHAPDF lhagrid1 file of one member, in the order of the file.

    The file has a header, then subgrids separated by lines '---'. A subgrid has a line of x values, a line of Q
    values, a line of parton codes, and then one line per (x, Q) point, Q running fastest, with x f of each parton.
    """
    blocks = Path(path).read_text().split("\n---")
    if "Format: lhagrid1" not in blocks[0]:
        raise ValueError(f"{path}: not an LHAPDF lhagrid1 file (no 'Format: lhagrid1' in the header)")
    subgrids = []
    for block in blocks[1:]:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        x, q = (np.array(line.split(), dtype=np.float64) for line in lines[:2])
        codes = [int(code) for code in lines[2].split()]
        values = np.array([line.split() for line in lines[3:]], dtype=np.float64)
        if values.shape != (len(x) * len(q), len(codes)):
            raise ValueError(f"{path}: a subgrid has {values.shape} values, expected "
                             f"({len(x)} x {len(q)}, {len(codes)})")
        values = values.reshape(len(x), len(q), len(codes))
        subgrids.append(Subgrid(x, q, {code: values[:, :, i] for i, code in enumerate(codes)}))
    if not subgrids:
        raise ValueError(f"{path}: no subgrid")
    return subgrids


class PartonDistributions:
    """x f(x, Q) of one member, by a cubic spline in (log x, log Q) inside the subgrid that holds Q."""

    def __init__(self, path: str | Path):
        self.subgrids = read_lhagrid(path)
        self.splines = [{code: RectBivariateSpline(np.log(s.x), np.log(s.q), xf, kx=3, ky=min(3, len(s.q) - 1))
                         for code, xf in s.xf.items()} for s in self.subgrids]

    def xf(self, code: int, x: np.ndarray, q: float) -> np.ndarray:
        """x f(x, Q) of the parton `code`. At the edge Q shared by two subgrids, the upper subgrid gives it."""
        holding = [i for i, s in enumerate(self.subgrids) if s.q[0] <= q <= s.q[-1]]
        if not holding:
            raise ValueError(f"Q = {q} GeV is outside the grid, {self.subgrids[0].q[0]} to "
                             f"{self.subgrids[-1].q[-1]} GeV")
        subgrid, splines = self.subgrids[holding[-1]], self.splines[holding[-1]]
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < subgrid.x[0]) or np.any(x > subgrid.x[-1]):
            raise ValueError(f"x from {x.min()} to {x.max()} is outside the grid, {subgrid.x[0]} to {subgrid.x[-1]}")
        return splines[code](np.log(x), np.log(q), grid=False)


def acceptance(mass: float, y: np.ndarray) -> np.ndarray:
    """The share of the pairs of mass `mass` and rapidity y whose two muons pass |eta| < 2.5 and pT > 7 GeV, for a pair
    without transverse momentum and muons distributed as 1 + cos^2 theta."""
    c_max = np.minimum(np.tanh(np.clip(ETA_MAX - np.abs(y), 0, None)), np.sqrt(max(1 - 4 * PT_MIN ** 2 / mass ** 2, 0)))
    return 0.375 * 2 * (c_max + c_max ** 3 / 3)


def luminosity(partons: PartonDistributions, quarks: tuple[int, ...], mass: float) -> float:
    """The luminosity of Eq. 48 times the acceptance, summed over `quarks`, at the mass `mass` in GeV."""
    y = (np.arange(RAPIDITY_POINTS) + 0.5) * ETA_MAX / RAPIDITY_POINTS
    x1, x2 = mass / SQRT_S * np.exp(y), mass / SQRT_S * np.exp(-y)
    inside = x1 < 1                                     # f = 0 at x >= 1
    x1, x2 = x1[inside], x2[inside]
    total = np.zeros_like(y)
    for q in quarks:
        f1, f1_bar = partons.xf(q, x1, mass) / x1, partons.xf(-q, x1, mass) / x1
        f2, f2_bar = partons.xf(q, x2, mass) / x2, partons.xf(-q, x2, mass) / x2
        total[inside] += f1 * f2_bar + f1_bar * f2
    return float(2 * np.sum(total * acceptance(mass, y)) * ETA_MAX / RAPIDITY_POINTS)


def luminosity_table(partons: PartonDistributions, masses: np.ndarray = TABLE_MASSES) -> dict[str, np.ndarray]:
    """L_up and L_down at each mass, as {"mass_gev", "L_up", "L_down"}."""
    masses = np.asarray(masses, dtype=np.float64)
    return {"mass_gev": masses, **{f"L_{name}": np.array([luminosity(partons, quarks, m) for m in masses])
                                   for name, quarks in QUARK_TYPES.items()}}


def write_table(path: str | Path, table: dict[str, np.ndarray], grid: str | Path) -> None:
    """Write the table as the csv that `lineshape.read_luminosity_table` reads."""
    masses = table["mass_gev"]
    header = (f"# The luminosity table of the lineshape fit, from {Path(grid).name}.\n"
              "# Written by python -m shellsteer.luminosity; see shellsteer/luminosity.py.\n"
              "mass_gev,L_up,L_down\n")
    rows = "".join(f"{m:.6f},{up:.9e},{down:.9e}\n"
                   for m, up, down in zip(masses, table["L_up"], table["L_down"]))
    Path(path).write_text(header + rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m shellsteer.luminosity", description=__doc__.split("\n\n")[0])
    parser.add_argument("grid", type=Path, help="LHAPDF lhagrid1 file of one member, such as "
                                                "NNPDF40_nlo_as_01180_0000.dat")
    parser.add_argument("out", type=Path, help="the csv to write")
    args = parser.parse_args(argv)
    write_table(args.out, luminosity_table(PartonDistributions(args.grid)), args.grid)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
