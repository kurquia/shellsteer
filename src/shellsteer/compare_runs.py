"""Several finished steering runs side by side: paper App. C, Fig. 13 and Table 4.

    python -m shellsteer.compare_runs RUN [RUN ...] --out FOLDER

Each RUN is the folder of a steering run whose readout is done (runs/<name>/ with config.yaml and readout.json), in
legend order; the first is drawn in front. For App. C, the runs are the Z vector at block 1 of ShellFlow (Fig. 5
settings), ShellFlow-B, and ShellGATr (README, "App. C"). The command writes to FOLDER:

    pole_and_width.pdf  Fig. 13: fitted pole M and width Gamma against alpha of every run, seed mean with the seed
                        standard deviation (ddof 1) as error bars, as the paper's figure; dotted, each run's
                        baseline value times exp(alpha S).
    table.json          Table 4: per run, S = sum_k S_C(t_k) (Eq. 21), and the means over the Euler steps of
                        cos(G_C, xi) (Eq. 22) and of the FKT eigenvalue lambda_C (Eq. 18), from readout.json.

The legend label of a run is its plots.baseline_label. The runs must share the target and the block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shellsteer.config import Config
from shellsteer.plots import networks_pole_and_width_figure, save

FIGURE_FILE = "pole_and_width.pdf"
TABLE_FILE = "table.json"
TABLE_KEYS = ("gradient response", "mean alignment", "mean FKT eigenvalue")


class CompareError(Exception):
    """The runs cannot be compared; the message says why."""


def read_run(run: Path) -> tuple[Config, dict[str, Any]]:
    """The resolved config and the readout of a finished steering run."""
    for name in ("config.yaml", "readout.json"):
        if not (run / name).is_file():
            raise CompareError(f"{run / name} is missing; run `steer all` (or `steer readout`) for this run first")
    config = Config.model_validate(yaml.safe_load((run / "config.yaml").read_text()))
    if config.chain != "steering":
        raise CompareError(f"{run}: config {config.name} is a {config.chain} run, not a steering run")
    return config, json.loads((run / "readout.json").read_text())


def pole_and_width_series(run: Path, readout: dict[str, Any]) -> dict[str, np.ndarray]:
    """alpha, the seed mean of the fitted pole and width, and their seed standard deviation (ddof 1), from the
    lineshape fit of every seed and alpha in readout.json; alpha 0 is the fit of each seed's baseline."""
    rows = readout["per seed and alpha"]
    if any(row["lineshape"] is None for row in rows):
        raise CompareError(f"{run}: readout.json has no lineshape fits (the target is {readout['target']}); "
                           f"the figure needs the Z")
    seeds, alphas = readout["seeds"], sorted({row["alpha"] for row in rows})
    if len(seeds) < 2:
        raise CompareError(f"{run}: one seed; the error bars are the standard deviation over seeds, which needs two")
    fits = {(row["seed"], row["alpha"]): row["lineshape"] for row in rows}
    missing = [(seed, alpha) for seed in seeds for alpha in alphas if (seed, alpha) not in fits]
    if missing:
        raise CompareError(f"{run}: readout.json has no fit for (seed, alpha) {missing[0]}")
    out: dict[str, np.ndarray] = {"alpha": np.array(alphas, dtype=np.float64)}
    for key in ("pole", "width"):
        values = np.array([[fits[(seed, alpha)][key] for seed in seeds] for alpha in alphas])
        out[key], out[f"{key} std"] = values.mean(1), values.std(1, ddof=1)
    return out


def compare(runs: list[Path], out: Path) -> dict[str, Any]:
    """Write the figure and the table of `runs` to `out` and return the table."""
    if not runs:
        raise CompareError("name at least one run folder")
    read = [(run, *read_run(run)) for run in runs]
    first = read[0][1]
    for run, config, readout in read:
        if (config.target, readout["block"]) != (first.target, read[0][2]["block"]):
            raise CompareError(f"{run}: target {config.target} at block {readout['block']}, the first run has "
                               f"{first.target} at block {read[0][2]['block']}")
    labels = [config.plots.baseline_label for _, config, _ in read]
    if len(set(labels)) != len(labels):
        raise CompareError(f"the runs need different plots.baseline_label, got {labels}")
    series = {label: pole_and_width_series(run, readout) for label, (run, _, readout) in zip(labels, read)}
    slopes = {label: readout["gradient response"] for label, (_, _, readout) in zip(labels, read)}
    table = {"target": first.target, "block": read[0][2]["block"], "runs": [
        {"run": str(run), "label": label, "network": config.network, "method": config.steering_vectors.method,
         **{key: readout.get(key) for key in TABLE_KEYS},
         "unavailable": {key: readout.get(f"{key} unavailable") for key in TABLE_KEYS
                         if readout.get(key) is None}}
        for label, (run, config, readout) in zip(labels, read)]}
    out.mkdir(parents=True, exist_ok=True)
    save(networks_pole_and_width_figure(series, slopes), out / FIGURE_FILE)
    (out / TABLE_FILE).write_text(json.dumps(table, indent=1) + "\n")
    return table


def number(value: float | None, digits: int) -> str:
    return "null" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shellsteer.compare_runs", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN", help="run folders, such as runs/<name>")
    parser.add_argument("--out", type=Path, required=True, help="folder for pole_and_width.pdf and table.json")
    args = parser.parse_args(argv)
    try:
        table = compare(args.runs, args.out)
    except CompareError as error:
        print(f"compare_runs: {error}", file=sys.stderr)
        return 1
    print(f"{table['target']} at block {table['block']}: sum_k S, <cos(G, xi)>_k, <lambda>_k")
    for row in table["runs"]:
        print(f"  {row['network']:<12} {number(row['gradient response'], 3)}  {number(row['mean alignment'], 3)}  "
              f"{number(row['mean FKT eigenvalue'], 2)}  ({row['run']})")
    print(f"wrote {args.out / FIGURE_FILE} and {args.out / TABLE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
