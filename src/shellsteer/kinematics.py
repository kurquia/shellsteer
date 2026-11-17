"""Kinematics of every particle type of a steering sample of every final state: paper Fig. 4 (the edited muons) and
Fig. 14 (every other type).

particle_kinematics  p_T, eta, phi, and E of each particle from its four-momentum.
bin_edges            the paper's bins of one panel, from the real data and the baseline of one particle type.
type_panels          counts of real data, baseline, and each alpha on those bins, per type and variable.
largest_changes      the readout of Sec. 5.1 and App. D.1: the largest |steered / baseline - 1| of a panel over the bins
                     with enough baseline particles.
save_panels          writes the counts to one .npz; load_panels reads them back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VARIABLES = ("pt", "eta", "phi", "E")
LOG_VARIABLES = ("pt", "E")   # binned in log, bin centres geometric
TAIL = 1e-5                   # p_T and E bins start at this quantile; the eta range leaves out this fraction of |eta|
MARGIN = 1.15                 # room beyond the range, as a factor
ETA_STEP = 0.25               # the eta range is a multiple of this


@dataclass(frozen=True)
class Panel:
    """Counts of one particle type and one variable on common bins."""
    edges: np.ndarray                 # (bins + 1,) increasing
    real: np.ndarray                  # (bins,) real data
    baseline: np.ndarray              # (bins,) generated without the edit
    steered: dict[float, np.ndarray]  # alpha -> (bins,) generated with the edit, increasing alpha


def particle_kinematics(p4: np.ndarray) -> dict[str, np.ndarray]:
    """p_T, eta, phi, and E in float64 of four-momenta (..., 4) in the order px, py, pz, E. eta is 0 where p_T is 0."""
    p4 = np.asarray(p4, dtype=np.float64)
    if p4.shape[-1] != 4:
        raise ValueError(f"p4: expected four-momenta (..., 4), got shape {p4.shape}")
    px, py, pz, energy = np.moveaxis(p4, -1, 0)
    pt = np.hypot(px, py)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    return {"pt": pt, "eta": eta, "phi": np.arctan2(py, px), "E": energy}


def bin_edges(variable: str, values: np.ndarray, bins: int) -> np.ndarray:
    """(bins + 1,) edges of one panel of paper Figs. 4 and 14, from the finite `values` of the panel's real data and
    baseline together.

    p_T and E: equal bins in log from the TAIL quantile of the positive values divided by MARGIN to the largest value
    times MARGIN. eta: equal bins over [-a, a], with a the 1 - TAIL quantile of |eta| times MARGIN, rounded up to a
    multiple of ETA_STEP. phi: equal bins over [-pi, pi].
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if variable == "phi":
        return np.linspace(-np.pi, np.pi, bins + 1)
    if variable in LOG_VARIABLES:
        values = values[values > 0]
        if len(values) == 0:
            raise ValueError(f"{variable}: no positive value to bin")
        low, high = np.quantile(values, TAIL), values.max()
        return np.logspace(np.log10(low / MARGIN), np.log10(high * MARGIN), bins + 1)
    if variable == "eta":
        if len(values) == 0:
            raise ValueError("eta: no finite value to bin")
        limit = ETA_STEP * np.ceil(np.quantile(np.abs(values), 1 - TAIL) * MARGIN / ETA_STEP - 1e-9)
        return np.linspace(-limit, limit, bins + 1)
    raise ValueError(f"variable: {variable!r} is not one of {VARIABLES}")


def type_masks(particle_type: np.ndarray, particle_mask: np.ndarray, codes: list[int]) -> dict[int, np.ndarray]:
    """code -> (E, N) True on the real particles of that ShellFlow type code."""
    real = np.asarray(particle_mask) > 0.5
    return {code: real & (np.asarray(particle_type) == code) for code in codes}


def counts(kinematics: dict[str, np.ndarray], masks: dict[int, np.ndarray], edges: dict[int, dict[str, np.ndarray]]
           ) -> dict[int, dict[str, np.ndarray]]:
    """code -> variable -> (bins,) number of particles of that type in each bin, from `particle_kinematics`."""
    return {code: {variable: np.histogram(kinematics[variable][mask], edges[code][variable])[0]
                   for variable in VARIABLES} for code, mask in masks.items()}


def type_panels(real_p4: np.ndarray, baseline_p4: np.ndarray, steered_p4: dict[float, np.ndarray],
                masks: dict[int, np.ndarray], *, bins: int) -> dict[int, dict[str, Panel]]:
    """code -> variable -> Panel of paper Figs. 4 and 14.

    real_p4, baseline_p4: (E, N, 4) four-momenta of the same events. steered_p4: alpha -> (E, N, 4). masks: from
    `type_masks`, the same for every sample; every type needs a particle. The bins of each panel come from
    `bin_edges` of the real data and the baseline of that type.
    """
    shapes = {"real_p4": np.shape(real_p4), **{f"steered_p4[{alpha:+g}]": np.shape(p4)
                                                for alpha, p4 in steered_p4.items()}}
    for name, shape in shapes.items():
        if shape != np.shape(baseline_p4):
            raise ValueError(f"{name} has shape {shape}, baseline_p4 {np.shape(baseline_p4)}; expected the same events")
    empty = [code for code, mask in masks.items() if not mask.any()]
    if empty:
        raise ValueError(f"particle types {empty}: no particle in these events, so no bins")
    real, baseline = particle_kinematics(real_p4), particle_kinematics(baseline_p4)
    edges = {code: {variable: bin_edges(variable, np.concatenate([real[variable][mask], baseline[variable][mask]]),
                                        bins)
                    for variable in VARIABLES} for code, mask in masks.items()}
    real_counts, baseline_counts = counts(real, masks, edges), counts(baseline, masks, edges)
    steered = {alpha: counts(particle_kinematics(steered_p4[alpha]), masks, edges) for alpha in sorted(steered_p4)}
    return {code: {variable: Panel(edges=edges[code][variable], real=real_counts[code][variable],
                                   baseline=baseline_counts[code][variable],
                                   steered={alpha: steered[alpha][code][variable] for alpha in steered})
                   for variable in VARIABLES} for code in masks}


def steered_over_baseline(steered: np.ndarray, baseline: np.ndarray, min_baseline: int) -> np.ndarray:
    """(bins,) steered over baseline counts, NaN in bins with fewer than `min_baseline` baseline particles."""
    ratio = np.full(len(baseline), np.nan)
    np.divide(steered, baseline, out=ratio, where=(baseline >= min_baseline) & (baseline > 0))
    return ratio


def bin_centre(variable: str, edges: np.ndarray, index: int) -> float:
    """Centre of bin `index`: geometric for p_T and E, arithmetic for eta and phi."""
    low, high = edges[index], edges[index + 1]
    return float(np.sqrt(low * high) if variable in LOG_VARIABLES else 0.5 * (low + high))


def largest_changes(variable: str, panel: Panel, min_baseline: int) -> dict[float, dict | None]:
    """alpha -> the largest |steered / baseline - 1| of the panel over the bins with at least `min_baseline` baseline
    particles, with its bin centre and baseline count; None if no bin has that many. Paper Sec. 5.1 quotes it for the
    muons with min_baseline 1,000."""
    out = {}
    for alpha, steered in panel.steered.items():
        change = np.abs(steered_over_baseline(steered, panel.baseline, min_baseline) - 1)
        if not np.isfinite(change).any():
            out[alpha] = None
            continue
        index = int(np.nanargmax(change))
        out[alpha] = {"largest change": float(change[index]), "bin centre": bin_centre(variable, panel.edges, index),
                      "baseline particles in the bin": int(panel.baseline[index])}
    return out


def save_panels(path: str | Path, panels: dict[int, dict[str, Panel]], metadata: dict) -> None:
    """Write the panels to one .npz: <code>/<variable>/edges, real, baseline, and alpha_<alpha:+g>. Creates the folder
    of `path` if needed."""
    arrays = {"metadata": np.array(json.dumps(metadata, indent=1))}
    for code, by_variable in panels.items():
        for variable, panel in by_variable.items():
            key = f"{code}/{variable}"
            arrays |= {f"{key}/edges": panel.edges, f"{key}/real": panel.real, f"{key}/baseline": panel.baseline}
            arrays |= {f"{key}/alpha_{alpha:+g}": steered for alpha, steered in panel.steered.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_panels(path: str | Path) -> tuple[dict[int, dict[str, Panel]], dict]:
    """The panels and the metadata from `save_panels`."""
    panels: dict[int, dict[str, Panel]] = {}
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata"]))
        keys = [key.split("/") for key in data.files if key != "metadata"]
        for code, variable in sorted({(int(parts[0]), parts[1]) for parts in keys}):
            prefix = f"{code}/{variable}"
            alphas = sorted(float(parts[2].removeprefix("alpha_")) for parts in keys
                            if parts[:2] == [str(code), variable] and parts[2].startswith("alpha_"))
            panels.setdefault(code, {})[variable] = Panel(
                edges=data[f"{prefix}/edges"], real=data[f"{prefix}/real"], baseline=data[f"{prefix}/baseline"],
                steered={alpha: data[f"{prefix}/alpha_{alpha:+g}"] for alpha in alphas})
    return panels, metadata
