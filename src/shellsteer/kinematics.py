"""Kinematics of every particle type of a steering sample of every final state: paper Fig. 4 (the edited muons) and
Fig. 14 (every other type).

particle_kinematics  p_T, eta, phi, and E of each particle from its four-momentum.
real_mass            the real-data mass of each jet, large-R jet, and hadronic tau from the ShellFlow rows.
bin_edges            the paper's bins of one panel, from the real data and the baseline of one particle type.
type_panels          counts of real data, baseline, and each alpha on those bins, per type and variable.
mass_panels          the same for the mass m of the massive types (Fig. 14 only).
largest_changes      the readout of Sec. 5.1 and App. D.1: the largest |steered / baseline - 1| of a panel over the bins
                     with enough baseline particles.
save_panels          writes the counts to one .npz; load_panels reads them back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from shellflow.data.particle_vector_collator import _ETA_CLAMP, ParticleVectorCollator
from shellflow.models.riemannian_DiTs_geometry import MASSIVE_TYPES

VARIABLES = ("pt", "eta", "phi", "E")   # every type; the readout's largest change of Sec. 5.1 and App. D.1 uses these
MASS = "m"                    # the fifth column of Fig. 14, for the types of MASSIVE_TYPES only
LOG_VARIABLES = ("pt", "E")   # binned in log, bin centres geometric
TAIL = 1e-5                   # p_T and E bins start at this quantile; the eta range leaves out this fraction of |eta|
MARGIN = 1.15                 # room beyond the range, as a factor
ETA_STEP = 0.25               # the eta range is a multiple of this
MASS_QUANTILE = 0.9999        # m bins run from 0 to this quantile times MASS_MARGIN
MASS_MARGIN = 1.05
SLOTS = 10                    # particle slots per event, as the event files
MISSING = -999.0              # a missing feature in the ShellFlow rows


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


def real_mass(rows: np.ndarray, particle_type: np.ndarray, particle_mask: np.ndarray, batch_size: int = 65536
              ) -> np.ndarray:
    """(E, SLOTS) float32 real-data mass in GeV of each jet, large-R jet, and hadronic tau, NaN on other types and
    padding: sqrt(max(E^2 - (p_T cosh eta)^2, 0)) in float64 from p_T, eta, and E as the dataset stores them, before
    the conversion to (px, py, pz, E). This is the real data of the mass column of paper Fig. 14.

    rows: (E, 2, 266) float32 ShellFlow rows of the events, the `rows` key of the event files. particle_type,
    particle_mask: (E, SLOTS) of the same events. ShellFlow's collator places each feature in its particle slot; eta
    is clamped to +-5 and a particle without E takes E = p_T, as the collator does for the four-momenta. In float32
    four-momenta, E^2 - |p|^2 of a hadronic tau (m below 1 GeV, E of tens of GeV) is lost to cancellation.
    """
    if rows.ndim != 3 or rows.shape[1] != 2:
        raise ValueError(f"rows: expected ShellFlow rows (E, 2, L), got shape {rows.shape}")
    if np.shape(particle_type) != (len(rows), SLOTS) or np.shape(particle_mask) != (len(rows), SLOTS):
        raise ValueError(f"particle_type {np.shape(particle_type)}, particle_mask {np.shape(particle_mask)}: "
                         f"expected ({len(rows)}, {SLOTS}), one row per event of rows")
    collator = ParticleVectorCollator(max_particles=SLOTS)
    parts = []
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        features = torch.from_numpy(np.ascontiguousarray(block[:, 0, 2:])).float()
        packed = torch.from_numpy(np.ascontiguousarray(block[:, 1, 2:])).long()
        particle_idxs, feature_idxs = packed // 1000, packed % 1000          # as ParticleDataset.__getitem__
        features = torch.where(features == MISSING, torch.zeros_like(features), features)   # as its normaliser
        values, has = collator._scatter_all(features, feature_idxs, particle_idxs, feature_idxs != 0, len(block),
                                            SLOTS)
        pt = values[..., collator._S_PT]
        eta = values[..., collator._S_ETA].clamp(-_ETA_CLAMP, _ETA_CLAMP)
        energy = torch.where(has[..., collator._S_E], values[..., collator._S_E], pt)
        momentum = pt.numpy().astype(np.float64) * np.cosh(eta.numpy().astype(np.float64))
        parts.append(np.sqrt(np.clip(energy.numpy().astype(np.float64) ** 2 - momentum ** 2, 0.0, None)))
    massive = np.isin(particle_type, list(MASSIVE_TYPES)) & (np.asarray(particle_mask) > 0.5)
    return np.where(massive, np.concatenate(parts), np.nan).astype(np.float32)


def bin_edges(variable: str, values: np.ndarray, bins: int) -> np.ndarray:
    """(bins + 1,) edges of one panel of paper Figs. 4 and 14, from the finite `values` of the panel's real data and
    baseline together.

    p_T and E: equal bins in log from the TAIL quantile of the positive values divided by MARGIN to the largest value
    times MARGIN. eta: equal bins over [-a, a], with a the 1 - TAIL quantile of |eta| times MARGIN, rounded up to a
    multiple of ETA_STEP. phi: equal bins over [-pi, pi]. m: equal bins over [0, b], with b the MASS_QUANTILE
    quantile times MASS_MARGIN.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if variable == "phi":
        return np.linspace(-np.pi, np.pi, bins + 1)
    if variable == MASS:
        if len(values) == 0:
            raise ValueError("m: no finite value to bin")
        high = np.quantile(values, MASS_QUANTILE) * MASS_MARGIN
        if not high > 0:
            raise ValueError(f"m: the {MASS_QUANTILE} quantile is {high / MASS_MARGIN}, expected a positive mass")
        return np.linspace(0.0, high, bins + 1)
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
    raise ValueError(f"variable: {variable!r} is not one of {(*VARIABLES, MASS)}")


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


def mass_panels(real_mass: np.ndarray, baseline_mass: np.ndarray, steered_mass: dict[float, np.ndarray],
                masks: dict[int, np.ndarray], *, bins: int) -> dict[int, Panel]:
    """code -> Panel of the mass column of paper Fig. 14, for the types of `masks` in MASSIVE_TYPES.

    real_mass: (E, N) from `real_mass`. baseline_mass, steered_mass[alpha]: (E, N) decoded masses of the generated
    events (sampler.decoded_mass). masks: from `type_masks`, as for `type_panels`; every type needs a particle. The bins
    come from `bin_edges` of the real data and the baseline of each type.
    """
    for name, mass in (("real_mass", real_mass), *((f"steered_mass[{alpha:+g}]", m) for alpha, m in
                                                     steered_mass.items())):
        if np.shape(mass) != np.shape(baseline_mass):
            raise ValueError(f"{name} has shape {np.shape(mass)}, baseline_mass {np.shape(baseline_mass)}; "
                             f"expected the same events")
    panels = {}
    for code, mask in masks.items():
        if code not in MASSIVE_TYPES:
            continue
        if not mask.any():
            raise ValueError(f"particle type {code}: no particle in these events, so no bins")
        edges = bin_edges(MASS, np.concatenate([real_mass[mask], baseline_mass[mask]]), bins)
        count = lambda mass: np.histogram(mass[mask], edges)[0]
        panels[code] = Panel(edges=edges, real=count(real_mass), baseline=count(baseline_mass),
                             steered={alpha: count(steered_mass[alpha]) for alpha in sorted(steered_mass)})
    return panels


def steered_over_baseline(steered: np.ndarray, baseline: np.ndarray, min_baseline: int) -> np.ndarray:
    """(bins,) steered over baseline counts, NaN in bins with fewer than `min_baseline` baseline particles."""
    ratio = np.full(len(baseline), np.nan)
    np.divide(steered, baseline, out=ratio, where=(baseline >= min_baseline) & (baseline > 0))
    return ratio


def bin_centre(variable: str, edges: np.ndarray, index: int) -> float:
    """Centre of bin `index`: geometric for p_T and E, arithmetic for eta, phi, and m."""
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
