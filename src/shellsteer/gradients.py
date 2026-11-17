"""The gradients of log m of the pair with respect to the residual stream, summed by class (paper Sec. 4.1).

For every pair, block L, and Euler step k, one backward pass through a zero `Probe` per block gives the gradient on
the two tokens of the pair, negative token first. The raw gradients do not fit on disk, so the pass keeps per class
the sums the steering vectors need, in float64: "unit second moment" (sum of u u^T for the unit gradients u), "unit
sum", "raw sum" (its direction fixes the sign), and "count". The same three sums of the step-summed gradient are
under "step-summed ..." keys, for vectors that are constant in time.

The class of a pair comes from the mass the network generates without edits (`steered_generation.generate_pairs`,
`events.select`). The pass runs on every event of the sample, so every batch has the noise of the baseline, and only
the kept pairs enter a class.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from shellsteer.config import Classes
from shellsteer.events import Events, batches, class_names, class_of, log_pair_mass, pair_mass_squared
from shellsteer.hooks import Probe, attach, on_step
from shellsteer.networks import Network
from shellsteer.sampler import NanCounts, OnNan, batch_seed, generate, prior


def unit(x: torch.Tensor) -> torch.Tensor:
    """x / |x| along the last axis; a zero vector stays zero."""
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, x / norm.clamp(min=1e-300), torch.zeros_like(x))


def gradient_moments(network: Network, events: Events, kept: np.ndarray, classes: Classes, *, blocks: list[int],
                     steps: int, batch_size: int, seed: int, device: str | torch.device,
                     on_nan: OnNan = "set_to_zero") -> tuple[dict[str, np.ndarray], NanCounts]:
    """Run the gradient pass over every event of `events`; return the per-class sums of the module docstring and the
    NaN counts of the generation (`sampler.generate` with `on_nan`).

    kept: (E,) class of each pair from `events.select`, "" for pairs left out.
    A kept pair enters the class of the mass of this pass, which is the baseline mass when the pass runs on the same
    device type. A pair whose gradient has a non-finite entry enters no class and is counted in "non-finite pairs".
    """
    names = class_names(classes)
    width = 2 * network.model.d_model
    n_classes, n_blocks = len(names), len(blocks)
    f64 = dict(dtype=torch.float64, device=device)
    second = torch.zeros(n_classes, n_blocks, steps, width, width, **f64)
    unit_sum = torch.zeros(n_classes, n_blocks, steps, width, **f64)
    raw_sum = torch.zeros(n_classes, n_blocks, steps, width, **f64)
    summed_second = torch.zeros(n_classes, n_blocks, width, width, **f64)
    summed_unit_sum = torch.zeros(n_classes, n_blocks, width, **f64)
    summed_raw_sum = torch.zeros(n_classes, n_blocks, width, **f64)
    count = np.zeros(n_classes, dtype=np.int64)
    non_finite = 0
    nans = []

    for index, batch, minus, plus in batches(events, batch_size, device):
        rows = slice(index * batch_size, index * batch_size + len(minus))
        probes = [Probe(network, steps, minus, plus) for _ in blocks]
        with ExitStack() as stack:
            for block, probe in zip(blocks, probes):
                stack.enter_context(attach(network, block, probe))
            generated = generate(network, batch, prior(network, batch, batch_seed(seed, index)), steps, on_nan=on_nan,
                                 label=f"batch {index} of the gradient pass, seed {seed}", grad=True,
                                 on_step=on_step(*probes))
            p4 = generated.four_momenta
            log_m = log_pair_mass(p4, minus, plus)
            grads = torch.autograd.grad(log_m.sum(), [probe.delta for probe in probes])

        finite = torch.stack([grad.isfinite().flatten(-2).all(-1).all(0) for grad in grads]).all(0)
        non_finite += int((~finite).sum())
        mass = pair_mass_squared(p4, minus, plus).detach().clamp(min=1e-6).sqrt()
        nans.append(NanCounts.of_batch(generated, mass))
        pair_class = class_of(mass.double().cpu().numpy(), classes)
        pair_class[(kept[rows] == "") | ~finite.cpu().numpy()] = ""
        members = [torch.from_numpy(np.flatnonzero(pair_class == name)).to(device) for name in names]
        count += [len(m) for m in members]

        for b, grad in enumerate(grads):                                       # one block at a time (memory)
            g = grad.flatten(-2).double()          # (steps, pairs, width): the two tokens side by side, negative first
            g = torch.where(finite[None, :, None], g, torch.zeros_like(g))
            summed = g.sum(0)                                                  # (pairs, width): G_p
            u, u_summed = unit(g), unit(summed)
            for c, m in enumerate(members):
                if len(m) == 0:
                    continue
                xu = u[:, m]                                                   # (steps, n, width)
                second[c, b] += xu.transpose(-1, -2) @ xu
                unit_sum[c, b] += xu.sum(1)
                raw_sum[c, b] += g[:, m].sum(1)
                xsu = u_summed[m]                                              # (n, width)
                summed_second[c, b] += xsu.T @ xsu
                summed_unit_sum[c, b] += xsu.sum(0)
                summed_raw_sum[c, b] += summed[m].sum(0)
            del g, summed, u, u_summed

    moments = {
        "classes": np.array(names), "blocks": np.array(blocks), "count": count,
        "non-finite pairs": np.array(non_finite),
        "unit second moment": second.cpu().numpy(), "unit sum": unit_sum.cpu().numpy(),
        "raw sum": raw_sum.cpu().numpy(),
        "step-summed unit second moment": summed_second.cpu().numpy(),
        "step-summed unit sum": summed_unit_sum.cpu().numpy(), "step-summed raw sum": summed_raw_sum.cpu().numpy(),
    }
    return moments, NanCounts(on_nan, events.event, np.concatenate(nans))


def save_moments(path: str | Path, moments: dict[str, np.ndarray], metadata: dict) -> None:
    """Write the sums and the settings that made them (config, network and dataset revisions) to one .npz."""
    np.savez(path, **moments, metadata=np.array(json.dumps(metadata, indent=1)))


def load_moments(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        moments = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return moments, metadata
