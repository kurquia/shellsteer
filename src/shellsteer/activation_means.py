"""The mean of the residual stream on the pair's two tokens, by class, for the mean ablation (paper Sec. 5.3).

One forward pass without edits over the directions sample, with the noise of the gradient pass (seed
`gradients.seed`), records h_L on the two tokens of every pair at every Euler step, negative token first. Per class
it keeps "sum" (float64) and "count". `mean_projection` turns them into c_k of Eq. 25: the mean over the pairs of all
classes, projected on the unit steering vector. The run writes them to <run>/activation_means.npz.

A kept pair enters the class of the mass of this pass. Without edits this pass repeats the baseline of the same seed,
so on the same device type every kept pair enters the class it has in `kept`.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from shellsteer.config import Classes
from shellsteer.events import Events, batches, class_names, class_of, pair_mass
from shellsteer.hooks import Record, attach, on_step
from shellsteer.networks import Network
from shellsteer.sampler import batch_seed, generate, prior


def activation_means(network: Network, events: Events, kept: np.ndarray, classes: Classes, *, blocks: list[int],
                     steps: int, batch_size: int, seed: int, device: str | torch.device) -> dict[str, np.ndarray]:
    """Run the forward pass over every event of `events` and return the per-class sums of the module docstring.

    kept: (E,) class of each pair from `events.select`, "" for pairs left out.
    "sum" has shape (classes, blocks, steps, 2 width). A pair with a non-finite token enters no class and is counted
    in "non-finite pairs".
    """
    names = class_names(classes)
    width = 2 * network.model.d_model
    total = torch.zeros(len(names), len(blocks), steps, width, dtype=torch.float64, device=device)
    count = np.zeros(len(names), dtype=np.int64)
    non_finite = 0

    for index, batch, minus, plus in batches(events, batch_size, device):
        rows = slice(index * batch_size, index * batch_size + len(minus))
        records = [Record(network, steps, minus, plus) for _ in blocks]
        with ExitStack() as stack:
            for block, record in zip(blocks, records):
                stack.enter_context(attach(network, block, record))
            p4 = generate(network, batch, prior(network, batch, batch_seed(seed, index)), steps,
                          on_step=on_step(*records))
        finite = torch.stack([record.tokens.isfinite().all(-1).all(0) for record in records]).all(0)
        non_finite += int((~finite).sum())
        pair_class = class_of(pair_mass(p4, minus, plus).double().cpu().numpy(), classes)
        pair_class[(kept[rows] == "") | ~finite.cpu().numpy()] = ""
        for c, name in enumerate(names):
            members = torch.from_numpy(np.flatnonzero(pair_class == name)).to(device)
            count[c] += len(members)
            for b, record in enumerate(records):
                total[c, b] += record.tokens[:, members].double().sum(1)

    return {"classes": np.array(names), "blocks": np.array(blocks), "count": count,
            "non-finite pairs": np.array(non_finite), "sum": total.cpu().numpy()}


def mean_projection(means: dict[str, np.ndarray], xi: np.ndarray | torch.Tensor, block: int) -> np.ndarray:
    """(steps,) float64 c_k of Eq. 25: the mean of h_L over the pairs of every class, projected on xi(t_k).

    means: the sums of `activation_means`. xi: (steps, 2, width), as from `load_steering_vector`; it is projected
    after scaling to unit length in float64, the vector `hooks.Ablate` uses.
    """
    blocks = [int(b) for b in means["blocks"]]
    if block not in blocks:
        raise ValueError(f"block: {block} is not one of the blocks {blocks} of the activation means")
    count = int(np.sum(means["count"]))
    if count == 0:
        raise ValueError("count: no pairs in any class of the activation means")
    mean = means["sum"][:, blocks.index(block)].sum(0) / count                     # (steps, 2 width)
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 3 or xi.shape[1] != 2 or (xi.shape[0], 2 * xi.shape[2]) != mean.shape:
        raise ValueError(f"xi has shape {xi.shape}, expected (steps, 2, width) = ({mean.shape[0]}, 2, "
                         f"{mean.shape[1] // 2}) to match the activation means")
    unit = xi.reshape(len(xi), -1)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    return np.einsum("kw,kw->k", mean, unit)


def save_activation_means(path: str | Path, means: dict[str, np.ndarray], metadata: dict) -> None:
    """Write the sums and the settings that made them (config, network and dataset revisions) to one .npz."""
    np.savez(path, **means, metadata=np.array(json.dumps(metadata, indent=1)))


def load_activation_means(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        means = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return means, metadata
