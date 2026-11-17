"""Steering vectors xi from the per-class sums of the gradients (the output of gradients.py), paper Sec. 4.2.

For a target class C, the other classes form C' as a mixture with weight w_D = 1 / (number of other classes) for
`balance: equal` and n_D / (pairs in C') for `balance: by_count`. The methods:

fkt:      the Fukunaga-Koontz direction (Eq. 14). The paper's Sigma_C' is the sum over the other classes, which is
          (number of other classes) times the mixture here; this scale does not change xi.
fisher:   xi = (K_C + K_C')^(-1) (m_C - m_C'), with K the covariance of the unit gradients.
meandiff: xi = m_C - m_C'.

Every vector has unit length over both tokens, and its sign makes G_C(t_k)^T xi(t_k) > 0 for the mean raw gradient.
`per_step` solves at every step; `constant` solves once with the step-summed gradient and uses it at every step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.linalg
import torch

# (classes, blocks, steps) S_C(t_k) in a vectors file; the sum over k is S (Eq. 21). Files from before it lack it.
GRADIENT_RESPONSE_KEY = "gradient response per step"
NETWORK_KEYS = ("hf_repo", "revision", "file")  # the network entries a vectors file is checked on

Method = Literal["fkt", "fisher", "meandiff"]
InTime = Literal["per_step", "constant"]
Balance = Literal["equal", "by_count"]


def others_weights(count: np.ndarray, target: int, balance: Balance) -> np.ndarray:
    """(classes,) weight w_D of each class in the mixture C'; zero for the target, the others sum to one."""
    if balance == "equal":
        weights = np.ones(len(count))
    elif balance == "by_count":
        weights = count.astype(np.float64)
    else:
        raise ValueError(f"balance: {balance!r} is not equal or by_count")
    weights[target] = 0.0
    return weights / weights.sum()


def positive_definite_eigh(matrix: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of a symmetric matrix; stops if it is not positive definite in float64."""
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
    tolerance = eigenvalues[-1] * len(matrix) * np.finfo(np.float64).eps
    if eigenvalues[0] <= tolerance:
        raise ValueError(f"{what} is not positive definite: smallest eigenvalue {eigenvalues[0]:.3g}, "
                         f"need more than {tolerance:.3g}")
    return eigenvalues, eigenvectors


def fkt(second: np.ndarray, second_others: np.ndarray, where: str) -> np.ndarray:
    """(2 width,) top FKT direction of Sigma_C = `second` against Sigma_C' = `second_others`, both square."""
    sigma, u = positive_definite_eigh(second + second_others, f"Sigma_C + Sigma_C' ({where})")
    m_fk = u / np.sqrt(sigma)
    transformed = m_fk.T @ second @ m_fk
    _, top = scipy.linalg.eigh(transformed, subset_by_index=[len(sigma) - 1, len(sigma) - 1])
    return m_fk @ top[:, 0]


def fisher(mean: np.ndarray, second: np.ndarray, mean_others: np.ndarray, second_others: np.ndarray,
           where: str) -> np.ndarray:
    """(2 width,) (K_C + K_C')^(-1) (m_C - m_C'), with K = Sigma - m m^T."""
    covariance = second - np.outer(mean, mean) + second_others - np.outer(mean_others, mean_others)
    sigma, u = positive_definite_eigh(covariance, f"K_C + K_C' ({where})")
    return u @ ((u.T @ (mean - mean_others)) / sigma)


def direction(method: Method, second: np.ndarray, unit_sum: np.ndarray, count: np.ndarray, target: int,
              balance: Balance, where: str) -> np.ndarray:
    """(2 width,) direction of `target` before the sign and the length are fixed.

    second: (classes, 2 width, 2 width) sums of hat G hat G^T. unit_sum: (classes, 2 width) sums of hat G.
    """
    weights = others_weights(count, target, balance)
    means = unit_sum / count[:, None]
    seconds = second / count[:, None, None]
    mean_others = np.einsum("c,cw->w", weights, means)
    if method == "meandiff":
        return means[target] - mean_others
    second_others = np.einsum("c,cvw->vw", weights, seconds)
    if method == "fkt":
        return fkt(seconds[target], second_others, where)
    if method == "fisher":
        return fisher(means[target], seconds[target], mean_others, second_others, where)
    raise ValueError(f"method: {method!r} is not fkt, fisher, or meandiff")


def steering_vector(moments: dict[str, np.ndarray], target: str, block: int, method: Method, in_time: InTime,
                    balance: Balance) -> np.ndarray:
    """(steps, 2, width) steering vector xi(t_k) of class `target` at `block`, negative token first.

    Unit length per step over both tokens, sign from the raw class mean. For `constant` the same vector at every step.
    """
    names = [str(name) for name in moments["classes"]]
    blocks = [int(b) for b in moments["blocks"]]
    if target not in names:
        raise ValueError(f"target: {target!r} is not one of the classes {names} of the moments")
    if block not in blocks:
        raise ValueError(f"block: {block} is not one of the blocks {blocks} of the moments")
    count = moments["count"].astype(np.float64)
    if (count == 0).any():
        raise ValueError(f"count: no pairs in class {names[int(np.argmin(count))]!r}, counts {dict(zip(names, count))}")
    c, b = names.index(target), blocks.index(block)
    steps, entries = moments["unit sum"].shape[2:]          # entries = 2 width

    if in_time == "per_step":
        second, unit_sum, raw_sum = (moments[key][:, b] for key in ("unit second moment", "unit sum", "raw sum"))
    elif in_time == "constant":     # one step axis of length 1
        second, unit_sum, raw_sum = (moments[f"step-summed {key}"][:, b, None]
                                     for key in ("unit second moment", "unit sum", "raw sum"))
    else:
        raise ValueError(f"in_time: {in_time!r} is not per_step or constant")

    vectors = np.stack([direction(method, second[:, k], unit_sum[:, k], count, c, balance,
                                  f"class {target}, block {block}, step {k}") for k in range(second.shape[1])])
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    projection = np.einsum("kw,kw->k", raw_sum[c], vectors)
    if (projection == 0).any():
        raise ValueError(f"class {target}, block {block}: the raw class mean is orthogonal to xi at step "
                         f"{int(np.flatnonzero(projection == 0)[0])}, the sign is undefined")
    vectors *= np.sign(projection)[:, None]
    return np.broadcast_to(vectors, (steps, entries)).reshape(steps, 2, entries // 2).copy()


def steering_vectors(moments: dict[str, np.ndarray], method: Method, in_time: InTime,
                     balance: Balance) -> np.ndarray:
    """(classes, blocks, steps, 2, width) steering vectors of every class and block of the moments."""
    return np.stack([np.stack([steering_vector(moments, str(name), int(block), method, in_time, balance)
                               for block in moments["blocks"]]) for name in moments["classes"]])


def save_steering_vectors(path: str | Path, vectors: np.ndarray, classes: np.ndarray, blocks: np.ndarray,
                          metadata: dict, response_per_step: np.ndarray | None = None) -> None:
    """Write the vectors (classes, blocks, steps, 2, width), their classes and blocks, and the settings to one .npz.

    response_per_step: (classes, blocks, steps) S_C(t_k) of each vector (`gradient_responses_per_step`), stored so
    that a run started from this file has S, the sum over k (Eq. 21), without the gradient pass. None stores none.
    """
    if vectors.shape[:2] != (len(classes), len(blocks)):
        raise ValueError(f"vectors have shape {vectors.shape}, expected ({len(classes)} classes, {len(blocks)} "
                         f"blocks, steps, 2, width)")
    arrays = {"steering vectors": vectors, "classes": np.asarray(classes), "blocks": np.asarray(blocks),
              "metadata": np.array(json.dumps(metadata, indent=1))}
    if response_per_step is not None:
        if np.shape(response_per_step) != vectors.shape[:3]:
            raise ValueError(f"response_per_step has shape {np.shape(response_per_step)}, expected "
                             f"{vectors.shape[:3]} (classes, blocks, steps)")
        arrays[GRADIENT_RESPONSE_KEY] = np.asarray(response_per_step, dtype=np.float64)
    np.savez(path, **arrays)


def gradient_response_per_step(moments: dict[str, np.ndarray], xi: np.ndarray | torch.Tensor, target: str,
                                block: int) -> np.ndarray:
    """(steps,) gradient response S_C(t_k) = G_C(t_k)^T xi(t_k) at each Euler step k; the sum over k is S (Eq. 21).

    G_C(t_k) = raw sum[C, block, k] / count[C] is the mean of the raw gradients of log m over the pairs of class C.
    To first order in alpha, the pole moves as M(alpha) = M(0) exp(alpha S). The paper's S for the Z at block 1 is
    0.122.

    moments: the sums of gradients.py. xi: (steps, 2, width) steering vector, negative token first, as from
    `load_steering_vector`; the raw sums hold the two tokens side by side in the same order. A vector of in_time
    `constant` has the same entries at every step, so it has the steps of the moments too.
    """
    names = [str(name) for name in moments["classes"]]
    blocks = [int(b) for b in moments["blocks"]]
    if target not in names:
        raise ValueError(f"target: {target!r} is not one of the classes {names} of the moments")
    if block not in blocks:
        raise ValueError(f"block: {block} is not one of the blocks {blocks} of the moments")
    c, b = names.index(target), blocks.index(block)
    count = int(moments["count"][c])
    if count == 0:
        raise ValueError(f"count: no pairs in class {target!r}")
    raw_sum = np.asarray(moments["raw sum"][c, b], dtype=np.float64)        # (steps, 2 width)
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 3 or xi.shape[1] != 2 or (xi.shape[0], 2 * xi.shape[2]) != raw_sum.shape:
        raise ValueError(f"xi has shape {xi.shape}, expected (steps, 2, width) = ({raw_sum.shape[0]}, 2, "
                         f"{raw_sum.shape[1] // 2}) to match the moments")
    if not (np.isfinite(xi).all() and np.isfinite(raw_sum).all()):
        raise ValueError(f"xi or the raw sum of class {target!r} at block {block} has a non-finite entry")
    return np.einsum("kw,kw->k", raw_sum, xi.reshape(len(xi), -1)) / count


def gradient_response(moments: dict[str, np.ndarray], xi: np.ndarray | torch.Tensor, target: str, block: int) -> float:
    """S of Eq. 21: the sum over the Euler steps k of `gradient_response_per_step`."""
    return float(gradient_response_per_step(moments, xi, target, block).sum())


def load_steering_vector(path: str | Path, target: str, block: int) -> torch.Tensor:
    """(steps, 2, width) float32 steering vector of class `target` at `block`, for `hooks.Steer`."""
    with np.load(path) as data:
        names = [str(name) for name in data["classes"]]
        blocks = [int(b) for b in data["blocks"]]
        if target not in names:
            raise ValueError(f"{path} has no class {target!r}, only {names}")
        if block not in blocks:
            raise ValueError(f"{path} has no block {block}, only {blocks}")
        vector = data["steering vectors"][names.index(target), blocks.index(block)]
    return torch.from_numpy(vector.astype(np.float32))



def gradient_responses_per_step(moments: dict[str, np.ndarray], vectors: np.ndarray) -> np.ndarray:
    """(classes, blocks, steps) S_C(t_k) of the vector of each class and block of `steering_vectors(moments, ...)`.

    The sum over the last axis is S of Eq. 21.
    """
    return np.array([[gradient_response_per_step(moments, vectors[c, b], str(name), int(block))
                      for b, block in enumerate(moments["blocks"])] for c, name in enumerate(moments["classes"])])


def load_gradient_response_per_step(path: str | Path, target: str, block: int) -> np.ndarray | None:
    """(steps,) S_C(t_k) of the vector of class `target` at `block` stored in `path`; its sum is S (Eq. 21).

    None if the file stores no response: a file written before `steer steering-vectors` stored it.
    """
    with np.load(path) as data:
        if GRADIENT_RESPONSE_KEY not in data.files:
            return None
        names = [str(name) for name in data["classes"]]
        blocks = [int(b) for b in data["blocks"]]
        if target not in names or block not in blocks:
            raise ValueError(f"{path} has no class {target!r} at block {block}; classes {names}, blocks {blocks}")
        return data[GRADIENT_RESPONSE_KEY][names.index(target), blocks.index(block)].astype(np.float64)


def check_steering_vectors_file(path: str | Path, *, method: Method, in_time: InTime, balance: Balance,
                                classes: dict, blocks: list[int], steps: int, network_name: str,
                                network: dict) -> list[str]:
    """Stop if the vectors in `path` do not fit these settings; return what the file does not record.

    classes: the classes section of the config (min_mass, windows). The file must hold the same class names, and the
    same windows if it records them. blocks: each must be one of the file's blocks. steps, and the width of the
    networks.yaml entry `network` named `network_name`: from the shape of the vectors, and the recorded steps if any.
    hf_repo, revision, and file of `network` must equal the recorded ones if the file records a network. method,
    in_time, and balance must be in the metadata and equal. A mismatch raises ValueError that names the key.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    with np.load(path) as data:
        missing = [key for key in ("steering vectors", "classes", "blocks", "metadata") if key not in data.files]
        if missing:
            raise ValueError(f"{path} has no {missing}; it is not a file of `steer steering-vectors`")
        vectors = data["steering vectors"]
        names = [str(name) for name in data["classes"]]
        file_blocks = [int(b) for b in data["blocks"]]
        metadata = json.loads(str(data["metadata"]))

    for key, value in (("method", method), ("in_time", in_time), ("balance", balance)):
        if key not in metadata:
            raise ValueError(f"{path} records no {key}; cannot check it against steering_vectors.{key}")
        if metadata[key] != value:
            raise ValueError(f"steering_vectors.{key}: the config has {value!r}, {path} was made with "
                             f"{metadata[key]!r}")
    expected_names = [*classes["windows"], "rest"]
    if sorted(names) != sorted(expected_names):
        raise ValueError(f"classes: the config has the classes {expected_names}, {path} has {names}")
    for block in blocks:
        if block not in file_blocks:
            raise ValueError(f"block: block {block} is not one of the blocks {file_blocks} of {path}")
    width = network["width"]
    if vectors.ndim != 5 or vectors.shape[:2] != (len(names), len(file_blocks)) or vectors.shape[3] != 2:
        raise ValueError(f"{path}: the vectors have shape {vectors.shape}, expected ({len(names)} classes, "
                         f"{len(file_blocks)} blocks, steps, 2, width)")
    if vectors.shape[2] != steps or metadata.get("steps", steps) != steps:
        raise ValueError(f"sampler.steps: the config has {steps} Euler steps, {path} has vectors for "
                         f"{vectors.shape[2]} (recorded steps {metadata.get('steps', 'none')})")
    if vectors.shape[4] != width:
        raise ValueError(f"networks.{network_name}.width: the network has width {width}, the vectors in {path} have "
                         f"{vectors.shape[4]}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{path}: the vectors have a non-finite entry")

    unrecorded = []
    if "classes" in metadata:
        if metadata["classes"] != classes:
            raise ValueError(f"classes: the config has {classes}, {path} was made with {metadata['classes']}")
    else:
        unrecorded.append("the class windows (classes.min_mass, classes.windows)")
    if "network" in metadata:
        for key in NETWORK_KEYS:
            if metadata["network"].get(key) != network[key]:
                raise ValueError(f"networks.{network_name}.{key}: the config has {network[key]!r}, {path} was made "
                                 f"with {metadata['network'].get(key)!r}")
    else:
        unrecorded.append(f"the network ({', '.join(NETWORK_KEYS)})")
    return unrecorded
