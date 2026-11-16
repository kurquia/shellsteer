"""Steering vectors xi from the per-class sums of the gradients (the output of gradients.py), paper Sec. 4.2.

A gradient G_p(t_k) of pair p at step k has 2 x width entries, negative token first; hat G_p is G_p / |G_p|. For a
target class C, the classes other than C form C'. Per class D the sums give the mean unit gradient m_D and the
second moment Sigma_D = (1 / n_D) sum over p in D of hat G_p hat G_p^T. The others enter as a mixture:

    m_C' = sum over D != C of w_D m_D,    Sigma_C' = sum over D != C of w_D Sigma_D,

with w_D = 1 / (number of other classes) for `balance: equal` and w_D = n_D / (pairs in C') for `balance: by_count`.

fkt:      xi maximises xi^T Sigma_C xi / xi^T (Sigma_C + Sigma_C') xi (paper Eq. 14). With the eigendecomposition
          Sigma_C + Sigma_C' = U sigma U^T and M_FK = U sigma^(-1/2), xi = M_FK xi~, with xi~ the top eigenvector
          of M_FK^T Sigma_C M_FK. The paper's Sigma_C' is the sum over D != C of Sigma_D, which is (number of other
          classes) times the mixture here; the scale of Sigma_C' does not change xi. by_count gives the same xi as
          sums over pairs, every pair with the same weight.
lda:      xi = (K_C + K_C')^(-1) (m_C - m_C'), with K = Sigma - m m^T the covariance of the unit gradients.
meandiff: xi = m_C - m_C'.

Every steering vector has unit length over both tokens. Its sign makes G_C(t_k)^T xi(t_k) > 0, with G_C the mean
of the raw gradients of C. `per_step` solves at every step; `constant` solves once with the step-summed gradient
G_p = sum over k of G_p(t_k) and uses that vector at every step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.linalg
import torch

Method = Literal["fkt", "lda", "meandiff"]
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


def lda(mean: np.ndarray, second: np.ndarray, mean_others: np.ndarray, second_others: np.ndarray,
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
    if method == "lda":
        return lda(means[target], seconds[target], mean_others, second_others, where)
    raise ValueError(f"method: {method!r} is not fkt, lda, or meandiff")


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
                          metadata: dict) -> None:
    """Write the vectors (classes, blocks, steps, 2, width), their classes and blocks, and the settings to one .npz."""
    if vectors.shape[:2] != (len(classes), len(blocks)):
        raise ValueError(f"vectors have shape {vectors.shape}, expected ({len(classes)} classes, {len(blocks)} "
                         f"blocks, steps, 2, width)")
    np.savez(path, **{"steering vectors": vectors, "classes": np.asarray(classes), "blocks": np.asarray(blocks),
                      "metadata": np.array(json.dumps(metadata, indent=1))})


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

