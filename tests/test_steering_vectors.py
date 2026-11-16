import json

import numpy as np
import pytest
import scipy.linalg
import torch

from shellsteer.steering_vectors import (load_steering_vector, others_weights, save_steering_vectors, steering_vector,
                                         steering_vectors)

CLASSES = ["Z", "Jpsi", "rest"]


def toy_moments(count=(40, 300, 900), blocks=(1, 3), steps=5, width=6, seed=0):
    """The sums of gradients.py from random gradients: every class has its own mean and spread at every step."""
    rng = np.random.default_rng(seed)
    moments = {key: [] for key in ("unit second moment", "unit sum", "raw sum", "step-summed unit second moment",
                                   "step-summed unit sum", "step-summed raw sum")}
    for n in count:
        per_block = {key: [] for key in moments}
        for _ in blocks:
            center = rng.normal(size=(steps, 1, width))
            scale = rng.uniform(0.2, 2.0, size=(steps, 1, width))
            g = center + scale * rng.normal(size=(steps, n, width))        # (steps, pairs, width)
            u = g / np.linalg.norm(g, axis=-1, keepdims=True)
            summed = g.sum(0)
            su = summed / np.linalg.norm(summed, axis=-1, keepdims=True)
            values = {"unit second moment": np.einsum("kpv,kpw->kvw", u, u), "unit sum": u.sum(1), "raw sum": g.sum(1),
                      "step-summed unit second moment": su.T @ su, "step-summed unit sum": su.sum(0),
                      "step-summed raw sum": summed.sum(0)}
            for key, value in values.items():
                per_block[key].append(value)
        for key in moments:
            moments[key].append(np.stack(per_block[key]))
    moments = {key: np.stack(value) for key, value in moments.items()}
    moments.update(classes=np.array(CLASSES), blocks=np.array(blocks), count=np.array(count))
    return moments


def diagonal_moments(target_diagonal, others_diagonal, raw_mean, count=(10, 10, 10)):
    """One block, one step, width 4: second moments diagonal, per class mean second moment as given."""
    second = np.stack([np.diag(target_diagonal), np.diag(others_diagonal), np.diag(others_diagonal)])
    second = second * np.array(count)[:, None, None]
    raw_sum = np.stack([raw_mean, np.zeros(4), np.zeros(4)]) * np.array(count)[:, None]
    moments = {"unit second moment": second[:, None, None], "unit sum": np.zeros((3, 1, 1, 4)),
               "raw sum": raw_sum[:, None, None]}
    moments.update({f"step-summed {key}": value[:, :, 0] for key, value in moments.items()})
    moments.update(classes=np.array(CLASSES), blocks=np.array([0]), count=np.array(count))
    return moments


def test_fkt_finds_the_axis_where_only_the_target_is_large():
    moments = diagonal_moments([0.1, 0.1, 0.7, 0.1], [0.25, 0.25, 0.25, 0.25], raw_mean=[0.5, 0.0, -2.0, 0.0])
    xi = steering_vector(moments, "Z", 0, "fkt", "per_step", "equal")
    assert xi.shape == (1, 2, 2)
    np.testing.assert_allclose(xi.reshape(4), [0, 0, -1, 0], atol=1e-12)     # sign: raw mean . xi > 0


@pytest.mark.parametrize("balance", ["equal", "by_count"])
def test_fkt_solves_the_generalized_eigenproblem(balance):
    """xi is the top eigenvector of Sigma_C xi = lambda (Sigma_C + Sigma_C') xi, with the paper's Sigma_C'."""
    moments = toy_moments()
    xi = steering_vector(moments, "Jpsi", 3, "fkt", "per_step", balance)
    count = moments["count"]
    for k in range(5):
        second = moments["unit second moment"][:, 1, k]
        if balance == "equal":                   # paper Eq. 13: every class with weight one
            target, others = second[1] / count[1], second[0] / count[0] + second[2] / count[2]
        else:                                    # every pair with weight one
            target, others = second[1], second[0] + second[2]
        _, vectors = scipy.linalg.eigh(target, target + others)
        expected = vectors[:, -1] / np.linalg.norm(vectors[:, -1])
        assert abs(xi[k].reshape(-1) @ expected) == pytest.approx(1, abs=1e-10)


def test_lda_and_meandiff_formulas():
    moments = toy_moments()
    count, weights = moments["count"], np.array([0.5, 0.0, 0.5])
    second = moments["unit second moment"][:, 0, 2] / count[:, None, None]
    mean = moments["unit sum"][:, 0, 2] / count[:, None]
    mean_others, second_others = weights @ mean, np.einsum("c,cvw->vw", weights, second)
    covariance = second[1] - np.outer(mean[1], mean[1]) + second_others - np.outer(mean_others, mean_others)
    expected = {"lda": np.linalg.solve(covariance, mean[1] - mean_others), "meandiff": mean[1] - mean_others}
    for method, vector in expected.items():
        xi = steering_vector(moments, "Jpsi", 1, method, "per_step", "equal")[2].reshape(-1)
        assert abs(xi @ vector) / np.linalg.norm(vector) == pytest.approx(1, abs=1e-12), method


@pytest.mark.parametrize("method", ["fkt", "lda", "meandiff"])
def test_unit_length_and_sign_at_every_step(method):
    moments = toy_moments()
    xi = steering_vector(moments, "Z", 1, method, "per_step", "equal")
    assert xi.shape == (5, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(xi.reshape(5, -1), axis=-1), 1, atol=1e-12)
    assert (np.einsum("kw,kw->k", moments["raw sum"][0, 0], xi.reshape(5, -1)) > 0).all()


def test_constant_is_one_vector_at_every_step():
    moments = toy_moments()
    constant = steering_vector(moments, "Z", 3, "fkt", "constant", "equal")
    per_step = steering_vector(moments, "Z", 3, "fkt", "per_step", "equal")
    assert constant.shape == per_step.shape == (5, 2, 3)
    assert (constant == constant[0]).all() and not (per_step == per_step[0]).all()
    assert moments["step-summed raw sum"][0, 1] @ constant[0].reshape(-1) > 0


def test_by_count_differs_from_equal_only_when_counts_differ():
    for method in ("fkt", "lda", "meandiff"):
        unequal = toy_moments(count=(40, 300, 900))
        a, b = (steering_vector(unequal, "Z", 1, method, "per_step", balance) for balance in ("equal", "by_count"))
        assert np.abs(np.einsum("kvw,kvw->k", a, b)).min() < 0.999, method
        same = toy_moments(count=(200, 200, 200))
        a, b = (steering_vector(same, "Z", 1, method, "per_step", balance) for balance in ("equal", "by_count"))
        np.testing.assert_allclose(a, b, atol=1e-12, err_msg=method)


def test_others_weights():
    count = np.array([40, 300, 900])
    np.testing.assert_allclose(others_weights(count, 0, "equal"), [0, 0.5, 0.5])
    np.testing.assert_allclose(others_weights(count, 0, "by_count"), [0, 0.25, 0.75])


def test_steering_vectors_of_every_class_and_block():
    moments = toy_moments()
    vectors = steering_vectors(moments, "meandiff", "per_step", "equal")
    assert vectors.shape == (3, 2, 5, 2, 3)
    assert np.array_equal(vectors[2, 1], steering_vector(moments, "rest", 3, "meandiff", "per_step", "equal"))


def test_save_and_load(tmp_path):
    moments = toy_moments()
    vectors = steering_vectors(moments, "fkt", "per_step", "equal")
    metadata = {"method": "fkt", "in_time": "per_step", "balance": "equal"}
    path = tmp_path / "steering_vectors.npz"
    save_steering_vectors(path, vectors, moments["classes"], moments["blocks"], metadata)
    xi = load_steering_vector(path, "Jpsi", 3)
    assert xi.dtype == torch.float32 and xi.shape == (5, 2, 3)
    assert torch.equal(xi, torch.from_numpy(vectors[1, 1].astype(np.float32)))
    with np.load(path) as data:
        assert json.loads(str(data["metadata"])) == metadata
    with pytest.raises(ValueError, match="has no class 'Upsilon', only \\['Z', 'Jpsi', 'rest'\\]"):
        load_steering_vector(path, "Upsilon", 3)
    with pytest.raises(ValueError, match="has no block 2, only \\[1, 3\\]"):
        load_steering_vector(path, "Z", 2)
    with pytest.raises(ValueError, match="expected \\(3 classes, 2 blocks"):
        save_steering_vectors(path, vectors[:2], moments["classes"], moments["blocks"], metadata)


def test_bad_inputs_stop():
    moments = toy_moments()
    with pytest.raises(ValueError, match="target: 'W' is not one of the classes"):
        steering_vector(moments, "W", 1, "fkt", "per_step", "equal")
    with pytest.raises(ValueError, match="block: 2 is not one of the blocks \\[1, 3\\]"):
        steering_vector(moments, "Z", 2, "fkt", "per_step", "equal")
    with pytest.raises(ValueError, match="count: no pairs in class 'Jpsi'"):
        steering_vector({**moments, "count": np.array([40, 0, 900])}, "Z", 1, "fkt", "per_step", "equal")
    with pytest.raises(ValueError, match="method: 'pca' is not"):
        steering_vector(moments, "Z", 1, "pca", "per_step", "equal")


@pytest.mark.parametrize("method, matrix", [("fkt", "Sigma_C \\+ Sigma_C'"), ("lda", "K_C \\+ K_C'")])
def test_singular_moments_stop(method, matrix):
    """No class has a gradient along the last axis, so no matrix has a component there."""
    moments = diagonal_moments([0.2, 0.3, 0.5, 0.0], [0.3, 0.3, 0.4, 0.0], raw_mean=[1.0, 0, 0, 0])
    moments["unit sum"][:, 0, 0] = [[0.1, 0.2, 0.1, 0.0], [0.0, 0.1, 0.0, 0.0], [0.2, 0.0, 0.1, 0.0]]
    with pytest.raises(ValueError, match=f"{matrix} \\(class Z, block 0, step 0\\) is not positive definite"):
        steering_vector(moments, "Z", 0, method, "per_step", "equal")


def test_raw_mean_orthogonal_to_xi_stops():
    moments = diagonal_moments([0.1, 0.1, 0.7, 0.1], [0.25, 0.25, 0.25, 0.25], raw_mean=[1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="orthogonal to xi at step 0"):
        steering_vector(moments, "Z", 0, "fkt", "per_step", "equal")
