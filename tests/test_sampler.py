from pathlib import Path

import numpy as np
import pytest
import torch

from shellsteer.config import read_networks
from shellsteer.networks import load_network
from shellsteer.sampler import generate, prior

DATA = Path(__file__).parent / "data"
NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(NETWORKS)["shellflow"], "cpu")


@pytest.fixture(scope="module")
def batch():
    """32 validation events of the paper's subset B: the conditioning only, no kinematics."""
    d = np.load(DATA / "batch32.npz")
    return {k: torch.from_numpy(d[k]).long() if k in ("particle_type", "particle_charge") else torch.from_numpy(d[k])
            for k in d.files}


def test_prior_depends_only_on_seed(network, batch):
    assert torch.equal(prior(network, batch, 7), prior(network, batch, 7))
    assert not torch.equal(prior(network, batch, 7), prior(network, batch, 8))


def test_matches_the_paper_sampler(network, batch):
    """Same events, seed, and steps as the sampler of the paper's runs (hep_mi.sampler.euler_generate) on CPU."""
    expected = np.load(DATA / "batch32_seed2041_steps200.npy")
    generated = generate(network, batch, prior(network, batch, 2041), 200)
    assert np.array_equal(generated.numpy(), expected)


def test_gradient_mode_gives_the_same_events_and_finite_gradients(network, batch):
    state = prior(network, batch, 2041)
    plain = generate(network, batch, state, 10)
    start = state.clone().requires_grad_(True)
    with_grad = generate(network, batch, start, 10, grad=True)
    assert torch.allclose(plain, with_grad, rtol=1e-5, atol=1e-5)
    with_grad[..., 3].sum().backward()
    assert start.grad is not None and torch.isfinite(start.grad).all() and start.grad.abs().sum() > 0
    steps_seen = []
    generate(network, batch, state, 3, on_step=steps_seen.append)
    assert steps_seen == [0, 1, 2]
