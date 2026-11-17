from pathlib import Path

import numpy as np
import pytest
import torch

from shellsteer.config import read_networks
from shellsteer.events import log_pair_mass
from shellsteer.hooks import Probe, Steer, attach, on_step
from shellsteer.networks import load_network
from shellsteer.sampler import generate, prior

DATA = Path(__file__).parent / "data"
NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"
SEED, STEPS = 2041, 200


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(NETWORKS)["shellflow"], "cpu")


@pytest.fixture(scope="module")
def batch():
    d = np.load(DATA / "batch32.npz")
    return {k: torch.from_numpy(d[k]).long() if k in ("particle_type", "particle_charge") else torch.from_numpy(d[k])
            for k in d.files}


@pytest.fixture(scope="module")
def slots():
    d = np.load(DATA / "batch32_muon_slots.npz")
    return torch.from_numpy(d["index_negative"]).long(), torch.from_numpy(d["index_positive"]).long()


def pair_masks(slots, n_tokens=10):
    minus, plus = slots
    rows = torch.arange(len(minus))
    m = torch.zeros(len(minus), n_tokens, dtype=torch.bool)
    p = torch.zeros(len(minus), n_tokens, dtype=torch.bool)
    m[rows, minus], p[rows, plus] = True, True
    return m, p


def log_mass(events, slots):
    return log_pair_mass(events, *slots)


def test_alpha_zero_is_the_baseline(network, batch, slots):
    xi = torch.from_numpy(np.load(DATA / "xi_z_block1.npy"))
    steer = Steer(xi, 0.0, *pair_masks(slots))
    with attach(network, 1, steer):
        steered = generate(network, batch, prior(network, batch, SEED), STEPS, on_step=steer.set_step).four_momenta
    assert np.array_equal(steered.numpy(), np.load(DATA / "batch32_seed2041_steps200.npy"))


def test_steering_matches_the_paper_code(network, batch, slots):
    """Z vector of the paper at block 1, alpha = 1; reference from the paper's edit (ResonanceShift) on CPU."""
    xi = torch.from_numpy(np.load(DATA / "xi_z_block1.npy"))
    steer = Steer(xi, 1.0, *pair_masks(slots))
    with attach(network, 1, steer):
        steered = generate(network, batch, prior(network, batch, SEED), STEPS, on_step=steer.set_step).four_momenta
    assert np.array_equal(steered.numpy(), np.load(DATA / "batch32_steer_z_block1_alpha1.npy"))


def test_probe_gradient_matches_the_paper_code(network, batch, slots):
    """Gradient of log m at block 1; reference from the paper's probe (z_only_read_gradients.Probe) on CPU."""
    probe = Probe(network, STEPS, *slots)
    with attach(network, 1, probe):
        events = generate(network, batch, prior(network, batch, SEED), STEPS, grad=True,
                          on_step=on_step(probe)).four_momenta
        gradient, = torch.autograd.grad(log_mass(events, slots).sum(), probe.delta)
    reference = np.load(DATA / "batch32_gradient_block1.npz")
    assert np.array_equal(gradient[reference["steps"]].numpy(), reference["gradient"])


def test_probe_gradient_matches_finite_differences(network, batch, slots):
    """The probe gradient along a random direction at one step equals the change of log m when steering along it."""
    steps, k = 10, 6
    state = prior(network, batch, SEED)
    probe = Probe(network, steps, *slots)
    with attach(network, 1, probe):
        events = generate(network, batch, state, steps, grad=True, on_step=on_step(probe)).four_momenta
        gradient, = torch.autograd.grad(log_mass(events, slots).sum(), probe.delta)
    direction = torch.zeros(steps, 2, 128)
    direction[k] = torch.randn(2, 128, generator=torch.Generator().manual_seed(0))
    direction[k] /= direction[k].norm()
    eps = 0.1   # smaller steps drown in float32 rounding along the trajectory
    change = []
    for alpha in (eps, -eps):
        steer = Steer(direction, alpha, *pair_masks(slots))
        with attach(network, 1, steer):
            change.append(log_mass(generate(network, batch, state, steps, on_step=steer.set_step).four_momenta, slots))
    finite_difference = (change[0] - change[1]) / (2 * eps)
    predicted = (gradient[k] * direction[k]).sum((-2, -1))
    # float32 rounding and second-order terms make a few pairs noisy; the typical pair agrees within 1%
    relative = (predicted - finite_difference).abs() / (finite_difference.abs() + 1e-3)
    assert relative.median() < 0.01
    assert torch.corrcoef(torch.stack([predicted, finite_difference]))[0, 1] > 0.99


def test_hook_is_removed_after_an_error(network):
    def failing(h):
        raise RuntimeError("inside the hook")
    with pytest.raises(RuntimeError):
        with attach(network, 1, failing):
            raise RuntimeError("inside the with block")
    assert len(network.model.blocks[1]._forward_hooks) == 0


def test_steer_refuses_overlapping_masks(slots):
    m, _ = pair_masks(slots)
    with pytest.raises(ValueError, match="both"):
        Steer(torch.zeros(STEPS, 2, 128), 1.0, m, m)


def test_wrong_width_stops_with_a_clear_error(network, batch, slots):
    steer = Steer(torch.zeros(STEPS, 2, 224), 1.0, *pair_masks(slots))
    with pytest.raises(ValueError, match="xi has width 224, the residual stream of this block has width 128"):
        with attach(network, 1, steer):
            generate(network, batch, prior(network, batch, SEED), 2, on_step=steer.set_step)
