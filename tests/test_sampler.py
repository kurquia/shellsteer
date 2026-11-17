from pathlib import Path

import numpy as np
import pytest
import torch

import shellsteer.sampler as sampler_module
from shellsteer.config import read_networks
from shellsteer.hooks import attach
from shellsteer.networks import load_network
from shellsteer.sampler import NanCounts, NanError, decode, generate, prior

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
    assert np.array_equal(generated.four_momenta.numpy(), expected)
    assert generated.nan_state.tolist() == [0] * 32 and generated.nan_four_momenta.tolist() == [0] * 32


def test_gradient_mode_gives_the_same_events_and_finite_gradients(network, batch):
    state = prior(network, batch, 2041)
    plain = generate(network, batch, state, 10).four_momenta
    start = state.clone().requires_grad_(True)
    with_grad = generate(network, batch, start, 10, grad=True).four_momenta
    assert torch.allclose(plain, with_grad, rtol=1e-5, atol=1e-5)
    with_grad[..., 3].sum().backward()
    assert start.grad is not None and torch.isfinite(start.grad).all() and start.grad.abs().sum() > 0
    steps_seen = []
    generate(network, batch, state, 3, on_step=steps_seen.append)
    assert steps_seen == [0, 1, 2]


def nan_in_event(event: int):
    """A hook on the output of a block that writes NaN into the residual stream of one event, at every step."""
    def hook(h):
        h = h.clone()
        h[event] = float("nan")
        return h
    return hook


def test_nan_in_the_state_is_counted_and_set_to_zero(network, batch):
    state = prior(network, batch, 2041)
    clean = generate(network, batch, state, 4)
    with attach(network, 1, nan_in_event(3)):
        dirty = generate(network, batch, state, 4, on_nan="set_to_zero")
    slots, others = batch["particle_type"].shape[1], torch.arange(32) != 3
    assert dirty.nan_state[3] > 0 and dirty.nan_state[others].sum() == 0
    assert dirty.nan_four_momenta.sum() == 0
    assert torch.equal(dirty.four_momenta[others], clean.four_momenta[others])
    zero = decode(network.model, torch.zeros(1, slots, 5), batch["particle_type"][3:4], batch["particle_mask"][3:4])
    assert torch.equal(dirty.four_momenta[3:4], zero)


def test_nan_in_the_state_stops_with_on_nan_stop(network, batch):
    state = prior(network, batch, 2041)
    with attach(network, 1, nan_in_event(3)):
        count = int(generate(network, batch, state, 4).nan_state.sum())
        with pytest.raises(NanError, match=f"batch 7 of seed 2041: {count} NaN coordinates in the final state of "
                                           f"1 of 32 events \\(sampler.on_nan is stop\\)"):
            generate(network, batch, state, 4, on_nan="stop", label="batch 7 of seed 2041")
    assert generate(network, batch, state, 4, on_nan="stop").nan_state.sum() == 0


def test_an_infinite_direction_decodes_to_nan(network, batch):
    """The decode clamps energies and masses; only an infinite direction coordinate gives NaN four-momenta."""
    state = torch.zeros(1, batch["particle_type"].shape[1], 5)
    assert batch["particle_mask"][0, 2] > 0.5                                 # slot 2 holds a muon of event 0
    state[0, 2, 0], state[0, 2, 4] = float("inf"), float("-inf")
    assert decode(network.model, state, batch["particle_type"][:1], batch["particle_mask"][:1]).isfinite().all()
    state[0, 2, 1] = float("inf")
    four = decode(network.model, state, batch["particle_type"][:1], batch["particle_mask"][:1])
    assert four[0, 2].isnan().any() and four[0, 3:].isfinite().all() and four[0, :2].isfinite().all()


def test_nan_in_the_four_momenta_is_counted_and_stops(network, batch, monkeypatch):
    decode_without_nan = sampler_module.decode

    def decode_with_nan(*args, **kwargs):
        four = decode_without_nan(*args, **kwargs)
        four[5, 0, 2] = float("nan")
        return four
    monkeypatch.setattr(sampler_module, "decode", decode_with_nan)
    state = prior(network, batch, 2041)
    generated = generate(network, batch, state, 2)
    assert generated.nan_four_momenta.tolist() == [0] * 5 + [1] + [0] * 26 and generated.nan_state.sum() == 0
    with pytest.raises(NanError, match="the batch: 1 NaN entries in the four-momenta of 1 of 32 events"):
        generate(network, batch, state, 2, on_nan="stop")


def test_unknown_on_nan_stops(network, batch):
    with pytest.raises(ValueError, match="on_nan: 'zero' is not one of 'set_to_zero', 'stop'"):
        generate(network, batch, prior(network, batch, 2041), 2, on_nan="zero")


def test_nan_counts_metadata_and_log_line():
    counts = np.array([[0, 0, 0], [10, 4, 1], [0, 0, 0], [5, 0, 0]])
    nans = NanCounts("set_to_zero", np.array([100, 101, 102, 103]), counts)
    assert nans.metadata() == {
        "sampler.on_nan": "set_to_zero", "events": 4, "NaN coordinates of the final state": 15,
        "events with NaN in the final state": [101, 103], "NaN entries of the four-momenta": 4,
        "events with NaN in the four-momenta": [101], "NaN pair masses": 1, "events with NaN pair mass": [101]}
    assert nans.text() == ("NaN: 15 final-state coordinates in 2 events (set to 0), 4 four-momentum entries in 1 "
                           "events, 1 pair masses")
