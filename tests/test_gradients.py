from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch

from shellsteer.config import Classes, read_networks
from shellsteer.events import BATCH_KEYS, Events, log_pair_mass, pair_slots, select
from shellsteer.gradients import baseline_mass, gradient_moments, load_moments, save_moments, unit
from shellsteer.hooks import Probe, attach, on_step
from shellsteer.networks import load_network
from shellsteer.sampler import batch_seed, generate, prior

DATA = Path(__file__).parent / "data"
NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(NETWORKS)["shellflow"], "cpu")


@pytest.fixture(scope="module")
def events():
    with np.load(DATA / "batch32.npz") as data:
        batch = {key: torch.from_numpy(data[key]) for key in BATCH_KEYS}
    batch["particle_type"] = batch["particle_type"].long()
    batch["particle_charge"] = batch["particle_charge"].long()
    minus, plus = pair_slots(batch, "muon")
    return Events(batch=batch, minus=minus, plus=plus, event=np.arange(32), priority=np.arange(32)[::-1].copy())


def per_pair_gradients(network, events, blocks, steps, batch_size, seed):
    """g_p(L, k) of every pair, one batch at a time, without the class sums: (blocks, steps, pairs, 2 x width)."""
    out = []
    for start in range(0, len(events), batch_size):
        part = slice(start, start + batch_size)
        batch = {key: value[part] for key, value in events.batch.items()}
        minus, plus = events.minus[part], events.plus[part]
        probes = [Probe(network, steps, minus, plus) for _ in blocks]
        with ExitStack() as stack:
            for block, probe in zip(blocks, probes):
                stack.enter_context(attach(network, block, probe))
            p4 = generate(network, batch, prior(network, batch, batch_seed(seed, start // batch_size)), steps,
                          grad=True, on_step=on_step(*probes))
            grads = torch.autograd.grad(log_pair_mass(p4, minus, plus).sum(), [p.delta for p in probes])
        out.append(torch.stack(grads).flatten(-2).double())
    return torch.cat(out, dim=2)


def test_moments_equal_the_sums_of_the_per_pair_gradients(network, events):
    """Two batches, two blocks, four steps; classes from wide windows so that every class has pairs."""
    steps, blocks, batch_size, seed = 4, [1, 3], 20, 7
    mass = baseline_mass(network, events, steps=steps, batch_size=batch_size, seed=seed, device="cpu")
    edges = np.quantile(mass, [0.2, 0.45, 0.55, 0.8])
    classes = Classes(min_mass=float(min(mass.min(), edges[0])) - 1e-3,
                      windows={"low": [float(edges[0]), float(edges[1])], "high": [float(edges[2]), float(edges[3])]})
    kept = select(mass, events.priority, classes, 28)
    moments = gradient_moments(network, events, kept, classes, blocks=blocks, steps=steps, batch_size=batch_size,
                               seed=seed, device="cpu")

    g = per_pair_gradients(network, events, blocks, steps, batch_size, seed)
    assert list(moments["classes"]) == ["low", "high", "rest"]
    for c, name in enumerate(moments["classes"]):
        members = np.flatnonzero(kept == name)
        assert moments["count"][c] == len(members) > 0
        x = g[:, :, members]
        u = x / x.norm(dim=-1, keepdim=True)
        summed = x.sum(1)
        su = summed / summed.norm(dim=-1, keepdim=True)
        expected = {
            "unit second moment": torch.einsum("lkpi,lkpj->lkij", u, u), "unit sum": u.sum(2), "raw sum": x.sum(2),
            "step-summed unit second moment": torch.einsum("lpi,lpj->lij", su, su),
            "step-summed unit sum": su.sum(1), "step-summed raw sum": summed.sum(1),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(moments[key][c], value.numpy(), rtol=1e-9, atol=1e-12, err_msg=key)
    assert moments["count"].sum() == 28 and int(moments["non-finite pairs"]) == 0


def test_unit_keeps_zero_at_zero():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    assert torch.equal(unit(x), torch.tensor([[0.6, 0.8], [0.0, 0.0]], dtype=torch.float64))


def test_save_and_load(tmp_path):
    moments = {"count": np.array([3, 4]), "raw sum": np.ones((2, 1, 5, 256))}
    save_moments(tmp_path / "m.npz", moments, {"seed": 2040, "network": "shellflow"})
    loaded, metadata = load_moments(tmp_path / "m.npz")
    assert metadata == {"seed": 2040, "network": "shellflow"}
    assert np.array_equal(loaded["raw sum"], moments["raw sum"]) and list(loaded) == ["count", "raw sum"]
