"""Tests on a CUDA GPU. They skip on a machine without one; select them with `pytest -m gpu`.

Fixture A_batch0_seed2040_steps200.npz: the first 8192 events of the paper's directions sample, their pair slots,
and the paper's baseline mass of each pair (batch 0, seed 2040, 200 steps, float32, no TF32, no torch.compile).
"""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from shellsteer.config import Classes, read_networks
from shellsteer.events import BATCH_KEYS, Events, class_of, edited_tokens, log_pair_mass, pair_mass, pair_slots, select
from shellsteer.gradients import gradient_moments
from shellsteer.hooks import Probe, Steer, attach, on_step
from shellsteer.networks import load_network
from shellsteer.sampler import batch_seed, generate, prior
from shellsteer.steered_generation import generate_pairs

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

DATA = Path(__file__).parent / "data"
CONFIGS = Path(__file__).parent.parent / "configs"
BATCH0 = DATA / "A_batch0_seed2040_steps200.npz"
SEED, STEPS, BLOCKS = 2040, 200, [0, 1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(CONFIGS / "networks.yaml")["shellflow"], "cuda")


@pytest.fixture(scope="module")
def stored():
    with np.load(BATCH0) as data:
        return {key: data[key] for key in data.files}


def make_events(stored, n):
    batch = {key: torch.from_numpy(stored[key][:n]) for key in BATCH_KEYS}
    batch["particle_type"] = batch["particle_type"].long()       # stored as int16
    batch["particle_charge"] = batch["particle_charge"].long()
    minus, plus = pair_slots(batch, "muon")
    return Events(batch=batch, minus=minus, plus=plus, event=stored["row"][:n], priority=np.arange(n))


def test_baseline_mass_equals_the_paper(network, stored):
    """Batch 0 of the directions sample gives the paper's baseline masses bit for bit.

    The paper's baseline ran on 8x A100 (2026-09-29) in float32, without TF32 and without torch.compile. A MIG
    slice or another GPU type (B200 included) draws other noise from the same seed or rounds the matmuls
    differently, so there the masses differ; the message then says how many and by how much.
    """
    events = make_events(stored, 8192)
    assert np.array_equal(events.minus.numpy(), stored["index_negative"])
    assert np.array_equal(events.plus.numpy(), stored["index_positive"])
    mass = generate_pairs(network, events, None, steps=STEPS, batch_size=8192, seed=SEED, device="cuda").mass
    mass = mass.astype(np.float32)
    paper = stored["mass"]
    differ = mass != paper
    largest = float(np.max(np.abs(mass - paper) / paper))
    assert not differ.any(), (f"{int(differ.sum())} of {len(paper)} masses differ from the paper's, "
                              f"largest relative difference {largest:.3g} ({torch.cuda.get_device_name()})")


def per_pair_gradients(network, events, steps, batch_size, seed):
    """g_p(L, k) of every pair, (blocks, steps, pairs, 2 x width), and the mass of the pass, on the GPU."""
    grads, masses = [], []
    for start in range(0, len(events), batch_size):
        part = slice(start, start + batch_size)
        batch = {key: value[part].cuda() for key, value in events.batch.items()}
        minus, plus = events.minus[part].cuda(), events.plus[part].cuda()
        probes = [Probe(network, steps, minus, plus) for _ in BLOCKS]
        with ExitStack() as stack:
            for block, probe in zip(BLOCKS, probes):
                stack.enter_context(attach(network, block, probe))
            p4 = generate(network, batch, prior(network, batch, batch_seed(seed, start // batch_size)), steps,
                          grad=True, on_step=on_step(*probes))
            g = torch.autograd.grad(log_pair_mass(p4, minus, plus).sum(), [p.delta for p in probes])
        grads.append(torch.stack(g).flatten(-2).double())
        masses.append(pair_mass(p4.detach(), minus, plus).double().cpu().numpy())
    return torch.cat(grads, dim=2), np.concatenate(masses)


def test_moments_equal_the_sums_of_the_per_pair_gradients(network, stored):
    """512 pairs in two batches, all six blocks, 200 steps, the paper's classes."""
    events, batch_size, keep = make_events(stored, 512), 256, 400
    classes = Classes.model_validate(yaml.safe_load((CONFIGS / "classes.yaml").read_text()))
    mass = generate_pairs(network, events, None, steps=STEPS, batch_size=batch_size, seed=SEED, device="cuda").mass
    kept = select(mass, events.priority, classes, keep)
    moments = gradient_moments(network, events, kept, classes, blocks=BLOCKS, steps=STEPS, batch_size=batch_size,
                               seed=SEED, device="cuda")

    g, pass_mass = per_pair_gradients(network, events, STEPS, batch_size, SEED)
    assert int(moments["non-finite pairs"]) == 0
    baseline_counts = [int((kept == name).sum()) for name in moments["classes"]]
    assert moments["count"].tolist() == baseline_counts, "a kept pair has another class in the gradient pass"
    pair_class = class_of(pass_mass, classes)                # the class as gradient_moments sees it
    for c, name in enumerate(moments["classes"]):
        members = np.flatnonzero((kept != "") & (pair_class == name))
        assert moments["count"][c] == len(members), name
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
            np.testing.assert_allclose(moments[key][c], value.cpu().numpy(), rtol=1e-9, atol=1e-12,
                                       err_msg=f"{name}: {key}")


def test_steering_at_alpha_zero_is_the_baseline_and_runs_at_alpha_one(network, stored):
    """Z vector of the paper at block 1 on 1024 pairs: alpha = 0 reproduces the unedited run bit for bit."""
    events = make_events(stored, 1024)
    batch = {key: value.cuda() for key, value in events.batch.items()}
    minus, plus = events.minus.cuda(), events.plus.cuda()
    xi = torch.from_numpy(np.load(DATA / "xi_z_block1.npy")).cuda()
    state = prior(network, batch, batch_seed(SEED, 0))
    baseline = generate(network, batch, state, STEPS)

    steered = {}
    for alpha in (0.0, 1.0):
        steer = Steer(xi, alpha, *edited_tokens(batch, minus, plus, "muon", "pair"))
        with attach(network, 1, steer):
            steered[alpha] = generate(network, batch, state, STEPS, on_step=steer.set_step)
    assert torch.equal(steered[0.0], baseline)
    assert steered[1.0].isfinite().all()
    shift = pair_mass(steered[1.0], minus, plus).log() - pair_mass(baseline, minus, plus).log()
    assert (shift != 0).float().mean() > 0.9, "alpha = 1 leaves most masses unchanged"
