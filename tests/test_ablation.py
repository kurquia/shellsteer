"""The mean ablation of paper Sec. 5.3: the Ablate and Record hooks, the activation means, and generate_pairs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from shellsteer.activation_means import (activation_means, load_activation_means, mean_projection,
                                         save_activation_means)
from shellsteer.config import Classes, read_networks
from shellsteer.events import BATCH_KEYS, Events, class_of, pair_mass, pair_slots
from shellsteer.hooks import Ablate, Record, attach, on_step
from shellsteer.networks import load_network
from shellsteer.sampler import batch_seed, generate, prior
from shellsteer.steered_generation import (ABLATION_METADATA_KEYS, Ablation, GeneratedPairs, ablated_path,
                                           generate_pairs, load_generated, save_generated)

DATA = Path(__file__).parent / "data"
NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"
STEPS, BATCH_SIZE, SEED, BLOCK, WIDTH = 4, 12, 2040, 1, 128     # 32 events in batches of 12, 12, and 8
CLASSES = Classes(min_mass=0.3, windows={"Z": [81.19, 101.19], "Jpsi": [2.9, 3.3], "Upsilon": [9.2, 10.6]})


def unit_vector(steps: int, seed: int) -> torch.Tensor:
    xi = torch.randn(steps, 2, WIDTH, generator=torch.Generator().manual_seed(seed))
    return xi / xi.flatten(1).norm(dim=1)[:, None, None]


XI = unit_vector(STEPS, 0)
UNIT = XI.flatten(1).double() / XI.flatten(1).double().norm(dim=1, keepdim=True)   # unit in float64, as Ablate uses
MEAN = torch.tensor([0.3, -0.2, 0.1, 0.5], dtype=torch.float64)


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
    return Events(batch=batch, minus=minus, plus=plus, event=np.arange(100, 132), priority=np.arange(32))


def random_stream(b: int = 6, n: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.randn(b, n, WIDTH, generator=torch.Generator().manual_seed(1))
    minus = torch.tensor([0, 1, 2, 0, 3, 4])[:b]
    plus = torch.tensor([1, 0, 4, 3, 2, 1])[:b]
    return h, minus, plus


def test_xi_orthogonal_to_h_changes_nothing():
    """With xi on entries where the pair's tokens are zero and c_k = 0, the projection is exactly zero already."""
    h, minus, plus = random_stream()
    h[..., :64] = 0.0
    xi = torch.zeros(STEPS, 2, WIDTH)
    xi[:, :, :64] = unit_vector(STEPS, 2)[:, :, :64]
    xi /= xi.flatten(1).norm(dim=1)[:, None, None]
    ablate = Ablate(xi, torch.zeros(STEPS), minus, plus)
    for k in range(STEPS):
        ablate.set_step(k)
        assert torch.equal(ablate(h), h)


def test_ablation_sets_the_projection_to_c_k():
    h, minus, plus = random_stream()
    rows = torch.arange(len(minus))
    ablate = Ablate(XI, MEAN, minus, plus)
    for k in range(STEPS):
        ablate.set_step(k)
        out = ablate(h)
        unit = XI[k].flatten().double()
        before = torch.cat((h[rows, minus], h[rows, plus]), -1).double()
        after = torch.cat((out[rows, minus], out[rows, plus]), -1).double()
        assert torch.allclose(after @ unit, MEAN[k].expand(len(rows)), atol=1e-6)
        # The part orthogonal to xi stays.
        orthogonal = lambda x: x - (x @ unit)[:, None] * unit           # noqa: E731
        assert torch.allclose(orthogonal(after), orthogonal(before), atol=1e-6)
        # Every other token is unchanged, bit for bit.
        other = torch.ones(h.shape[:2], dtype=torch.bool)
        other[rows, minus], other[rows, plus] = False, False
        assert torch.equal(out[other], h[other])


def test_ablate_stops_on_bad_input():
    h, minus, plus = random_stream()
    with pytest.raises(ValueError, match="unit length"):
        Ablate(2 * XI, MEAN, minus, plus)
    with pytest.raises(ValueError, match=r"mean has shape \(3,\), expected \(4,\)"):
        Ablate(XI, MEAN[:3], minus, plus)
    with pytest.raises(ValueError, match=r"xi has shape \(4, 256\), expected \(steps, 2, width\)"):
        Ablate(XI.flatten(1), MEAN, minus, plus)
    narrow = unit_vector(STEPS, 0)[..., :64]
    ablate = Ablate(narrow / narrow.flatten(1).norm(dim=1)[:, None, None], MEAN, minus, plus)
    with pytest.raises(ValueError, match="xi has width 64, the residual stream of this block has width 128"):
        ablate(h)
    with pytest.raises(ValueError, match="step 4, but xi has only 4 steps"):
        ablate.set_step(4)


def test_ablation_in_the_network_holds_the_projection_at_c_k(network, events):
    """A Record after the Ablate at the same block sees the projection c_k on every pair at every step."""
    batch = {key: value[:BATCH_SIZE] for key, value in events.batch.items()}
    minus, plus = events.minus[:BATCH_SIZE], events.plus[:BATCH_SIZE]
    ablate, record = Ablate(XI, MEAN, minus, plus), Record(network, STEPS, minus, plus)
    with attach(network, BLOCK, ablate), attach(network, BLOCK, record):
        generate(network, batch, prior(network, batch, SEED), STEPS, on_step=on_step(ablate, record))
    projection = torch.einsum("kbw,kw->kb", record.tokens.double(), XI.flatten(1).double())
    assert torch.allclose(projection, MEAN[:, None].expand_as(projection), atol=1e-5)


def test_activation_means_match_a_hand_computation(network, events):
    baseline = generate_pairs(network, events, None, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu")
    kept = class_of(baseline.mass, CLASSES)
    kept[[0, 5, 17]] = ""                                       # pairs left out by events.select
    means = activation_means(network, events, kept, CLASSES, blocks=[0, BLOCK], steps=STEPS, batch_size=BATCH_SIZE,
                             seed=SEED, device="cpu")
    assert means["count"].sum() == (kept != "").sum() and means["non-finite pairs"] == 0
    assert means["sum"].shape == (4, 2, STEPS, 2 * WIDTH)

    tokens = []                                                 # h at block 1 on the pair's tokens, by hand
    for index, start in enumerate(range(0, len(events), BATCH_SIZE)):
        part = slice(start, start + BATCH_SIZE)
        batch = {key: value[part] for key, value in events.batch.items()}
        minus, plus, seen = events.minus[part], events.plus[part], {}
        rows = torch.arange(len(minus))

        def keep(h, seen=seen, rows=rows, minus=minus, plus=plus):
            seen[len(seen)] = torch.cat((h[rows, minus], h[rows, plus]), -1).double()
            return h
        with attach(network, BLOCK, keep):
            p4 = generate(network, batch, prior(network, batch, batch_seed(SEED, index)), STEPS)
        assert np.array_equal(pair_mass(p4, minus, plus).double().numpy(), baseline.mass[part])
        tokens.append(torch.stack([seen[k] for k in range(STEPS)], 1))
    tokens = torch.cat(tokens)[torch.from_numpy(kept != "")]       # (kept pairs, steps, 2 width)
    by_hand = torch.einsum("pkw,kw->k", tokens, UNIT) / len(tokens)
    assert np.allclose(mean_projection(means, XI, BLOCK), by_hand.numpy(), rtol=1e-10, atol=1e-12)


def test_mean_projection_by_hand():
    rng = np.random.default_rng(3)
    sums = rng.normal(size=(4, 2, STEPS, 2 * WIDTH))
    means = {"blocks": np.array([0, 1]), "count": np.array([5, 0, 2, 3]), "sum": sums}
    xi = 3.0 * XI.numpy()                                       # scaled to unit length inside
    unit = xi.reshape(STEPS, -1).astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    expected = [sums[:, 1, k].sum(0) @ unit[k] / 10 for k in range(STEPS)]
    assert np.allclose(mean_projection(means, xi, 1), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="block: 3 is not one of the blocks"):
        mean_projection(means, xi, 3)
    with pytest.raises(ValueError, match="xi has shape"):
        mean_projection(means, xi[:, :, :64], 1)


def test_save_and_load_activation_means(tmp_path):
    means = {"classes": np.array(["Z", "rest"]), "blocks": np.array([1]), "count": np.array([3, 4]),
             "non-finite pairs": np.array(0), "sum": np.ones((2, 1, STEPS, 2 * WIDTH))}
    save_activation_means(tmp_path / "activation_means.npz", means, {"seed": SEED})
    loaded, metadata = load_activation_means(tmp_path / "activation_means.npz")
    assert metadata == {"seed": SEED} and loaded.keys() == means.keys()
    assert all(np.array_equal(loaded[key], means[key]) for key in means)


def test_generate_pairs_ablation_equals_a_loop_over_batches(network, events):
    ablation = Ablation(xi=XI, mean=MEAN, block=BLOCK)
    generated = generate_pairs(network, events, ablation, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu")
    baseline = generate_pairs(network, events, None, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu")
    masses = []
    for index, start in enumerate(range(0, len(events), BATCH_SIZE)):
        part = slice(start, start + BATCH_SIZE)
        batch = {key: value[part] for key, value in events.batch.items()}
        minus, plus = events.minus[part], events.plus[part]
        ablate = Ablate(XI, MEAN, minus, plus)
        with attach(network, BLOCK, ablate):
            p4 = generate(network, batch, prior(network, batch, batch_seed(SEED, index)), STEPS,
                          on_step=ablate.set_step)
        masses.append(pair_mass(p4, minus, plus).double().numpy())
    assert np.array_equal(generated.mass, np.concatenate(masses))
    assert (generated.mass != baseline.mass).all()


@pytest.mark.parametrize("change, message", [
    (dict(mean=MEAN[:3]), r"mean has shape \(3,\), expected \(4,\)"),
    (dict(xi=unit_vector(STEPS + 1, 0)), "xi has 5 steps, the sampler runs 4"),
    (dict(xi=2 * XI), "unit length"),
])
def test_a_wrong_ablation_stops(network, events, change, message):
    ablation = replace(Ablation(xi=XI, mean=MEAN, block=BLOCK), **change)
    with pytest.raises(ValueError, match=message):
        generate_pairs(network, events, ablation, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu")


def test_ablation_files(tmp_path):
    assert ablated_path("run", 2040, 1) == Path("run/ablated/seed_2040/block_1.npz")
    assert ablated_path("run", 2040, None) == Path("run/ablated/seed_2040/baseline.npz")
    metadata = {"block": 1, "target": "Z", "method": "fkt", "seed": 2040, "steps": 200, "sample": "directions",
                "steering vector file": "steering_vectors.npz", "activation means file": "activation_means.npz"}
    generated = GeneratedPairs(mass=np.array([91.2]), four_momenta=np.zeros((1, 2, 4), np.float32),
                               event=np.array([7]))
    save_generated(ablated_path(tmp_path, 2040, 1), generated, metadata, ABLATION_METADATA_KEYS)
    assert load_generated(ablated_path(tmp_path, 2040, 1))[1] == metadata
    with pytest.raises(ValueError, match=r"metadata has no \['activation means file'\]"):
        save_generated(ablated_path(tmp_path, 2040, 1), generated,
                       {key: metadata[key] for key in list(metadata)[:-1]}, ABLATION_METADATA_KEYS)
