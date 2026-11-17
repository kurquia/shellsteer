from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from shellsteer.config import read_networks
from shellsteer.events import BATCH_KEYS, NEGATIVE, POSITIVE, Events, edited_tokens, pair_mass, pair_slots
from shellsteer.hooks import Steer, attach
from shellsteer.networks import load_network
from shellsteer.sampler import NanError, batch_seed, generate, prior
from shellsteer.steered_generation import (Edit, GeneratedPairs, generate_pairs, generated_path, kept_path,
                                           load_generated, load_kept, save_generated, save_kept)

DATA = Path(__file__).parent / "data"
NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"
STEPS, BATCH_SIZE, SEED, BLOCK = 4, 12, 2041, 1      # 32 events in batches of 12, 12, and 8
THIRD_MUON = {1: NEGATIVE, 2: POSITIVE}              # event -> charge of the extra muon in slot 4

# A random steering vector, unit length at every step over both tokens.
XI = torch.randn(STEPS, 2, 128, generator=torch.Generator().manual_seed(0))
XI /= XI.flatten(1).norm(dim=1)[:, None, None]

METADATA = {"alpha": -1.5, "block": 1, "target": "Z", "method": "fkt", "seed": 2041, "steps": 200,
            "edited steps": None, "tokens": "pair", "norm_restore": False,
            "steering vector file": "steering_vectors.npz"}


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(NETWORKS)["shellflow"], "cpu")


@pytest.fixture(scope="module")
def events():
    """batch32, with the particle in slot 4 of events 1 and 2 made a muon, so all_of_type edits three tokens there."""
    with np.load(DATA / "batch32.npz") as data:
        batch = {key: torch.from_numpy(data[key]) for key in BATCH_KEYS}
    batch["particle_type"] = batch["particle_type"].long()
    batch["particle_charge"] = batch["particle_charge"].long()
    minus, plus = pair_slots(batch, "muon")
    for event, charge in THIRD_MUON.items():
        assert batch["particle_mask"][event, 4] == 1 and batch["particle_type"][event, 4] != 13
        batch["particle_type"][event, 4], batch["particle_charge"][event, 4] = 13, charge
    return Events(batch=batch, minus=minus, plus=plus, event=np.arange(100, 132), priority=np.arange(32))


def edit(alpha: float, tokens: str = "pair") -> Edit:
    return Edit(xi=XI, alpha=alpha, block=BLOCK, tokens=tokens, pair="muon", steps=None, norm_restore=False)


def run(network, events, edit):
    return generate_pairs(network, events, edit, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu")


def by_hand(network, events, alpha, tokens):
    """Mass and pair four-momenta from a loop over the batches with Steer, attach, and batch_seed."""
    masses, four_momenta = [], []
    for index, start in enumerate(range(0, len(events), BATCH_SIZE)):
        part = slice(start, start + BATCH_SIZE)
        batch = {key: value[part] for key, value in events.batch.items()}
        minus, plus = events.minus[part], events.plus[part]
        steer = Steer(XI, alpha, *edited_tokens(batch, minus, plus, "muon", tokens))
        with attach(network, BLOCK, steer):
            p4 = generate(network, batch, prior(network, batch, batch_seed(SEED, index)), STEPS,
                          on_step=steer.set_step).four_momenta
        rows = torch.arange(len(minus))
        masses.append(pair_mass(p4, minus, plus).double().numpy())
        four_momenta.append(torch.stack([p4[rows, minus], p4[rows, plus]], dim=1).numpy())
    return np.concatenate(masses), np.concatenate(four_momenta)


def test_alpha_zero_is_the_baseline(network, events):
    baseline, steered = run(network, events, None), run(network, events, edit(0.0))
    assert np.array_equal(steered.mass, baseline.mass)
    assert np.array_equal(steered.four_momenta, baseline.four_momenta)


@pytest.mark.parametrize("tokens", ["pair", "all_of_type"])
def test_equals_a_loop_over_batches(network, events, tokens):
    generated = run(network, events, edit(2.0, tokens))
    mass, four_momenta = by_hand(network, events, 2.0, tokens)
    assert np.array_equal(generated.mass, mass)
    assert np.array_equal(generated.four_momenta, four_momenta)
    assert np.array_equal(generated.event, events.event)


def test_the_edit_moves_the_pairs(network, events):
    """Guards the two tests above against an edit that does nothing or tokens that make no difference."""
    baseline, pair = run(network, events, None), run(network, events, edit(2.0, "pair"))
    all_of_type = run(network, events, edit(2.0, "all_of_type"))
    assert (pair.mass != baseline.mass).all()
    assert (all_of_type.mass != pair.mass)[list(THIRD_MUON)].all()


def test_mass_is_the_mass_of_the_four_momenta(network, events):
    generated = run(network, events, edit(1.0))
    assert generated.mass.shape == (32,) and generated.mass.dtype == np.float64
    assert generated.four_momenta.shape == (32, 2, 4) and generated.four_momenta.dtype == np.float32
    negative, positive = torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)
    mass = pair_mass(torch.from_numpy(generated.four_momenta), negative, positive).double().numpy()
    assert np.array_equal(mass, generated.mass)


@pytest.mark.parametrize("change, message", [
    (dict(xi=torch.zeros(STEPS + 1, 2, 128)), "xi has 5 steps, the sampler runs 4"),
    (dict(xi=torch.zeros(STEPS, 2, 224)), "xi has width 224, the network has width 128"),
    (dict(xi=torch.zeros(STEPS, 256)), r"xi has shape \(4, 256\), expected \(steps, 2, width\)"),
    (dict(steps=(2, 4)), r"edit steps: need 0 <= first <= last < 4, got \(2, 4\)"),
])
def test_a_wrong_edit_stops(network, events, change, message):
    with pytest.raises(ValueError, match=message):
        run(network, events, replace(edit(1.0), **change))


def test_file_names():
    assert generated_path("run", 2041, 0.0) == Path("run/generated/seed_2041/alpha_+0.npz")
    assert generated_path("run", 2041, -0.0).name == "alpha_+0.npz"
    assert generated_path("run", 2042, 2).name == "alpha_+2.npz"
    assert generated_path("run", 2042, -1.5).name == "alpha_-1.5.npz"
    assert kept_path("run") == Path("run/generated/kept.npz")
    with pytest.raises(ValueError, match="alpha: 1.0000001 is written as 1 in the file name"):
        generated_path("run", 2041, 1.0000001)


def test_save_and_load(tmp_path):
    generated = GeneratedPairs(mass=np.array([91.2, 3.1]),
                               four_momenta=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
                               event=np.array([117, 647]))
    path = generated_path(tmp_path, 2041, -1.5)
    save_generated(path, generated, METADATA | {"network revision": "d1b047a"})
    loaded, metadata = load_generated(path)
    assert metadata == METADATA | {"network revision": "d1b047a"}
    for key in ("mass", "four_momenta", "event"):
        assert np.array_equal(getattr(loaded, key), getattr(generated, key))
        assert getattr(loaded, key).dtype == getattr(generated, key).dtype
    with pytest.raises(ValueError, match=r"metadata has no \['steering vector file'\]"):
        save_generated(path, generated, {key: METADATA[key] for key in list(METADATA)[:-1]})


def test_save_and_load_kept(tmp_path):
    kept = np.array(["Z", "", "rest", "Jpsi"], dtype=object)
    save_kept(kept_path(tmp_path), kept, np.array([5, 6, 7, 8]), {"seed": 2041, "keep": 3})
    loaded, event, metadata = load_kept(kept_path(tmp_path))
    assert loaded.dtype == object and list(loaded) == ["Z", "", "rest", "Jpsi"]
    assert np.array_equal(event, [5, 6, 7, 8]) and metadata == {"seed": 2041, "keep": 3}
    with pytest.raises(ValueError, match="kept has 4 pairs, event has 3"):
        save_kept(kept_path(tmp_path), kept, np.array([5, 6, 7]), {})


def nan_in_row(row: int):
    """A hook on the output of a block that writes NaN into the residual stream of one row of every batch."""
    def hook(h):
        h = h.clone()
        h[row] = float("nan")
        return h
    return hook


def test_nan_counts_reach_the_metadata(network, events, tmp_path):
    clean = run(network, events, None)
    assert clean.nans.counts.shape == (32, 3) and clean.nans.counts.sum() == 0
    with attach(network, BLOCK, nan_in_row(2)):
        generated = run(network, events, None)
    nans = generated.nans.metadata()
    assert nans["sampler.on_nan"] == "set_to_zero" and nans["events"] == 32
    assert nans["events with NaN in the final state"] == [102, 114, 126]     # row 2 of the batches of 12, 12, and 8
    assert nans["NaN coordinates of the final state"] > 0 and nans["NaN entries of the four-momenta"] == 0
    others = ~np.isin(events.event, [102, 114, 126])
    assert np.array_equal(generated.mass[others], clean.mass[others])
    save_generated(tmp_path / "pairs.npz", generated, METADATA)
    assert load_generated(tmp_path / "pairs.npz")[1] == METADATA | {"NaN counts": nans}


def test_nan_stops_the_generation_with_on_nan_stop(network, events):
    with attach(network, BLOCK, nan_in_row(2)):
        with pytest.raises(NanError, match="batch 0 of seed 2041: .* NaN coordinates in the final state of 1 of 12 "
                                           "events"):
            generate_pairs(network, events, edit(1.0), steps=STEPS, batch_size=BATCH_SIZE, seed=SEED, device="cpu",
                           on_nan="stop")
