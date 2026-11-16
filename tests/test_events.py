from pathlib import Path

import numpy as np
import pytest
import torch

import shellsteer.events as events_module
from shellsteer.config import Classes, DatasetEntry, DatasetFiles
from shellsteer.events import batches, class_of, edited_tokens, load_events, pair_mass, pair_slots, select
from shellsteer.sampler import BATCH_KEYS

DATA = Path(__file__).parent / "data"


def batch32() -> dict[str, torch.Tensor]:
    with np.load(DATA / "batch32.npz") as data:
        batch = {key: torch.from_numpy(data[key]) for key in BATCH_KEYS}
    batch["particle_type"] = batch["particle_type"].long()
    batch["particle_charge"] = batch["particle_charge"].long()
    return batch


def entry(events: int = 32) -> DatasetEntry:
    return DatasetEntry(hf_repo="test/repo", revision="test", files=DatasetFiles(directions="d.npz", steering="s.npz"),
                        pair="muon", events=events)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A local file in place of the Hugging Face download: batch32 with an event index and a priority."""
    with np.load(DATA / "batch32.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["event"] = np.arange(1000, 1032)
    arrays["priority"] = np.random.default_rng(0).permutation(32)
    path = tmp_path / "s.npz"

    def write(drop: str | None = None):
        np.savez(path, **{key: value for key, value in arrays.items() if key != drop})
        return arrays

    monkeypatch.setattr(events_module, "hf_hub_download", lambda *args, **kwargs: str(path))
    return write


def test_slots_match_the_paper():
    minus, plus = pair_slots(batch32(), "muon")
    with np.load(DATA / "batch32_muon_slots.npz") as stored:
        assert np.array_equal(minus.numpy(), stored["index_negative"])
        assert np.array_equal(plus.numpy(), stored["index_positive"])


@pytest.mark.parametrize("charges, reason", [((0, 0), "two negative"), ((2, 1), "no negative")])
def test_event_without_one_pair_stops(charges, reason):
    batch = batch32()
    minus, plus = pair_slots(batch, "muon")
    batch["particle_charge"][5, minus[5]] = charges[0]
    batch["particle_charge"][5, plus[5]] = charges[1]
    with pytest.raises(ValueError, match="1 events have not exactly one muon pair, the first at row 5"):
        pair_slots(batch, "muon")


def test_no_electron_pairs_in_muon_events():
    with pytest.raises(ValueError, match="32 events have not exactly one electron pair"):
        pair_slots(batch32(), "electron")


def test_edited_tokens():
    # one event: mu- in slot 1, mu+ in slot 3, another mu- in slot 4, a jet in slot 0, padding in slot 5
    batch = {"particle_type": torch.tensor([[1, 13, 0, 13, 13, 13]]),
             "particle_charge": torch.tensor([[1, 0, 1, 2, 0, 2]]),
             "particle_mask": torch.tensor([[1.0, 1, 0, 1, 1, 0]])}
    minus, plus = torch.tensor([1]), torch.tensor([3])
    mask = lambda *slots: torch.tensor([[i in slots for i in range(6)]])
    expected = {"pair": (mask(1), mask(3)), "minus": (mask(1), mask()), "plus": (mask(), mask(3)),
                "all_of_type": (mask(1, 4), mask(3))}
    for tokens, (want_minus, want_plus) in expected.items():
        got_minus, got_plus = edited_tokens(batch, minus, plus, "muon", tokens)
        assert torch.equal(got_minus, want_minus) and torch.equal(got_plus, want_plus), tokens


def test_load_all_events_in_file_order(fake_hub):
    arrays = fake_hub()
    events = load_events(entry(), "steering", 32)
    assert len(events) == 32 and np.array_equal(events.event, arrays["event"])
    assert events.batch["particle_type"].dtype == torch.long
    assert torch.equal(events.batch["particle_extra"], torch.from_numpy(arrays["particle_extra"]))


def test_load_fewer_events_takes_the_lowest_priority(fake_hub):
    arrays = fake_hub()
    events = load_events(entry(), "steering", 10)
    want = np.sort(np.argsort(arrays["priority"])[:10])
    assert np.array_equal(events.event, arrays["event"][want])
    assert np.array_equal(np.sort(events.priority), np.arange(10))
    assert np.all(np.diff(events.event) > 0)


def test_load_stops_on_a_bad_file(fake_hub):
    fake_hub()
    with pytest.raises(ValueError, match="the config asks for 40"):
        load_events(entry(), "steering", 40)
    with pytest.raises(ValueError, match="datasets.yaml says 31"):
        load_events(entry(events=31), "steering", 10)
    fake_hub(drop="priority")
    with pytest.raises(ValueError, match="has no \\['priority'\\]"):
        load_events(entry(), "steering", 10)


def test_batches_cover_every_event_once():
    batch = batch32()
    minus, plus = pair_slots(batch, "muon")
    events = events_module.Events(batch=batch, minus=minus, plus=plus, event=np.arange(32), priority=np.arange(32))
    seen = list(batches(events, 12, "cpu"))
    assert [index for index, *_ in seen] == [0, 1, 2]
    assert [len(m) for _, _, m, _ in seen] == [12, 12, 8]
    assert torch.equal(torch.cat([b["particle_type"] for _, b, _, _ in seen]), batch["particle_type"])


def test_class_of():
    classes = Classes(min_mass=0.3, windows={"Z": [81.19, 101.19], "Jpsi": [2.9, 3.3], "Upsilon": [9.2, 10.6]})
    mass = np.array([91.0, 81.19, 3.0, 9.5, 50.0, 0.3, 0.1, np.nan, np.inf, 200.0])
    assert list(class_of(mass, classes)) == ["Z", "rest", "Jpsi", "Upsilon", "rest", "", "", "", "", "rest"]


def test_pair_mass():
    # event 0: two muons back to back, 45.6 GeV each, in slots 2 and 0
    # event 1: two massless particles along the same direction; m^2 = 0 hits the floor
    p4 = torch.zeros(2, 3, 4)
    p4[0, 2] = torch.tensor([0.0, 0.0, 45.6, 45.6])
    p4[0, 0] = torch.tensor([0.0, 0.0, -45.6, 45.6])
    p4[1, 0] = torch.tensor([3.0, 0.0, 0.0, 3.0])
    p4[1, 1] = torch.tensor([5.0, 0.0, 0.0, 5.0])
    p4.requires_grad_(True)
    m = pair_mass(p4, torch.tensor([2, 0]), torch.tensor([0, 1]))
    assert torch.allclose(m, torch.tensor([91.2, 1e-9 ** 0.5]))
    m.log().sum().backward()
    assert torch.isfinite(p4.grad).all()


def test_select_keeps_the_lowest_priority_with_a_class():
    classes = Classes(min_mass=0.3, windows={"Z": [81.19, 101.19]})
    mass = np.array([91.0, 0.1, 50.0, 95.0, np.nan, 20.0])
    priority = np.array([5, 0, 1, 2, 3, 4])
    assert list(select(mass, priority, classes, 3)) == ["", "", "rest", "Z", "", "rest"]
    with pytest.raises(ValueError, match="4 pairs have a class, events.keep asks for 5"):
        select(mass, priority, classes, 5)
