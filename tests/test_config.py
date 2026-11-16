from pathlib import Path

import pytest
import yaml

from shellsteer.config import load_config, read_datasets

MAIN = Path(__file__).parent.parent / "configs" / "paper" / "main_z_block1.yaml"


def test_main_config_loads():
    config = load_config(MAIN)
    assert config.target == "Z"
    assert config.classes.windows["Z"] == [81.19, 101.19]
    assert config.steering.block == 1
    assert config.steering.alphas == [-2, -1, 1, 2]
    assert config.steering.steps == "all"
    assert config.model_dump()["classes"]["windows"]["Jpsi"] == [2.9, 3.3]
    assert config.datasets[config.events.dataset].pair == "muon"
    assert config.events.directions == config.events.steering == 1040000


def write_variant(tmp_path, change):
    raw = yaml.safe_load(MAIN.read_text())
    raw["classes"] = str(MAIN.parent / raw["classes"])
    raw["networks"] = str(MAIN.parent / raw["networks"])
    raw["datasets"] = str(MAIN.parent / raw["datasets"])
    change(raw)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r["steering"].update(blok=1), "Extra inputs"),
        (lambda r: r["steering"].pop("seeds"), "Field required"),
        (lambda r: r["directions"].update(method="pca"), "Input should be"),
        (lambda r: r["steering"].update(alphas=[0, 1]), "leave out 0"),
        (lambda r: r["steering"].update(block=7), "not in gradients.blocks"),
        (lambda r: r["steering"].update(steps=[150, 100]), "first <= last"),
        (lambda r: r["steering"].update(norm_restore="no"), "valid boolean"),
        (lambda r: r.update(target="W"), "target"),
        (lambda r: r.update(networks=str(MAIN.parent.parent / "classes.yaml")), "networks"),
        (lambda r: r["events"].update(dataset="all_events"), "no entry in the datasets file"),
        (lambda r: r["events"].update(steering=2000000), "events in the file"),
        (lambda r: r["events"].update(directions=0), "events in the file"),
        (lambda r: r["events"].update(n_A=1000), "Extra inputs"),
        (lambda r: r["events"].update(steering=1000), "more than the 1000 events of events.steering"),
    ],
)
def test_bad_values_stop(tmp_path, change, message):
    path = write_variant(tmp_path, change)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_steps_range(tmp_path):
    config = load_config(write_variant(tmp_path, lambda r: r["steering"].update(steps=[180, 199])))
    assert config.steering.steps == [180, 199]


def test_read_datasets_alone():
    datasets = read_datasets(MAIN.parent.parent / "datasets.yaml")
    assert datasets["validation_muon_pairs"].files.steering == "steering.npz"
