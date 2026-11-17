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
        (lambda r: r["steering_vectors"].update(method="pca"), "Input should be"),
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
        (lambda r: r["readout"].update(spectrum_range=[300, 0.3]), "readout.spectrum_range"),
        (lambda r: r["readout"].update(r60_max_mass=500), "readout.r60_max_mass"),
        (lambda r: r["lineshape"].update(bins=5), "more bins than the 5 fitted parameters"),
        (lambda r: r["plots"]["zooms"].update(Z=[140, 50]), r"plots.zooms\['Z'\]"),
        (lambda r: r["sampler"].update(on_nan="zero"), "Input should be 'set_to_zero' or 'stop'"),
    ],
)
def test_bad_values_stop(tmp_path, change, message):
    path = write_variant(tmp_path, change)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_on_nan_defaults_to_the_paper(tmp_path):
    assert load_config(MAIN).sampler.on_nan == "set_to_zero"
    assert load_config(write_variant(tmp_path, lambda r: r["sampler"].pop("on_nan"))).sampler.on_nan == "set_to_zero"
    assert load_config(write_variant(tmp_path, lambda r: r["sampler"].update(on_nan="stop"))).sampler.on_nan == "stop"


def test_steps_range(tmp_path):
    config = load_config(write_variant(tmp_path, lambda r: r["steering"].update(steps=[180, 199])))
    assert config.steering.steps == [180, 199]


def test_read_datasets_alone():
    datasets = read_datasets(MAIN.parent.parent / "datasets.yaml")
    assert datasets["validation_muon_pairs"].files.steering == "steering-dataset.npz"


PAPER = MAIN.parent
HALF_GRID = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]


@pytest.mark.parametrize(
    "file, method, alphas",
    [
        ("methods_z_fisher.yaml", "fisher", [-4, -3, -2, -1, 1, 2, 3, 4]),
        ("methods_z_meandiff.yaml", "meandiff", [-1, -0.5, -0.25, 0.25, 0.5, 1]),
    ],
)
def test_methods_configs(file, method, alphas):
    config = load_config(PAPER / file)
    assert (config.target, config.steering.block) == ("Z", 1)
    assert config.steering_vectors.method == method
    assert config.steering_vectors.balance == "equal"
    assert config.steering_vectors.in_time == "per_step"
    assert config.steering.alphas == alphas
    assert config.steering.seeds == [2041, 2042, 2043]


@pytest.mark.parametrize("block", [0, 3])
def test_blocks_configs(block):
    config = load_config(PAPER / f"blocks_z_block{block}.yaml")
    assert (config.target, config.steering.block, config.steering_vectors.method) == ("Z", block, "fkt")
    assert config.steering.alphas == HALF_GRID
    assert config.steering.seeds == [2041, 2042, 2043]


@pytest.mark.parametrize("file, target", [("quarkonia_jpsi.yaml", "Jpsi"), ("quarkonia_upsilon.yaml", "Upsilon")])
def test_quarkonia_configs(file, target):
    config = load_config(PAPER / file)
    assert (config.target, config.steering.block, config.steering_vectors.method) == (target, 1, "fkt")
    assert config.steering.alphas == HALF_GRID
    assert config.steering.seeds == [2041, 2042, 2043]


def test_all_muons_config():
    config = load_config(PAPER / "all_muons.yaml")
    assert (config.target, config.steering.block, config.steering_vectors.method) == ("Z", 1, "fkt")
    assert config.steering.tokens == "all_of_type"
    assert config.steering.alphas == [-2, -1, 1, 2]
    assert config.steering.seeds == [2041]
    assert config.events.steering == 1000000


ABLATION = MAIN.parent / "ablation_z.yaml"


def test_ablation_config():
    config = load_config(ABLATION)
    assert (config.target, config.steering_vectors.method, config.steering_vectors.in_time) == ("Z", "fkt", "per_step")
    assert config.ablation.block == 1 and config.ablation.sample == "directions"
    assert config.ablation.seeds == [2040, 2041, 2042]
    assert config.gradients.seed == 2040
    assert load_config(MAIN).ablation is None


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r["ablation"].update(block=7), "ablation.block: block 7 is not in gradients.blocks"),
        (lambda r: r["ablation"].update(seeds=[]), "ablation.seeds: need at least one seed"),
        (lambda r: r["ablation"].update(seeds=[2040, 2040]), "no repeats"),
        (lambda r: r["ablation"].update(sample="test"), "Input should be 'directions' or 'steering'"),
        (lambda r: r["ablation"].update(mean="zero"), "Extra inputs"),
        (lambda r: r["ablation"].pop("sample"), "Field required"),
    ],
)
def test_bad_ablation_values_stop(tmp_path, change, message):
    raw = yaml.safe_load(ABLATION.read_text())
    for key in ("classes", "networks", "datasets"):
        raw[key] = str(ABLATION.parent / raw[key])
    change(raw)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "path, change, message",
    [
        (MAIN, lambda r: r.pop("steering"), "steering: required by the steering run"),
        (MAIN, lambda r: r.pop("plots"), "plots: required by the steering run"),
        (ABLATION, lambda r: r.update(steering=yaml.safe_load(MAIN.read_text())["steering"]),
         "steering: the ablation run does not read these"),
        (ABLATION, lambda r: r["ablation"].update(peak_range=[140, 50]), "ablation.peak_range"),
    ],
)
def test_each_run_has_exactly_its_sections(tmp_path, path, change, message):
    raw = yaml.safe_load(path.read_text())
    for key in ("classes", "networks", "datasets"):
        raw[key] = str(path.parent / raw[key])
    change(raw)
    variant = tmp_path / "run.yaml"
    variant.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load_config(variant)
