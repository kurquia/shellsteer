"""The `steer` command on CPU: the steering run and the mean ablation through `cli.main`, at the size of
test_pipeline.py.

The events are the 32 of batch32.npz repeated 64 times, read through a local file in place of the Hugging Face
download, as in test_pipeline.py. The size comes from --set: 2048 events, 4 Euler steps, block 1 only, two seeds, and
alpha = -1 and +1. The steering run goes once per module; the ablation run starts from a copy of its gradients and
steering vectors, which the two configs share. The other tests check the overrides, the stage dependencies, the
missing-file message, and the refusal to overwrite.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

import shellsteer.events as events_module
import shellsteer.sampler as sampler_module
from shellsteer.activation_means import load_activation_means
from shellsteer.cli import BY_NAME, STAGES, StageError, ancestors, main, read_config
from shellsteer.events import BATCH_KEYS
from shellsteer.gradients import load_moments
from shellsteer.steered_generation import load_generated, load_kept

DATA = Path(__file__).parent / "data"
PAPER = Path(__file__).parent.parent / "configs" / "paper"
MAIN, ABLATION = PAPER / "main_z_block1.yaml", PAPER / "ablation_z.yaml"
EVENTS = 2048
COMMON = [f"datasets.validation_muon_pairs.events={EVENTS}", f"events.directions={EVENTS}",
          f"events.steering={EVENTS}", "events.keep=1800", "sampler.steps=4", "sampler.batch_size=256",
          "gradients.blocks=[1]"]
SMALL = [*COMMON, "steering.seeds=[2041, 2042]", "steering.alphas=[-1, 1]", "name=small"]
SMALL_ABLATION = [*COMMON, "ablation.seeds=[2040, 2041]", "name=small_ablation"]
GPU_STAGES = {stage.name for stage in STAGES if stage.gpu}


def steer(stage: str, runs: Path, *extra: str, config: Path = MAIN, overrides: list[str] = SMALL) -> int:
    """Run one stage; `extra` holds more KEY=VALUE overrides and flags such as --overwrite."""
    flags = [value for value in extra if value.startswith("--")]
    sets = [arg for value in (*overrides, *extra) if not value.startswith("--") for arg in ("--set", value)]
    return main([stage, str(config), "--runs", str(runs), *sets, *flags,
                 *(["--device", "cpu"] if stage in GPU_STAGES else [])])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The run folder of the steering run, with every stage done, and that of the ablation run."""
    folder = tmp_path_factory.mktemp("cli")
    with np.load(DATA / "batch32.npz") as data:
        arrays = {key: np.tile(data[key], (EVENTS // 32,) + (1,) * (data[key].ndim - 1)) for key in BATCH_KEYS}
    arrays["event"] = np.arange(EVENTS)
    arrays["priority"] = np.random.default_rng(0).permutation(EVENTS)
    np.savez(folder / "sample.npz", **arrays)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(events_module, "hf_hub_download", lambda *args, **kwargs: str(folder / "sample.npz"))
        runs = folder / "runs"
        for stage in ("gradients", "steering-vectors", "generate", "readout", "plots"):
            assert steer(stage, runs) == 0
        ablation = runs / "small_ablation"
        ablation.mkdir()
        for name in ("config.yaml", "gradient_moments.npz", "directions_kept.npz", "steering_vectors.npz"):
            shutil.copy(runs / "small" / name, ablation / name)
        for stage in ("activation-means", "ablate", "ablation-readout"):
            assert steer(stage, runs, config=ABLATION, overrides=SMALL_ABLATION) == 0
        yield runs


def files_of(run: Path) -> list[str]:
    return sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())


def test_the_steering_run(runs, capsys):
    run = runs / "small"
    assert files_of(run) == ["config.yaml", "directions_kept.npz", "generated/kept.npz",
                             *(f"generated/seed_{seed}/alpha_{alpha}.npz"
                               for seed in (2041, 2042) for alpha in ("+0", "+1", "-1")),
                             "gradient_moments.npz", "plots/pole_and_width.pdf", "plots/selectivity.pdf",
                             "plots/steered_spectrum.pdf", "readout.json", "steering_vectors.npz"]
    stored = yaml.safe_load((run / "config.yaml").read_text())
    assert stored["sampler"]["steps"] == 4 and stored["steering"]["seeds"] == [2041, 2042]
    readout = json.loads((run / "readout.json").read_text())
    assert readout["selectivity curve"]["alpha"] == [-1.0, 1.0]
    assert readout["selectivity curve"]["target shift"][1] > readout["selectivity curve"]["target shift"][0]
    assert readout["gradient response"] > 0
    assert set(readout["pole response fit"]) == {"S_fit", "R^2"}
    assert len(readout["per seed and alpha"]) == 6 and sum(readout["kept pairs"].values()) == 1800


def test_the_ablation_run(runs):
    run = runs / "small_ablation"
    assert files_of(run) == ["ablated/seed_2040/baseline.npz", "ablated/seed_2040/block_1.npz",
                             "ablated/seed_2041/baseline.npz", "ablated/seed_2041/block_1.npz",
                             "ablation_readout.json", "activation_means.npz", "config.yaml", "directions_kept.npz",
                             "gradient_moments.npz", "steering_vectors.npz"]
    stored = yaml.safe_load((run / "config.yaml").read_text())
    assert stored["ablation"]["seeds"] == [2040, 2041] and stored["steering"] is None
    readout = json.loads((run / "ablation_readout.json").read_text())
    assert [row["seed"] for row in readout["per seed"]] == [2040, 2041]
    ratios = [row["spread of log m in the target window"]["ratio"] for row in readout["per seed"]]
    assert readout["spread ratio, mean over seeds"] == pytest.approx(np.mean(ratios))
    for key in ("peak height change, seed-average histograms", "deviation below max mass, seed-average histograms",
                "R_60, mean over seeds"):
        assert np.isfinite(readout[key])
    assert all(row["pairs"] <= 1800 for row in readout["per seed"])


def test_a_stage_of_the_other_run_stops(runs, capsys):
    assert steer("ablate", runs) == 1
    assert "is a stage of the ablation run" in capsys.readouterr().err
    assert steer("readout", runs, config=ABLATION, overrides=SMALL_ABLATION) == 1
    assert "is a stage of the steering run" in capsys.readouterr().err


def test_rerunning_a_stage_refuses(runs, capsys):
    assert steer("generate", runs) == 1
    assert "exists; pass --overwrite" in capsys.readouterr().err
    assert steer("ablate", runs, config=ABLATION, overrides=SMALL_ABLATION) == 1
    assert "exists; pass --overwrite" in capsys.readouterr().err


def test_settings_of_a_needed_stage_must_not_change(runs, capsys):
    assert steer("readout", runs, "sampler.batch_size=128", "--overwrite") == 1
    assert "sampler.batch_size" in capsys.readouterr().err
    assert steer("ablation-readout", runs, "ablation.seeds=[2040, 2042]", config=ABLATION, overrides=SMALL_ABLATION) == 1
    assert "ablation.seeds" in capsys.readouterr().err


def test_overwrite_after_the_later_outputs_are_gone(runs, capsys):
    """Runs last in this file: it changes the steering run."""
    run = runs / "small"
    assert steer("readout", runs, "readout.spectrum_bins=100", "--overwrite") == 1
    assert "plots was made with the old ones" in capsys.readouterr().err
    shutil.rmtree(run / "plots")
    assert steer("readout", runs, "readout.spectrum_bins=100", "--overwrite") == 0
    assert "removed" in capsys.readouterr().out
    assert yaml.safe_load((run / "config.yaml").read_text())["readout"]["spectrum_bins"] == 100


def test_a_stage_stops_when_a_needed_output_is_missing(tmp_path, capsys):
    assert steer("readout", tmp_path) == 1
    assert "gradient_moments.npz is missing; run `steer gradients` first" in capsys.readouterr().err
    assert steer("ablate", tmp_path, config=ABLATION, overrides=SMALL_ABLATION) == 1
    assert "gradient_moments.npz is missing; run `steer gradients` first" in capsys.readouterr().err
    assert not (tmp_path / "small").exists() and not (tmp_path / "small_ablation").exists()


def test_stage_dependencies():
    def names(stage: str) -> list[str]:
        return [s.name for s in ancestors(BY_NAME[stage])]

    assert names("plots") == ["gradients", "steering-vectors", "generate", "readout"]
    assert names("ablation-readout") == ["gradients", "steering-vectors", "activation-means", "ablate"]
    assert names("activation-means") == ["gradients"]


def test_overrides_are_validated():
    config = read_config(MAIN, ["sampler.batch_size=1024", "steering.seeds=[2041, 2042]", "steering.steps=[0, 9]"])
    assert config.sampler.batch_size == 1024 and config.steering.seeds == [2041, 2042]
    assert config.steering.steps == [0, 9]
    with pytest.raises(StageError, match="expected key=value"):
        read_config(MAIN, ["sampler.batch_size"])
    with pytest.raises(StageError, match="has no key 'sampelr'"):
        read_config(MAIN, ["sampelr.steps=4"])
    with pytest.raises(StageError, match="steering is not in this config"):
        read_config(ABLATION, ["steering.block=1"])
    with pytest.raises(Exception, match="batch_sise"):                       # unknown leaf: pydantic, extra forbidden
        read_config(MAIN, ["sampler.batch_sise=4"])
    with pytest.raises(Exception, match="method"):                           # outside the options
        read_config(MAIN, ["steering_vectors.method=pca"])
    with pytest.raises(StageError, match="plain folder name"):
        read_config(MAIN, ["name=../elsewhere"])


def test_each_set_takes_one_key_and_config_may_come_last(tmp_path, capsys):
    assert main(["readout", "--set", "sampler.steps=four", str(MAIN), "--runs", str(tmp_path)]) == 1
    assert "sampler.steps" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["gradient", str(MAIN)])                                        # not a stage: argparse stops


def test_nan_counts_are_in_the_metadata_of_every_file(runs):
    small, ablation = runs / "small", runs / "small_ablation"
    counts = [load_generated(small / "generated" / "seed_2042" / "alpha_-1.npz")[1]["NaN counts"],
              load_generated(ablation / "ablated" / "seed_2040" / "block_1.npz")[1]["NaN counts"],
              load_kept(small / "directions_kept.npz")[2]["NaN counts"],
              *(load_moments(small / "gradient_moments.npz")[1][key] for key in ("NaN counts",
                                                                                 "NaN counts of the baseline")),
              load_activation_means(ablation / "activation_means.npz")[1]["NaN counts"]]
    for count in counts:
        assert count["sampler.on_nan"] == "set_to_zero" and count["events"] == EVENTS
        assert count["NaN coordinates of the final state"] == count["NaN entries of the four-momenta"] == 0


def test_a_run_made_before_on_nan_existed_counts_as_set_to_zero(runs, tmp_path, capsys):
    shutil.copytree(runs / "small", tmp_path / "small")
    config = yaml.safe_load((tmp_path / "small" / "config.yaml").read_text())
    del config["sampler"]["on_nan"]
    (tmp_path / "small" / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    assert steer("readout", tmp_path, "--overwrite") == 0
    assert steer("readout", tmp_path, "sampler.on_nan=stop", "--overwrite") == 1
    assert "sampler.on_nan" in capsys.readouterr().err


def test_on_nan_stop_ends_the_stage_with_the_batch(tmp_path, capsys, monkeypatch):
    decode_without_nan = sampler_module.decode

    def decode_with_nan(*args, **kwargs):
        four = decode_without_nan(*args, **kwargs)
        four[0, 2, 0] = float("nan")
        return four
    with np.load(DATA / "batch32.npz") as data:
        arrays = {key: np.tile(data[key], (EVENTS // 32,) + (1,) * (data[key].ndim - 1)) for key in BATCH_KEYS}
    arrays["event"], arrays["priority"] = np.arange(EVENTS), np.arange(EVENTS)
    np.savez(tmp_path / "sample.npz", **arrays)
    monkeypatch.setattr(events_module, "hf_hub_download", lambda *args, **kwargs: str(tmp_path / "sample.npz"))
    monkeypatch.setattr(sampler_module, "decode", decode_with_nan)
    assert steer("gradients", tmp_path / "runs", "sampler.on_nan=stop") == 1
    assert ("batch 0 of seed 2040: 1 NaN entries in the four-momenta of 1 of 256 events (sampler.on_nan is stop)"
            in capsys.readouterr().err)
