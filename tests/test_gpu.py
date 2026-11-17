"""Tests on a CUDA GPU against the paper's stored outputs. Without a GPU they skip; select them with `-m gpu`.

Run, from the repository root, on a machine with one CUDA GPU:

    .venv/bin/python -m pytest -q -m gpu tests -rs

GPU: a whole NVIDIA B200 or A100, or a MIG slice with at least 24 GB. Peak memory is about 15 GB (the one-batch
gradient pass at block 1). The tests use float32 and switch on neither TF32 nor torch.compile, as the paper; the
package never switches them on either.
Time (estimate): 9 generations of 8192 events at 200 Euler steps, two gradient passes, and small runs. The paper's
steering run (B200, float32, no compile) took about 7 s per 8192-event generation, so expect 3 to 5 minutes on a B200
and about twice that on an A100, plus the model load. The 3,445 pairs/s of the paper's baseline used TF32 and
torch.compile and does not apply here.
Disk: nothing is written. The first run downloads the network from Hugging Face (pomidori73/ShellFlow, 24 MB, into
~/.cache/huggingface); the events come from the fixtures in tests/data, so no dataset is downloaded.

Paper references (batch 0 = the first 8192 events of a sample, file order, noise seed `batch_seed(seed, 0)`):

    A_batch0_seed2040_steps200.npz          directions sample: conditioning, pair slots, and the baseline mass of the
                                            gradient pass (seed 2040), made on the old A100s.
    directions_batch0_paper_moments_block1.npz
                                            the paper's class of each kept pair of that batch, and the full-sample
                                            class means of the unit and of the raw gradients at block 1 (float32).
    steering_batch0_seed2041_steps200.npz   steering sample: conditioning, event, priority, and the masses of the
                                            paper's main steering run (seed 2041, block 1, FKT per step) at alpha = 0
                                            and at the coefficients -0.122045 and +0.122045, made on B200s.
    xi_z_block1.npy                         the paper's unit steering vector of the Z at block 1, (200, 2, 128).

The paper's code stored xi / S and a coefficient c = alpha S rounded to 6 digits (S = 0.1220447, Eq. 21); the release
stores the unit xi and takes alpha, so the tests steer with alpha = c / S = 1.0000029 to repeat the paper's edit.

Noise. The GPU draws the prior with a launch grid capped at (SMs x threads per SM) threads, so a GPU with fewer SMs
than the reference GPU gives other noise to the last rows of a batch: on a B200 the rows 7372 to 8191 of an A100
batch differ, and the other way round. `rows_with_reference_noise` computes these rows from the GPU's properties; the
comparison leaves them out and prints how many. On a 1g MIG slice almost no row has the reference noise, so the
comparisons with the paper skip there and say so.
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
from shellsteer.steered_generation import Edit, generate_pairs
from shellsteer.steering_vectors import gradient_response, steering_vector

def needs_gpu(test):
    """Mark `test` as a GPU test (select with -m gpu) and skip it without CUDA."""
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")(test))

DATA = Path(__file__).parent / "data"
CONFIGS = Path(__file__).parent.parent / "configs"
DIRECTIONS_BATCH0 = DATA / "A_batch0_seed2040_steps200.npz"
DIRECTIONS_MOMENTS = DATA / "directions_batch0_paper_moments_block1.npz"
STEERING_BATCH0 = DATA / "steering_batch0_seed2041_steps200.npz"
XI_Z_BLOCK1 = DATA / "xi_z_block1.npy"
DIRECTIONS_SEED, STEERING_SEED, STEPS, BATCH, BLOCKS = 2040, 2041, 200, 8192, [0, 1, 2, 3, 4, 5]

# (SMs, threads per SM) of the GPUs that made the reference files.
A100 = (108, 2048)    # the paper's directions sample and its gradient pass (2026-09-29)
B200 = (148, 2048)    # the paper's steering run step_balanced_Z_half (2026-10-14)

# Pair-by-pair tolerances on |log m - log m_paper|, rows with the reference noise only. Measured on the paper's two
# baselines of the whole steering sample (seed 2041) made on A100s and on B200s, 936,244 pairs with the same noise,
# per batch of 8192: median 1.9e-7 (largest batch median 3.3e-7), signed median 0 in every batch, within 1e-3 for
# 97.1-97.9%, within 1e-2 for 99.4-99.7%, same class for 99.93-99.99%; 37% bit-identical. Float32 rounding of the
# matmuls grows through 200 Euler steps for a few percent of the pairs, so the tolerances are on fractions.
MEDIAN_DIFFERENCE_MAX = 1e-5        # median |difference|; a wrong step count or edit size moves it by > 1e-3
SIGNED_MEDIAN_MAX = 1e-6            # |median of the signed difference|; a systematic shift of the masses
WITHIN_1E3_MIN = 0.95               # fraction within 1e-3; measured 0.971 in the worst batch
WITHIN_1E2_MIN = 0.98               # fraction within 1e-2; measured 0.994 in the worst batch
SAME_CLASS_MIN = 0.995              # fraction with the same class (configs/classes.yaml); measured 0.9993 worst
MIN_ROWS = 1000                     # fewer rows with the reference noise than this: the comparison skips

# One-batch gradient pass against the paper's full-sample moments (no per-batch sums were stored).
# S of Eq. 21 with the paper's xi: 0.1220 over 181,987 Z pairs. Per pair the sum over steps has a standard deviation
# of at most 0.223 (sum over steps of the per-step deviations, from the paper's raw second moments), so the mean of
# the ~560 Z pairs of batch 0 scatters by at most 0.0094. The tolerance is 4 times that, rounded up, because batch 0
# is not a random draw: the files are sorted by physics sample.
PAPER_S_Z_BLOCK1 = 0.12204465
S_TOLERANCE = 0.04
# Cosine per step of the one-batch unit class mean with the paper's. Sampling noise alone gives about 0.98 or more
# (mean length 0.15 to 0.8, 500 to 4600 pairs per class); the margin is again for the composition of batch 0. A wrong
# block, token order, or step index gives cosines far below.
UNIT_MEAN_COSINE_MIN = 0.9          # median over the 200 steps, every class


@pytest.fixture(scope="module")
def network():
    return load_network(read_networks(CONFIGS / "networks.yaml")["shellflow"], "cuda")


@pytest.fixture(scope="module")
def classes():
    return Classes.model_validate(yaml.safe_load((CONFIGS / "classes.yaml").read_text()))


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def directions():
    return load(DIRECTIONS_BATCH0)


@pytest.fixture(scope="module")
def steering():
    return load(STEERING_BATCH0)


def make_events(stored, n, event_key):
    batch = {key: torch.from_numpy(stored[key][:n]) for key in BATCH_KEYS}
    batch["particle_type"] = batch["particle_type"].long()       # stored as int16
    batch["particle_charge"] = batch["particle_charge"].long()
    minus, plus = pair_slots(batch, "muon")
    return Events(batch=batch, minus=minus, plus=plus, event=stored[event_key][:n], priority=np.arange(n))


def draw_identity(numel, gpu, offset):
    """(4, numel) identity of each number of one torch.randn call on CUDA: Philox offset, thread, loop, component.

    torch launches 256-thread blocks, at most SMs x threads per SM / 256 of them; element j goes to thread j % stride,
    component (j // stride) % 4 of curand_normal4 in loop j // (4 stride), stride = 256 x blocks
    (ATen/native/cuda/DistributionTemplates.h). Returns the identities and the generator's offset after the call.
    """
    sms, threads = gpu
    stride = min(-(-numel // 256), sms * threads // 256) * 256
    j = np.arange(numel)
    identity = np.stack([np.full(numel, offset), j % stride, j // (4 * stride), (j // stride) % 4])
    return identity, offset + ((numel - 1) // (4 * stride) + 1) * 4


def rows_with_reference_noise(batch_size, slots, reference):
    """(batch_size,) True where `sampler.prior` draws the same numbers on this GPU as on the reference GPU.

    prior draws randn(B, N), randn(B, N, 3), and randn(B, N) from one fresh generator; a row counts if all its
    numbers match.
    """
    props = torch.cuda.get_device_properties(0)
    here = (props.multi_processor_count, props.max_threads_per_multi_processor)
    same = np.ones(batch_size, dtype=bool)
    offset_here = offset_reference = 0
    for per_row in (slots, 3 * slots, slots):
        a, offset_here = draw_identity(batch_size * per_row, here, offset_here)
        b, offset_reference = draw_identity(batch_size * per_row, reference, offset_reference)
        same &= (a == b).all(0).reshape(batch_size, per_row).all(1)
    return same


def compare_masses(mass, paper, rows, classes, what):
    """Assert the pair-by-pair tolerances above on `rows` and print the numbers that matter."""
    if rows.sum() < MIN_ROWS:
        pytest.skip(f"{what}: only {int(rows.sum())} of {len(rows)} rows have the reference noise on "
                    f"{torch.cuda.get_device_name()} (a MIG slice draws other noise); need {MIN_ROWS}")
    mass, paper = np.asarray(mass, np.float64)[rows], np.asarray(paper, np.float64)[rows]
    signed = np.log(mass) - np.log(paper)
    difference = np.abs(signed)
    numbers = {
        "rows compared": int(rows.sum()), "rows left out (other noise)": int((~rows).sum()),
        "bit-identical": float(np.mean(mass.astype(np.float32) == paper.astype(np.float32))),
        "within 1e-3": float(np.mean(difference <= 1e-3)), "within 1e-2": float(np.mean(difference <= 1e-2)),
        "median |difference|": float(np.median(difference)), "signed median": float(np.median(signed)),
        "largest |difference|": float(difference.max()),
        "same class": float(np.mean(class_of(mass, classes) == class_of(paper, classes))),
    }
    report = f"{what} on {torch.cuda.get_device_name()}: " + ", ".join(f"{k} {v:.4g}" for k, v in numbers.items())
    print(report)
    assert numbers["median |difference|"] <= MEDIAN_DIFFERENCE_MAX, report
    assert abs(numbers["signed median"]) <= SIGNED_MEDIAN_MAX, report
    assert numbers["within 1e-3"] >= WITHIN_1E3_MIN, report
    assert numbers["within 1e-2"] >= WITHIN_1E2_MIN, report
    assert numbers["same class"] >= SAME_CLASS_MIN, report


def test_paper_coefficients_are_alpha_one_with_the_unit_vector(steering):
    """CPU check of the conversion: S of Eq. 21 from the paper's raw Z mean and the unit xi is the paper's S, and the
    stored coefficients are -S and +S to 6 digits."""
    moments = load(DIRECTIONS_MOMENTS)
    xi = np.load(XI_Z_BLOCK1).astype(np.float64)
    assert np.allclose(np.linalg.norm(xi.reshape(STEPS, -1), axis=1), 1, rtol=0, atol=1e-6)
    s = float(np.einsum("kw,kw->", moments["raw mean"][0].astype(np.float64), xi.reshape(STEPS, -1)))
    assert s == pytest.approx(PAPER_S_Z_BLOCK1, rel=1e-6)          # float32 storage of the mean
    assert float(steering["scale"]) == pytest.approx(PAPER_S_Z_BLOCK1, rel=1e-7)
    for name, sign in (("minus", -1), ("plus", 1)):
        assert float(steering[f"coefficient {name}"]) / float(steering["scale"]) == pytest.approx(sign, abs=5e-6)


@needs_gpu
def test_directions_baseline_agrees_with_the_paper(network, directions, classes):
    """Batch 0 of the directions sample (seed 2040) against the paper's baseline masses, made on A100s."""
    events = make_events(directions, BATCH, "row")
    assert np.array_equal(events.minus.numpy(), directions["index_negative"])
    assert np.array_equal(events.plus.numpy(), directions["index_positive"])
    mass = generate_pairs(network, events, None, steps=STEPS, batch_size=BATCH, seed=DIRECTIONS_SEED,
                          device="cuda").mass
    rows = rows_with_reference_noise(BATCH, directions["particle_type"].shape[1], A100)
    compare_masses(mass, directions["mass"], rows, classes, "directions baseline, batch 0, seed 2040")


@needs_gpu
def test_steering_baseline_and_alpha_minus_one_and_plus_one_agree_with_the_paper(network, steering, classes):
    """Batch 0 of the steering sample (seed 2041): the baseline, and the Z edit at block 1 for alpha = -1 and +1, FKT
    per step, the paper's xi, against the paper's main steering run, made on B200s (paper Sec. 5, Fig. 3)."""
    events = make_events(steering, BATCH, "event")
    rows = rows_with_reference_noise(BATCH, steering["particle_type"].shape[1], B200)
    xi = torch.from_numpy(np.load(XI_Z_BLOCK1))
    common = dict(steps=STEPS, batch_size=BATCH, seed=STEERING_SEED, device="cuda")
    baseline = generate_pairs(network, events, None, **common).mass
    compare_masses(baseline, steering["mass baseline"], rows, classes, "steering baseline, batch 0, seed 2041")

    z = class_of(steering["mass baseline"], classes) == "Z"
    for name in ("minus", "plus"):
        alpha = float(steering[f"coefficient {name}"]) / float(steering["scale"])
        edit = Edit(xi=xi, alpha=alpha, block=1, tokens="pair", pair="muon", steps=None, norm_restore=False)
        mass = generate_pairs(network, events, edit, **common).mass
        shift = np.median(np.log(mass[z]) - np.log(baseline[z]))
        paper_shift = np.median(np.log(steering[f"mass {name}"][z]) - np.log(steering["mass baseline"][z]))
        print(f"alpha = {alpha:+.7f}: median shift of log m of the Z pairs {shift:+.5f}, paper {paper_shift:+.5f}")
        compare_masses(mass, steering[f"mass {name}"], rows, classes, f"steered alpha {alpha:+.7f}, batch 0")


@needs_gpu
def test_one_batch_gradient_pass_agrees_with_the_paper(network, directions, classes):
    """Gradient pass on batch 0 of the directions sample at block 1 (paper Sec. 4.1), with the paper's kept pairs.

    The paper stored only full-sample sums, so the test checks what one batch can show: the class of each pair, S of
    Eq. 21 with the paper's xi, and the per-step cosine of the unit class means with the paper's. It prints the cosine
    of the one-batch FKT vector with the paper's xi; 560 Z pairs in 256 dimensions make that too noisy to assert.
    """
    paper = load(DIRECTIONS_MOMENTS)
    events = make_events(directions, BATCH, "row")
    kept = paper["class"].astype(object)
    moments, _ = gradient_moments(network, events, kept, classes, blocks=[1], steps=STEPS, batch_size=BATCH,
                                  seed=DIRECTIONS_SEED, device="cuda")
    assert [str(c) for c in moments["classes"]] == [str(c) for c in paper["classes"]]
    assert int(moments["non-finite pairs"]) == 0
    paper_count = np.array([(kept == str(c)).sum() for c in paper["classes"]])
    print(f"pairs per class {dict(zip(paper['classes'], moments['count']))}, paper {paper_count.tolist()}")
    # A kept pair can change class where the noise differs from the A100's, or through rounding (SAME_CLASS_MIN).
    other_noise = int((~rows_with_reference_noise(BATCH, directions["particle_type"].shape[1], A100)).sum())
    allowed = other_noise + int(np.ceil((1 - SAME_CLASS_MIN) * (kept != "").sum()))
    assert np.all(np.abs(moments["count"] - paper_count) <= allowed), f"counts differ by more than {allowed}"

    xi = np.load(XI_Z_BLOCK1)
    s = gradient_response(moments, xi, "Z", 1)
    print(f"S (Eq. 21) of batch 0: {s:.4f}, paper {PAPER_S_Z_BLOCK1:.4f} over the whole sample")
    assert abs(s - PAPER_S_Z_BLOCK1) <= S_TOLERANCE

    for c, name in enumerate(moments["classes"]):
        mean = moments["unit sum"][c, 0] / moments["count"][c]                       # (steps, 2 width)
        reference = paper["unit mean"][c].astype(np.float64)
        cosine = np.einsum("kw,kw->k", mean, reference) / (np.linalg.norm(mean, axis=1)
                                                           * np.linalg.norm(reference, axis=1))
        print(f"{name}: cosine of the unit class mean with the paper's, median {np.median(cosine):.4f}, "
              f"smallest {cosine.min():.4f}")
        assert np.median(cosine) >= UNIT_MEAN_COSINE_MIN, name

    vector = steering_vector(moments, "Z", 1, "fkt", "per_step", "equal")
    cosine = np.einsum("kw,kw->k", vector.reshape(STEPS, -1), xi.reshape(STEPS, -1).astype(np.float64))
    print(f"one-batch FKT xi of the Z against the paper's: cosine median {np.median(cosine):.3f}, "
          f"smallest {cosine.min():.3f}, largest {cosine.max():.3f}")


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
                          grad=True, on_step=on_step(*probes)).four_momenta
            g = torch.autograd.grad(log_pair_mass(p4, minus, plus).sum(), [p.delta for p in probes])
        grads.append(torch.stack(g).flatten(-2).double())
        masses.append(pair_mass(p4.detach(), minus, plus).double().cpu().numpy())
    return torch.cat(grads, dim=2), np.concatenate(masses)


@needs_gpu
def test_moments_equal_the_sums_of_the_per_pair_gradients(network, directions, classes):
    """512 pairs in two batches, all six blocks, 200 steps, the paper's classes."""
    events, batch_size, keep = make_events(directions, 512, "row"), 256, 400
    mass = generate_pairs(network, events, None, steps=STEPS, batch_size=batch_size, seed=DIRECTIONS_SEED,
                          device="cuda").mass
    kept = select(mass, events.priority, classes, keep)
    moments, _ = gradient_moments(network, events, kept, classes, blocks=BLOCKS, steps=STEPS,
                                  batch_size=batch_size, seed=DIRECTIONS_SEED, device="cuda")

    g, pass_mass = per_pair_gradients(network, events, STEPS, batch_size, DIRECTIONS_SEED)
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


@needs_gpu
def test_steering_at_alpha_zero_is_the_baseline_and_runs_at_alpha_one(network, directions):
    """Z vector of the paper at block 1 on 1024 pairs: alpha = 0 reproduces the unedited run bit for bit."""
    events = make_events(directions, 1024, "row")
    batch = {key: value.cuda() for key, value in events.batch.items()}
    minus, plus = events.minus.cuda(), events.plus.cuda()
    xi = torch.from_numpy(np.load(XI_Z_BLOCK1)).cuda()
    state = prior(network, batch, batch_seed(DIRECTIONS_SEED, 0))
    baseline = generate(network, batch, state, STEPS).four_momenta

    steered = {}
    for alpha in (0.0, 1.0):
        steer = Steer(xi, alpha, *edited_tokens(batch, minus, plus, "muon", "pair"))
        with attach(network, 1, steer):
            steered[alpha] = generate(network, batch, state, STEPS, on_step=steer.set_step).four_momenta
    assert torch.equal(steered[0.0], baseline)
    assert steered[1.0].isfinite().all()
    shift = pair_mass(steered[1.0], minus, plus).log() - pair_mass(baseline, minus, plus).log()
    assert (shift != 0).float().mean() > 0.9, "alpha = 1 leaves most masses unchanged"
