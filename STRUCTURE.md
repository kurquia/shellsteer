# Repository structure

This is the planned layout. Files are added one by one, so some paths here do not exist yet.

```
README.md              what the code does, install, quickstart, paper settings and their config keys
STRUCTURE.md           this file
CITATION.cff
LICENSE
pyproject.toml         direct dependencies only, plus uv.lock
configs/
  classes.yaml         mass windows of the resonance classes
  networks.yaml        ShellFlow, ShellFlow-B, ShellGATr: checkpoint and residual width
  datasets.yaml        event samples on Hugging Face: the directions and the steering sample
  paper/
    main_z_block1.yaml   Sec. 5: Z, block 1, FKT
    ablation_z.yaml      Sec. 5.3: ablation of the steering vector
    methods_z.yaml       App. B: Fisher and difference of means
    networks_z.yaml      App. C: ShellFlow-B and ShellGATr
    blocks_z.yaml        App. D: blocks 0 to 5
    quarkonia.yaml       App. D: J/psi and Upsilon
    all_muons.yaml       Fig. 3: every muon, all final states
vectors/
  main_block1.npz      the steering vectors of the main run at block 1 (Sec. 5), for steering_vectors.from_file
examples/
  run_paper_main.sh    one command per stage of the main run (Sec. 5); the runs of Figs. 3 and 5 from vectors/
  run_paper_ablation.sh  one command per stage of the mean ablation (Sec. 5.3)
src/<package>/
  cli.py               steer <stage> CONFIG [--set key=value]...; steer all CONFIG runs the chain of the config
  config.py            reads the YAML into dataclasses; unknown keys are an error; steering or ablation sections
  networks.py          loads a network by name from networks.yaml
  sampler.py           Euler sampler with seeded noise; counts the NaNs of each event, sets them to 0 or stops
  hooks.py             probe for the gradients; steer and ablate edits as `with` blocks
  events.py            the events of each sample with their real-data mass, the slots of the pair, the edited tokens,
                       the classes
  gradients.py         gradient pass on the directions sample; per-class sums of the unit gradients
  activation_means.py  mean residual stream of the pair's tokens by class, and c_k of the mean ablation (Eq. 25)
  steering_vectors.py  fkt, fisher, meandiff; per step or constant in time; saved with S (Eq. 21) and their metadata;
                       the check of a vectors file against a config
  steered_generation.py  the edit at a block; generation of the pairs, steered or baseline; files per seed and alpha
  lineshape.py         the paper's fit of the Z lineshape: pole M and width Gamma of a mass spectrum
  luminosity.py        writes the luminosity table from the NNPDF4.0 grid file: python -m shellsteer.luminosity
  metrics.py           R_60, the shift of the target, the selectivity curve, and the numbers of the mean ablation
  plots.py             steered spectrum with ratio panel and zooms, selectivity curve, pole and width against alpha,
                       mean ablation with inset and ratio panel (Fig. 7); real data in both spectra
  paper.mplstyle       the paper's figure style, applied only inside plots.py
  data/
    luminosity_nnpdf40_13tev.csv  quark luminosities with the muon acceptance, written by luminosity.py
tests/
  test_config.py       the paper config loads; each bad value stops with its message
  test_networks.py     the network loads from Hugging Face, frozen
  test_sampler.py      the sampler matches the paper's sampler bit for bit on CPU; NaNs counted, set to 0, or stop
  test_hooks.py        alpha = 0 reproduces the baseline bit for bit; probe and steer match the paper's code
  test_events.py       pair slots, edited tokens, loading, classes, and selection of the pairs
  test_gradients.py    the per-class sums equal the sums of the per-pair gradients
  test_steering_vectors.py  FKT on a toy problem with a known answer; save and load round trip
  test_steered_generation.py  alpha = 0 is the baseline; the batches equal a loop by hand; save and load
  test_lineshape.py    the paper's baseline histogram refits to the paper's pole and width
  test_luminosity.py   the grid reader and the integral on a grid with a known answer; the csv round trip
  test_metrics.py      R_60, the target shift, and the ablation numbers on spectra with known answers
  test_ablation.py     the Ablate and Record hooks, the activation means, and the ablated generation
  test_cli.py          both runs through `steer` on CPU at a small size; overrides, stage dependencies, refusals;
                       the real-data line of both spectra
  test_plots.py        each figure draws its panels and lines; bad inputs stop
  test_vectors_from_file.py  a run from a vectors file generates the same masses without the gradient pass; S from
                       the file or reported unavailable; each metadata mismatch stops; the shipped file fits the
                       main config
  test_gpu.py          the paper's baseline and sums on a CUDA GPU; skipped without one
```

## Outputs

A run writes to `runs/<name>/`, which git ignores. A config runs one of two chains of stages (`cli.py`):

    steering run:  gradients, steering-vectors, generate, readout, plots
    mean ablation: gradients, steering-vectors, activation-means, ablate, ablation-readout, ablation-plots
                   (config with `ablation`)

With `steering_vectors.from_file`, a run starts from the `steering_vectors.npz` of an earlier run. `steer
steering-vectors` checks the file against the config (method, in_time, balance, classes, the edited block, Euler
steps, token width, and the network and class windows if the file records them) and copies it into the run folder.
The steering run then has no gradients stage. In the mean ablation, `steer gradients` makes only
`directions_kept.npz`, the kept pairs that c_k needs, and no `gradient_moments.npz`. The readout takes S (Eq. 21)
from the vectors file; for a file that stores no S, `readout.json` holds `null` with the reason, and Fig. 5 has no
first-order line.

```
runs/<name>/
  config.yaml          the resolved config of the run, overrides applied
  gradient_moments.npz per-class sums of the unit gradients, with their metadata         (gradients)
  directions_kept.npz  the class of each kept pair of the directions sample               (gradients)
  steering_vectors.npz steering vectors, S of each class and block, and their metadata    (steering-vectors)
  generated/                                                                              (generate)
    kept.npz           the class of each kept pair of the steering sample, chosen on the first seed's baseline
    seed_<seed>/
      alpha_+0.npz     the baseline: mass and four-momenta of each pair, with the metadata
      alpha_-1.npz     one file per alpha, named by alpha_<alpha:+g>
  readout.json         R_60, target shift, selectivity curve, response fits, lineshape fits  (readout)
  plots/               steered spectrum with real data, selectivity curve, pole and width   (plots)
  activation_means.npz per-class sums of the residual stream for c_k                       (activation-means)
  ablated/                                                                                (ablate)
    kept.npz           only when ablation.sample is steering: kept pairs on the first seed's baseline
    seed_<seed>/
      baseline.npz     the baseline of the ablated sample
      block_<L>.npz    the mean ablation at block L
  ablation_readout.json  spread and height of the target peak, deviation below 60 GeV       (ablation-readout)
  plots/
    ablation.pdf       Fig. 7: baseline, mean ablation, and real data, counts averaged over seeds  (ablation-plots)
```

The events load from the Hugging Face cache on every run, so the run folder holds no copy of them. The plots stages
read the real-data masses (`m_real`) of the kept pairs from the same files.

Every stage that generates events counts the NaNs of the final state, of the four-momenta, and of the pair masses
(`sampler.on_nan` in the config: set them to 0 as the paper, or stop). The counts appear in a `NaN:` log line and
under "NaN counts" in the metadata of each file the stage writes.

## Shipped vectors

`vectors/main_block1.npz` is `runs/full_main/steering_vectors.npz` of the full main run on one B200, unchanged (md5
6283fc6f789a2b8a5d9ce47f9e3c2684), written 2026-11-17 02:08 CET. The command that wrote it, from the repository root:

    steer all configs/paper/main_z_block1.yaml --set name=full_main --set 'gradients.blocks=[1]'

Its steering-vectors stage wrote the file from a gradient pass of 1,040,000 events of the directions sample, seed
2040, 1,000,000 kept pairs, block 1 only. The file holds the FKT vectors (per step, balance equal) of the classes Z,
Jpsi, Upsilon, and rest at block 1: 200 Euler steps, 2 muon tokens of width 128, float64. The code wrote it before the
vectors file stored S, the network, and the class windows, so its metadata records only the method, in_time, and
balance. A run from it reports S as unavailable and checks neither the network nor the class windows. Three other runs
of the same night (the ablation and the two Fig. 5 runs) wrote byte-identical files.

## Rules

- The package installs with `uv sync`. No file changes `sys.path`.
- No sleep loops, timeouts, or silent fallbacks. A missing file or a wrong shape stops the run with an error.
- Every setting lives in the config. No environment variables change results.
