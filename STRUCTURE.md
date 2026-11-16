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
examples/
  run_paper_main.sh    one command per stage
src/<package>/
  cli.py               steer <stage> CONFIG [--set key=value]
  config.py            reads the YAML into dataclasses; unknown keys are an error
  networks.py          loads a network by name from networks.yaml
  sampler.py           Euler sampler with seeded noise
  hooks.py             probe for the gradients; steer and ablate edits as `with` blocks
  events.py            the events of each sample, the slots of the pair, the edited tokens, the classes
  gradients.py         gradient pass on the directions sample; per-class sums of the unit gradients
  steering_vectors.py  fkt, fisher, meandiff; per step or constant in time; saved with their metadata
  steered_generation.py  the edit at a block; generation of the pairs, steered or baseline; files per seed and alpha
  lineshape.py         the paper's fit of the Z lineshape: pole M and width Gamma of a mass spectrum
  luminosity.py        writes the luminosity table from the NNPDF4.0 grid file: python -m shellsteer.luminosity
  metrics.py           R_60, the shift of the target, and the selectivity curve over alpha
  plots.py             steered spectrum with ratio panel and zooms, selectivity curve, pole and width against alpha
  paper.mplstyle       the paper's figure style, applied only inside plots.py
  data/
    luminosity_nnpdf40_13tev.csv  quark luminosities with the muon acceptance, written by luminosity.py
tests/
  test_config.py       the paper config loads; each bad value stops with its message
  test_networks.py     the network loads from Hugging Face, frozen
  test_sampler.py      the sampler matches the paper's sampler bit for bit on CPU
  test_hooks.py        alpha = 0 reproduces the baseline bit for bit; probe and steer match the paper's code
  test_events.py       pair slots, edited tokens, loading, classes, and selection of the pairs
  test_gradients.py    the per-class sums equal the sums of the per-pair gradients
  test_steering_vectors.py  FKT on a toy problem with a known answer; save and load round trip
  test_steered_generation.py  alpha = 0 is the baseline; the batches equal a loop by hand; save and load
  test_lineshape.py    the paper's baseline histogram refits to the paper's pole and width
  test_luminosity.py   the grid reader and the integral on a grid with a known answer; the csv round trip
  test_metrics.py      R_60 and the target shift on spectra with known answers
  test_plots.py        each figure draws its panels and lines; bad inputs stop
  test_gpu.py          the paper's baseline and sums on a CUDA GPU; skipped without one
```

## Outputs

A run writes to `runs/<name>/`, which git ignores:

```
runs/<name>/
  config.yaml          the resolved config of the run
  gradient_moments.npz per-class sums of the unit gradients, with their metadata
  steering_vectors.npz steering vectors and their metadata
  generated/
    kept.npz           the class of each kept pair of the steering sample, chosen on the first seed's baseline
    seed_<seed>/
      alpha_+0.npz     the baseline: mass and four-momenta of each pair, with the metadata
      alpha_-1.npz     one file per alpha, named by alpha_<alpha:+g>
  plots/
```

The events load from the Hugging Face cache on every run, so the run folder holds no copy of them.

## Rules

- The package installs with `uv sync`. No file changes `sys.path`.
- No sleep loops, timeouts, or silent fallbacks. A missing file or a wrong shape stops the run with an error.
- Every setting lives in the config. No environment variables change results.
