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
    methods_z.yaml       App. B: LDA and difference of means
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
  steering_vectors.py  fkt, lda, meandiff; per step or constant in time; saved with their metadata
  metrics.py           R_60 and the selectivity curve over alpha
  plots.py             spectrum with ratio panel, selectivity curve
tests/
  test_steering_vectors.py  FKT on a toy problem with a known answer; save and load round trip
  test_hooks.py        alpha = 0 reproduces the baseline bit for bit
```

## Outputs

A run writes to `runs/<name>/`, which git ignores:

```
runs/<name>/
  config.yaml          the resolved config of the run
  events_directions.npz
  events_steering.npz
  gradient_moments.npz
  steering_vectors.npz steering vectors and their metadata
  generated/
    alpha_+1.0/        one folder per alpha; alpha_0.0 is the baseline
  plots/
```

## Rules

- The package installs with `uv sync`. No file changes `sys.path`.
- No sleep loops, timeouts, or silent fallbacks. A missing file or a wrong shape stops the run with an error.
- Every setting lives in the config. No environment variables change results.
