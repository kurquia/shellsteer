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
  run_paper_main.sh    one command per stage of the main run (Sec. 5)
  run_paper_ablation.sh  one command per stage of the mean ablation (Sec. 5.3)
src/<package>/
  cli.py               steer <stage> CONFIG [--set key=value]... [--devices LIST]; steer all CONFIG runs the chain of
                       the config
  config.py            reads the YAML into dataclasses; unknown keys are an error; steering or ablation sections
  networks.py          loads a network by name from networks.yaml
  devices.py           the devices of --devices; one copy of the network per device; batch i on device i mod n,
                       the results in batch order
  sampler.py           Euler sampler with seeded noise; counts the NaNs of each event, sets them to 0 or stops
  hooks.py             probe for the gradients; steer and ablate edits as `with` blocks
  events.py            the events of each sample with their real-data mass, the slots of the pair, the edited tokens,
                       the classes
  gradients.py         gradient pass on the directions sample; per-class sums of the unit gradients
  activation_means.py  mean residual stream of the pair's tokens by class, and c_k of the mean ablation (Eq. 25)
  steering_vectors.py  fkt, fisher, meandiff; per step or constant in time; saved with their metadata
  steered_generation.py  the edit at a block; generation of the pairs, steered or baseline; files per seed and alpha
  lineshape.py         the paper's fit of the Z lineshape: pole M and width Gamma of a mass spectrum
  luminosity.py        writes the luminosity table from the NNPDF4.0 grid file: python -m shellsteer.luminosity
  metrics.py           R_60, the shift of the target, the selectivity curve, and the numbers of the mean ablation
  plots.py             steered spectrum with ratio panel and zooms, selectivity curve, pole and width against alpha,
                       mean ablation with inset and ratio panel (Fig. 7); real data in both spectra
  paper.mplstyle       the paper's figure style, applied only inside plots.py
  data/
    luminosity_nnpdf40_13tev.csv  quark luminosities with the muon acceptance, written by luminosity.py
```

## Outputs

A run writes to `runs/<name>/`, which git ignores. A config runs one of two chains of stages (`cli.py`):

    steering run:  gradients, steering-vectors, generate, readout, plots
    mean ablation: gradients, steering-vectors, activation-means, ablate, ablation-readout, ablation-plots
                   (config with `ablation`)

```
runs/<name>/
  config.yaml          the resolved config of the run, overrides applied
  gradient_moments.npz per-class sums of the unit gradients, with their metadata         (gradients)
  directions_kept.npz  the class of each kept pair of the directions sample               (gradients)
  steering_vectors.npz steering vectors and their metadata                                (steering-vectors)
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

The stages that run the network (gradients, activation-means, generate, ablate) use every visible GPU, or the CPU
if there is none; `--devices cuda:0,cuda:1` chooses them, and `--devices cpu,cpu` runs two CPU workers. Batch i runs
on device i mod n with the noise seed of batch i, and the stage adds the results in batch order. On GPUs of one type
(name and number of streaming multiprocessors) every file is bit-identical to that of a run on one of them. The log
names the devices and warns if their types differ. The "device" entry in the metadata of a file is the device type,
not the number of devices.

## Rules

- The package installs with `uv sync`. No file changes `sys.path`.
- No sleep loops, timeouts, or silent fallbacks. A missing file or a wrong shape stops the run with an error.
- Every setting lives in the config. No environment variables change results.
