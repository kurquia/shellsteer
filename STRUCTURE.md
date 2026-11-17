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
  datasets.yaml        event samples on Hugging Face: the directions and the steering sample of muon pairs, and
                       the steering sample of every final state (sample C, no pairs)
  paper/
    main_z_block1.yaml   Sec. 5: Z, block 1, FKT
    ablation_z.yaml      Sec. 5.3: ablation of the steering vector
    methods_z.yaml       App. B: Fisher and difference of means
    networks_z.yaml      App. C: ShellFlow-B and ShellGATr
    blocks_z.yaml        App. D: blocks 0 to 5
    quarkonia.yaml       App. D: J/psi and Upsilon
    all_muons.yaml       Figs. 4 and 14: the Z vector on every muon of sample C, kinematics of every particle type
examples/
  run_paper_main.sh    one command per stage of the main run (Sec. 5)
  run_paper_ablation.sh  one command per stage of the mean ablation (Sec. 5.3)
  run_paper_all_muons.sh  one command per stage of the kinematics run (Figs. 4 and 14)
src/<package>/
  cli.py               steer <stage> CONFIG [--set key=value]...; steer all CONFIG runs the chain of the config
  config.py            reads the YAML into dataclasses; unknown keys are an error; steering or ablation sections
  networks.py          loads a network by name from networks.yaml
  sampler.py           Euler sampler with seeded noise; counts the NaNs of each event, sets them to 0 or stops
  hooks.py             probe for the gradients; steer and ablate edits as `with` blocks
  events.py            the events of each sample with their real-data mass, the slots of the pair, the edited tokens,
                       the classes; the events of every final state with their real four-momenta
  gradients.py         gradient pass on the directions sample; per-class sums of the unit gradients
  activation_means.py  mean residual stream of the pair's tokens by class, and c_k of the mean ablation (Eq. 25)
  steering_vectors.py  fkt, fisher, meandiff; per step or constant in time; saved with their metadata
  steered_generation.py  the edit at a block; generation of the pairs, steered or baseline, or of every particle of
                       a sample without pairs; files per seed and alpha
  kinematics.py        p_T, eta, phi, E of every particle type; the paper's bins; counts and the largest change per bin
  lineshape.py         the paper's fit of the Z lineshape: pole M and width Gamma of a mass spectrum
  luminosity.py        writes the luminosity table from the NNPDF4.0 grid file: python -m shellsteer.luminosity
  metrics.py           R_60, the shift of the target, the selectivity curve, and the numbers of the mean ablation
  plots.py             steered spectrum with ratio panel and zooms, selectivity curve, pole and width against alpha,
                       mean ablation with inset and ratio panel (Fig. 7); real data in both spectra; kinematics of
                       the edited muons (Fig. 4) and of every other particle type (Fig. 14)
  paper.mplstyle       the paper's figure style, applied only inside plots.py
  data/
    luminosity_nnpdf40_13tev.csv  quark luminosities with the muon acceptance, written by luminosity.py
```

## Outputs

A run writes to `runs/<name>/`, which git ignores. A config runs one of three chains of stages (`cli.py`):

    steering run:   gradients, steering-vectors, generate, readout, plots
    kinematics run: gradients, steering-vectors, generate, kinematics-readout, kinematics-plots
                    (config whose `events.steering_dataset` has no pairs, such as all_muons.yaml)
    mean ablation:  gradients, steering-vectors, activation-means, ablate, ablation-readout, ablation-plots
                    (config with `ablation`)

`events.dataset` names the dataset of the directions sample, and of the steering sample unless
`events.steering_dataset` names another one. The kinematics run takes its gradients and steering vectors from the muon
pairs and steers sample C, 1,000,000 events of every final state.

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
  kinematics.npz       kinematics run: counts per particle type and variable, real data,      (kinematics-readout)
                       baseline, each alpha; generated/ then holds every particle of each event, without kept.npz
  kinematics_readout.json  largest change per bin of each type, variable, and alpha          (kinematics-readout)
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
    kinematics.pdf     Fig. 4: p_T, eta, phi of the edited muons                         (kinematics-plots)
    all_particles.pdf  Fig. 14: p_T, eta, phi, E of every other particle type            (kinematics-plots)
```

The events load from the Hugging Face cache on every run, so the run folder holds no copy of them. The plots stages
read the real-data masses (`m_real`) of the kept pairs from the same files; kinematics-readout reads the real
four-momenta (`p4`) of sample C.

Paper Fig. 14 also has the mass of the jets, large-R jets, and hadronic taus; all_particles.pdf does not. The mass of a
hadronic tau (below 1 GeV) does not survive E^2 - p^2 of four-momenta in float32. That column needs the decoder's mass
of each generated particle (`sampler.decode`) and the real-data masses, which sample C does not hold.

Every stage that generates events counts the NaNs of the final state, of the four-momenta, and of the pair masses
(`sampler.on_nan` in the config: set them to 0 as the paper, or stop). The counts appear in a `NaN:` log line and
under "NaN counts" in the metadata of each file the stage writes.

## Rules

- The package installs with `uv sync`. No file changes `sys.path`.
- No sleep loops, timeouts, or silent fallbacks. A missing file or a wrong shape stops the run with an error.
- Every setting lives in the config. No environment variables change results.
