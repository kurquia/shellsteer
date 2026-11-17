#!/usr/bin/env bash
# The paper's main run (Sec. 5: Z, block 1, FKT), one stage per command. Outputs go to runs/main_z_block1/.
# It makes Fig. 3 (seed 2041, alpha -2, -1, +1, +2; the real-data line is not drawn yet). It does not reproduce
# Fig. 5: that figure needs alpha from -3 to 3 in steps of 0.5 and three seeds, which this config does not hold.
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 1,040,000 events per sample, 200 Euler steps, batch 8192, float32.
set -euo pipefail
CONFIG=configs/paper/main_z_block1.yaml

# Gradient pass on the directions sample: baseline generation, kept pairs, then gradients of log m at blocks 0-5
# (Sec. 4.1). One CUDA GPU (B200 class, tens of GB of memory); about 1-2 h; writes gradient_moments.npz, about
# 2.5 GB, and directions_kept.npz, about 40 MB.
steer gradients "$CONFIG"

# Steering vectors of every class and block from the per-class sums (Sec. 4.2).
# CPU; about a minute; writes steering_vectors.npz, about 10 MB.
steer steering-vectors "$CONFIG"

# Baseline and steered generation of the steering sample: seed 2041, alpha 0, -2, -1, +1, +2 (Sec. 5, Eq. 8).
# One CUDA GPU; 5.2 million pairs, about 0.5-1 h; writes generated/, about 50 MB per seed and alpha, 290 MB in all.
steer generate "$CONFIG"

# R_60, target shift, selectivity curve, response fits, and the Z lineshape fits (Sec. 5, App. E).
# CPU; about a minute; writes readout.json, a few kB.
steer readout "$CONFIG"

# Steered spectrum (Fig. 3), selectivity curve, and pole and width at the five alphas of this config.
# CPU; under a minute; writes plots/, three PDFs.
steer plots "$CONFIG"
