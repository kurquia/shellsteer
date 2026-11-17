#!/usr/bin/env bash
# The paper's main run (Sec. 5: Z, block 1, FKT), one stage per command. Outputs go to runs/main_z_block1/.
# It makes Fig. 3 (seed 2041, alpha -2, -1, +1, +2, and real data). It does not reproduce Fig. 5: that figure needs
# alpha from -3 to 3 in steps of 0.5 and three seeds, which this config does not hold (see the end of this file).
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 1,040,000 events per sample, 200 Euler steps, batch 8192, float32.
#
# Without the gradient pass: vectors/main_block1.npz holds the vectors of the steering-vectors stage below (block 1
# only, STRUCTURE.md). Each command below then replaces this script, from the repository root; each generation of
# 1,040,000 events takes about 14 min on one B200.
#   Fig. 3, 5 generations, about 70 min:
#     steer all configs/paper/main_z_block1.yaml --set steering_vectors.from_file=vectors/main_block1.npz \
#       --set name=main_z_block1_from_vectors
#   Fig. 5, 13 alphas x 3 seeds = 39 generations, about 9 h; its steered spectrum draws all 12 alphas:
#     steer all configs/paper/main_z_block1.yaml --set steering_vectors.from_file=vectors/main_block1.npz \
#       --set 'steering.alphas=[-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]' \
#       --set 'steering.seeds=[2041, 2042, 2043]' --set name=main_z_block1_fig5_from_vectors
# The file stores the gradient response per Euler step; its sum is S (Eq. 21), the first-order line of Fig. 5.
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
