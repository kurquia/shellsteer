#!/usr/bin/env bash
# The paper's mean ablation (Sec. 5.3, Eqs. 24 and 25, Fig. 7): the Z steering vector at block 1, ablated on the
# steering sample for seeds 2041, 2042, and 2043, with xi_Z and c_k from the directions sample. Outputs go to
# runs/ablation_z/.
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 1,040,000 events per sample, 200 Euler steps, batch 8192, float32.
# The GPU times are for one B200. `steer` uses every visible GPU (--devices chooses them); with n GPUs the GPU times
# should shrink about n-fold, which is not measured yet.
#
# Without the gradient pass: add --set steering_vectors.from_file=vectors/main_block1.npz to every command
# (README, "Run the full chain").
# `steer gradients` then makes only the baseline of the directions sample and its kept pairs, which c_k needs: about
# 14 min instead of about 1 h on one B200. `steer steering-vectors` copies the file after a check against the config.
set -euo pipefail
CONFIG=configs/paper/ablation_z.yaml

# Gradient pass on the directions sample, as in the main run (Sec. 4.1).
# GPU; about 1-2 h on one B200; writes gradient_moments.npz, about 2.5 GB, and directions_kept.npz.
steer gradients "$CONFIG"

# Steering vectors of every class and block (Sec. 4.2).
# CPU; about a minute; writes steering_vectors.npz, about 10 MB.
steer steering-vectors "$CONFIG"

# One forward pass without edits over the directions sample, seed 2040: the mean residual stream of the kept pairs at
# block 1, the input of c_k (Eq. 25). GPU; about 10-20 min on one B200; writes activation_means.npz, about 2 MB.
steer activation-means "$CONFIG"

# Baseline and mean-ablated generation (Eq. 24) of the steering sample for each of the three seeds; kept pairs from
# the baseline of seed 2041 (ablated/kept.npz).
# GPU; 6.2 million pairs, about 0.5-1.5 h on one B200; writes ablated/, about 50 MB per file, 300 MB in all.
steer ablate "$CONFIG"

# Spread and height of the Z peak and the deviation below 60 GeV, per seed and over seeds (Sec. 5.3).
# CPU; about a minute; writes ablation_readout.json, a few kB.
steer ablation-readout "$CONFIG"

# Fig. 7: the spectrum of the baseline, the mean ablation, and real data, an inset of the Z peak, and the ratio.
# CPU; seconds; writes plots/ablation.pdf.
steer ablation-plots "$CONFIG"
