#!/usr/bin/env bash
# The paper's mean ablation (Sec. 5.3, Eqs. 24 and 25, Fig. 7): the Z steering vector at block 1, ablated on the
# directions sample for seeds 2040, 2041, and 2042. Outputs go to runs/ablation_z/.
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 1,040,000 events per sample, 200 Euler steps, batch 8192, float32.
set -euo pipefail
CONFIG=configs/paper/ablation_z.yaml

# Gradient pass on the directions sample, as in the main run (Sec. 4.1).
# One CUDA GPU (B200 class); about 1-2 h; writes gradient_moments.npz, about 2.5 GB, and directions_kept.npz.
steer gradients "$CONFIG"

# Steering vectors of every class and block (Sec. 4.2).
# CPU; about a minute; writes steering_vectors.npz, about 10 MB.
steer steering-vectors "$CONFIG"

# One forward pass without edits over the directions sample, seed 2040: the mean residual stream of the kept pairs at
# block 1, the input of c_k (Eq. 25). One CUDA GPU; about 10-20 min; writes activation_means.npz, about 2 MB.
steer activation-means "$CONFIG"

# Baseline and mean-ablated generation (Eq. 24) of the directions sample for each of the three seeds.
# One CUDA GPU; 6.2 million pairs, about 0.5-1.5 h; writes ablated/, about 50 MB per file, 300 MB in all.
steer ablate "$CONFIG"

# Spread and height of the Z peak and the deviation below 60 GeV, per seed and over seeds (Sec. 5.3).
# CPU; about a minute; writes ablation_readout.json, a few kB.
steer ablation-readout "$CONFIG"

# Fig. 7: the spectrum of the baseline, the mean ablation, and real data, an inset of the Z peak, and the ratio.
# CPU; seconds; writes plots/ablation.pdf.
steer ablation-plots "$CONFIG"
