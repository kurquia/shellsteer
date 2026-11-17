#!/usr/bin/env bash
# Paper Figs. 4 and 14 (Sec. 5.1, App. D.1): the Z steering vector at block 1 on every muon of sample C, 1,000,000
# validation events of every final state, one stage per command. Outputs go to runs/all_muons/.
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 200 Euler steps, batch 8192, float32.
# The GPU times are for one B200. `steer` uses every visible GPU (--devices chooses them); with n GPUs the GPU times
# should shrink about n-fold, which is not measured yet.
#
# Without the gradient pass: vectors/main_block1.npz holds the Z vectors of main_z_block1.yaml at block 1, the vectors
# this run adds (STRUCTURE.md). This command then replaces this script, from the repository root, and has no
# gradients stage; the generation takes about 70 min on one B200:
#   steer all configs/paper/all_muons.yaml --set steering_vectors.from_file=vectors/main_block1.npz
set -euo pipefail
CONFIG=configs/paper/all_muons.yaml

# Gradient pass on the muon-pair directions sample, as in run_paper_main.sh (Sec. 4.1). GPU; on one B200 about 1-2 h
# for blocks 0-5, about 1 h with --set "gradients.blocks=[1]"; writes gradient_moments.npz and directions_kept.npz.
steer gradients "$CONFIG"

# Steering vectors of every class and block (Sec. 4.2). CPU; about a minute; writes steering_vectors.npz.
steer steering-vectors "$CONFIG"

# Baseline and steered generation of sample C: seed 2041, alpha 0, -2, -1, +1, +2, every muon edited. GPU; on one
# B200 about 14 min per alpha, 70 min in all; writes generated/, every particle of every event, about 200 MB per alpha.
steer generate "$CONFIG"

# Counts of p_T, eta, phi, and E of every particle type, and the largest change per bin (Sec. 5.1, App. D.1).
# CPU; a few minutes; writes kinematics.npz and kinematics_readout.json.
steer kinematics-readout "$CONFIG"

# Fig. 4 (muons) and Fig. 14 without its mass column (every other type). CPU; under a minute; writes plots/.
steer kinematics-plots "$CONFIG"
