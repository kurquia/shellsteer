#!/usr/bin/env bash
# Paper Figs. 4 and 14 (Sec. 5.1, App. D.1): the Z steering vector at block 1 on every muon of sample C, 1,000,000
# validation events of every final state, one stage per command. Outputs go to runs/all_muons/.
# Times and disk sizes are estimates from the code and from earlier runs on one NVIDIA B200, not measurements of
# this script: 200 Euler steps, batch 8192, float32.
set -euo pipefail
CONFIG=configs/paper/all_muons.yaml

# Gradient pass on the muon-pair directions sample, as in run_paper_main.sh (Sec. 4.1). One CUDA GPU; about 1-2 h for
# blocks 0-5, about 1 h with --set "gradients.blocks=[1]"; writes gradient_moments.npz and directions_kept.npz.
steer gradients "$CONFIG"

# Steering vectors of every class and block (Sec. 4.2). CPU; about a minute; writes steering_vectors.npz.
steer steering-vectors "$CONFIG"

# Baseline and steered generation of sample C: seed 2041, alpha 0, -2, -1, +1, +2, every muon edited. One CUDA GPU;
# about 14 min per alpha, 70 min in all; writes generated/, every particle of every event, about 200 MB per alpha.
steer generate "$CONFIG"

# Counts of p_T, eta, phi, and E of every particle type, and the largest change per bin (Sec. 5.1, App. D.1).
# CPU; a few minutes; writes kinematics.npz and kinematics_readout.json.
steer kinematics-readout "$CONFIG"

# Fig. 4 (muons) and Fig. 14 without its mass column (every other type). CPU; under a minute; writes plots/.
steer kinematics-plots "$CONFIG"
