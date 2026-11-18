#!/usr/bin/env bash
# Paper App. C: the Z steering vector at block 1 of ShellFlow-B and ShellGATr (Figs. 11 to 13, Table 4), one command
# per stage. Outputs go to runs/networks_z_shellflow_b/, runs/networks_z_shellgatr/, and runs/networks_z/.
#
# The two networks read the 2to4lep events of datasets.yaml entry validation_muon_pairs_2to4lep.
# ShellGATr needs the gatr extra: uv sync --extra gatr.
#
# Times and disk sizes are estimates for one NVIDIA B200, not measurements of this script: 1,040,000 events per
# sample, 200 Euler steps, batch 8192, float32. The shellsteer runs of ShellFlow in runs/ (full_main.log) took
# 14 min per generation and 61 min for the gradient pass at block 1. The paper's own runs of the three networks on
# the same B200s took 0.7 (ShellFlow-B) and 1.0 (ShellGATr) times as long per generation as ShellFlow, and 1.3 times
# as long for the gradient pass of ShellGATr as for ShellFlow-B. So: about 10 min per generation for ShellFlow-B,
# 14 min for ShellGATr, and 1-2 h for each gradient pass at block 1.
#
# Figs. 11 and 12 show seed 2041 at four alphas; the steered spectrum of these runs draws all 12. The two commands at
# the end make the figures with the paper's alphas from the vectors of these runs (5 generations each).
set -euo pipefail

for NAME in networks_z_shellflow_b networks_z_shellgatr; do
    CONFIG=configs/paper/$NAME.yaml

    # Gradient pass on the directions sample: baseline generation, kept pairs, gradients of log m at block 1.
    # GPU; about 1-2 h; writes gradient_moments.npz (about 0.4 GB for ShellFlow-B, 1.3 GB for ShellGATr, whose tokens
    # are 224 wide) and directions_kept.npz, about 40 MB.
    steer gradients "$CONFIG"

    # Steering vectors of every class and block, with S, cos(G, xi), and lambda per Euler step (Sec. 4.2).
    # CPU; a few minutes; writes steering_vectors.npz, about 10 MB (ShellGATr about 17 MB).
    steer steering-vectors "$CONFIG"

    # Baseline and steered generation of the steering sample: 3 seeds x (alpha 0 and 12 alphas) = 39 generations.
    # GPU; about 6.5 h for ShellFlow-B and 9 h for ShellGATr; writes generated/, about 50 MB per seed and alpha, 2 GB.
    steer generate "$CONFIG"

    # R_60, target shift, response fits, the Z lineshape fits, and the means over the steps of cos(G, xi) and lambda.
    # CPU; a few minutes; writes readout.json.
    steer readout "$CONFIG"

    # Steered spectrum of seed 2041 at all 12 alphas, selectivity curve, and pole and width of this network alone.
    # CPU; under a minute; writes plots/.
    steer plots "$CONFIG"
done

# Fig. 13 and Table 4: the three networks side by side. ShellFlow is the Fig. 5 run of README, "Change settings
# with --set" (runs/main_z_block1_fig5). CPU; seconds; writes runs/networks_z/pole_and_width.pdf and table.json,
# and prints the table.
python -m shellsteer.compare_runs runs/main_z_block1_fig5 runs/networks_z_shellflow_b runs/networks_z_shellgatr \
    --out runs/networks_z

# Figs. 11 and 12 with the paper's alphas, from the vectors of the runs above: 5 generations each, about 50 min
# (ShellFlow-B) and 70 min (ShellGATr).
steer all configs/paper/networks_z_shellflow_b.yaml \
    --set steering_vectors.from_file=runs/networks_z_shellflow_b/steering_vectors.npz \
    --set 'steering.alphas=[-2, -1, 1, 2]' --set 'steering.seeds=[2041]' --set name=networks_z_shellflow_b_fig11
steer all configs/paper/networks_z_shellgatr.yaml \
    --set steering_vectors.from_file=runs/networks_z_shellgatr/steering_vectors.npz \
    --set 'steering.alphas=[-4, -2, 2, 4]' --set 'steering.seeds=[2041]' --set name=networks_z_shellgatr_fig12
