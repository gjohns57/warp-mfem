#!/bin/bash
# Refinement re-tune: correspondence (material-point) objective, scored against the
# fine-mesh reference run instead of the tracked surface (no tool exclusion needed).
cd /home/gabriel/auras/warp-mfem
COMMON="--strategy coordinate --frames 150 --repeats 3 --objective corr_mse --reference-recording results/reference_runs/fine_default.npz"
uv run python -u -m examples.refinement.sweep_octopus --space results/sweep_refine_corr_space.json \
    --sim-arg=--refine-scoring --sim-arg=geometric $COMMON \
    --out results/sweep_refine_corr_fine_geometric.jsonl > results/sweep_refine_corr_fine_geometric.log 2>&1
uv run python -u -m examples.refinement.sweep_octopus --space refine-legacy $COMMON \
    --out results/sweep_refine_corr_fine_legacy.jsonl > results/sweep_refine_corr_fine_legacy.log 2>&1
echo "[queue] both sweeps finished" >> results/sweep_refine_corr_fine_legacy.log
