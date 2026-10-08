#!/bin/bash
# Solver timing: CUDA graph capture on vs off. Octopus, coarse mesh, no refinement, 12 Newton
# iterations, block-Jacobi PCG + line search, neo-Hookean, headless, 150 frames.
# Without -g the CG solve also runs without its own CUDA graph (cg_use_cuda_graph follows -g) and
# --cg-check-every resolves to 10 (host-side convergence checks); with -g it is 0 (in-graph exit).
cd /home/gabriel/auras/warp-mfem
run() { name=$1; shift; echo "[run] $name: $*"; uv run python -u -m examples.refinement.octopus_refinement --episode octopus --quiet --headless -p -l --energy neohookean --mesh coarse --record-frames 150 --record results/timing_runs_graph/$name.npz "$@" > results/timing_runs_graph/$name.log 2>&1; echo "[run] $name exit=$? steps=$(grep -c 'Step and readback took' results/timing_runs_graph/$name.log)"; }
run gpu_graph   -g
run gpu_nograph
echo "[run] ALL DONE"
