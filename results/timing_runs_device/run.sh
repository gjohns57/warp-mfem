#!/bin/bash
# Solver timing: GPU with / without CUDA graph capture, and the CPU device. Octopus, coarse mesh,
# no refinement, 12 Newton iterations, block-Jacobi PCG + line search, neo-Hookean, headless.
cd /home/gabriel/auras/warp-mfem
COMMON="--episode octopus --quiet --headless -p -l --energy neohookean --mesh coarse"
run() { name=$1; shift; echo "[run] $name: $*"; uv run python -u -m examples.refinement.octopus_refinement $COMMON --record results/timing_runs_device/$name.npz "$@" > results/timing_runs_device/$name.log 2>&1; echo "[run] $name exit=$? steps=$(grep -c 'Step and readback took' results/timing_runs_device/$name.log)"; }
run gpu_graph   --record-frames 150 -g
run gpu_nograph --record-frames 150
run cpu         --record-frames 6 --device cpu --cg-check-every 0   # CG runs all 5000 iterations per Newton step
echo "[run] ALL DONE"
# CPU with host-side CG convergence checks every 10 iterations. Since 2026-09-11 this is what
# --device cpu does by default (octopus_refinement resolves --cg-check-every to 0 on CUDA, 10 on the CPU);
# the "cpu" run above was made before that change and needs the explicit --cg-check-every 0 now.
# run cpu_check10 --record-frames 10 --device cpu --cg-check-every 10
