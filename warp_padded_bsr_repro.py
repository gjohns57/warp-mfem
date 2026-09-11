"""
Reproduction of a deterministic illegal memory access / "Failed to load CUDA module"
crash inside warp.sparse's "padded"-topology bsr_mm, triggered by problem *size* alone
via mfem.refinement.solver.RefinementSolver.

Observed on:
  - warp-lang 1.15.0 and 1.16.0 (both fail identically, confirmed via `uv run --with`)
  - NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, sm_89), CUDA Toolkit 12.9, Driver 13.2

Reliable boundary on this machine as of this writing, with max_particles/max_tets
held FIXED at 2000/10000 for every size tested (so this isn't an exact-fit-vs-padded
difference, just active mesh size against identical headroom): a 8x8x8 soft grid
(729 particles, 2560 tets) crashes every time (4/4 runs); a 7x7x7 grid (512
particles, 1715 tets) never does (4/4 runs). Making max_particles/max_tets larger
still (e.g. 2-3x the active counts) reliably avoids the crash in every size tested,
which is the current workaround - i.e. the failure is not simply "not enough
capacity," since both sizes above are well within the configured 2000/10000 headroom.

IMPORTANT: this boundary is NOT a stable property of matrix size. It measurably
shifted (from a 9-vs-8 boundary down to an 8-vs-7 boundary) between two points in the
same debugging session, with no change to the code path actually exercised for this
zero-attachment test case - only unrelated solver code was added elsewhere in the
file. Treat the specific dim numbers here as "this reproduces the bug today, on this
machine," not as a fixed threshold - re-run with a few sizes to find the current
boundary if it stops reproducing.

Ruled out via `wp.config.verify_cuda` + manual instrumentation:
  - GPU memory pressure (6+ GiB free at the point of failure)
  - BSR row/nnz capacity (well within configured row_capacity/nnz_capacity)
  - bsr_mm's tile_size heuristic (still fails with tile_size=-1 forced)
  - wp.config.enable_mempools_at_init (still fails with mempools disabled)
  - kernel compile-vs-cache timing (still fails with all kernels pre-cached on disk)
  - A minimal standalone script that only performs the same-shaped
    bsr_set_from_triplets + bsr_mm calls (no surrounding solver state) does NOT
    reproduce the crash - it requires the accumulated kernel-launch history from the
    full solver pipeline, not just matching matrix sizes/capacities in isolation.

`wp.config.verify_cuda=True` pins the corruption to first-use of a lazily-allocated
internal buffer inside a fresh "padded" BsrMatrix (row_counts, the internal overflow
"status" scalar, ...) when that first touch happens from inside
bsr_mm/bsr_set_from_triplets (_bsr_ensure_independent_row_counts / _ensure_status in
warp/_src/sparse.py). Pre-touching one such buffer before the real call shifts the
crash to the next lazily-allocated buffer downstream rather than fixing it.

Usage:
    uv run python warp_padded_bsr_repro.py            # 7 (passes) then 8 (fails)
    uv run python warp_padded_bsr_repro.py 7
    uv run python warp_padded_bsr_repro.py 8
"""

import sys

import warp as wp
import newton

from mfem.refinement.sim import MFEMRefinementModel, load_sim_model
from mfem.refinement.solver import RefinementSolver


def run(dim: int) -> None:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=0.2,
        cell_y=0.2,
        cell_z=0.2,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
        density=1.0,
    )
    model = builder.finalize()
    print(f"dim={dim}: particles={model.particle_count}, tets={model.tet_count}")

    # Fixed padding, identical for every dim tested: this is what makes the boundary
    # deterministic - a size-dependent failure with the *same* max_particles/max_tets
    # headroom, not an exact-fit-vs-padded difference.
    refinement_model = MFEMRefinementModel.from_model(model, max_particles=2000, max_tets=10000)
    builder2 = newton.ModelBuilder(gravity=0.0)
    model2 = load_sim_model(refinement_model, builder2)

    solver = RefinementSolver(model=model2, iterations=8, max_tets=refinement_model.max_tets)

    state0 = model2.state()
    state1 = model2.state()
    control = model2.control()
    contacts = model2.contacts()
    solver.step(state0, state1, control, contacts, 1.0 / 24.0)
    wp.synchronize()
    print(f"dim={dim}: OK")


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [7, 8]
    for d in dims:
        run(d)
