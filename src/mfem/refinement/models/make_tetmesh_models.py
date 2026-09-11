"""Convert Gmsh tetmesh files from the workspace ``models/`` folder into the
``.npz`` format consumed by :class:`MFEMRefinementModel`.

Mirrors ``make_fem_beam_example.py`` but sources geometry from a ``.msh`` file
(read with ``meshio``) instead of a generated soft grid.

Examples
--------
Convert every ``.msh`` in the workspace ``models/`` folder, writing
``<stem>.npz`` next to each one::

    python -m mfem.refinement.models.make_tetmesh_models

Convert a single mesh with a custom output path and material params::

    python -m mfem.refinement.models.make_tetmesh_models \
        models/octopus_frame0_coarse.msh -o octopus_coarse.npz \
        --k-mu 6.0 --k-lambda 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import meshio
import newton
import numpy as np
import warp as wp

from mfem.refinement.models import MFEMRefinementModel

# ``src/mfem/refinement/models/make_tetmesh_models.py`` -> repo root is 4 up.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = WORKSPACE_ROOT / "models"

# Over-allocation ceilings for refinement growth. These are baked into the .npz
# and become the *padded* system size the RefinementSolver allocates for -- the
# CG solve runs over max_particles x max_particles blocks and the global-LHS row
# capacity scales with max_tets, so loose ceilings directly cost solve time.
#
# A flat multiple is the wrong shape: refinement concentrates in the tool-contact
# region, so it adds a roughly *fixed* count of verts/tets regardless of the base
# resolution -- a 3x/4x multiple that just fits the coarse mesh is wildly loose on
# medium/full. So the ceiling is ``base * MULT + SLACK``, clamped to the old flat
# multiple so a ceiling never grows relative to the previous convention.
#
# Tuned against the measured worst-case of the production geometric-refinement
# config (--refine-scoring geometric --refine-geometric-threshold 2 --mesh coarse,
# 150-frame episode): ~1800 verts / ~7600 tets / ~1470 surface tris from a
# 600 v / 2025 t / 880 tri base. The coarse ceilings below still clamp to the old
# 3x / 4x (i.e. unchanged); medium and full tighten ~20-60%.
PARTICLE_MULT, PARTICLE_SLACK, PARTICLE_MAX_MULT = 1.15, 1250, 3
TET_MULT, TET_SLACK, TET_MAX_MULT = 1.20, 5800, 4
TRI_MULT, TRI_SLACK, TRI_MAX_MULT = 2.20, 500, 4


def _headroom(base: int, mult: float, slack: int, max_mult: int) -> int:
    """``base * mult + slack``, clamped so it never exceeds the old flat
    ``base * max_mult`` convention."""
    return int(min(base * max_mult, -(-int(base * mult)) + slack))


def read_tetmesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, tets)`` from a Gmsh file as float32 / int32 arrays."""
    mesh = meshio.read(path)
    points = np.ascontiguousarray(mesh.points, dtype=np.float32)

    tet_blocks = [cb.data for cb in mesh.cells if cb.type == "tetra"]
    if not tet_blocks:
        raise ValueError(f"{path}: no 'tetra' cells found (cell types: "
                         f"{[cb.type for cb in mesh.cells]})")
    tets = np.ascontiguousarray(np.concatenate(tet_blocks, axis=0), dtype=np.int32)
    return points, tets


def build_refinement_model(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    density: float,
) -> MFEMRefinementModel:
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=points,
        indices=tets.flatten(),
        density=density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
        add_surface_mesh_edges=False,
        validate_mesh=False,
    )
    model = builder.finalize()

    return MFEMRefinementModel.from_model(
        model,
        max_particles=_headroom(model.particle_count, PARTICLE_MULT, PARTICLE_SLACK, PARTICLE_MAX_MULT),
        max_tets=_headroom(model.tet_count, TET_MULT, TET_SLACK, TET_MAX_MULT),
        max_tris=_headroom(model.tri_indices.shape[0], TRI_MULT, TRI_SLACK, TRI_MAX_MULT),
    )


def convert(
    src: Path,
    dst: Path,
    *,
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    density: float,
) -> None:
    points, tets = read_tetmesh(src)
    refinement_model = build_refinement_model(
        points,
        tets,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
        density=density,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    refinement_model.save(str(dst))
    print(f"{src.name}: {points.shape[0]} verts, {tets.shape[0]} tets "
          f"-> {dst} (max_particles={refinement_model.max_particles}, "
          f"max_tets={refinement_model.max_tets}, "
          f"max_tris={refinement_model.max_tris})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "meshes",
        nargs="*",
        type=Path,
        help="Gmsh .msh files to convert (default: every .msh in the workspace models/ folder).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .npz path (only valid with a single input mesh; "
             "default: <mesh>.npz next to each input).",
    )
    parser.add_argument("--k-mu", type=float, default=6.0, help="Lame mu (default: 6.0).")
    parser.add_argument("--k-lambda", type=float, default=1.0, help="Lame lambda (default: 1.0).")
    parser.add_argument("--k-damp", type=float, default=0.0, help="Damping coefficient (default: 0.0).")
    parser.add_argument("--density", type=float, default=1.0, help="Mass density (default: 1.0).")
    args = parser.parse_args()

    meshes = list(args.meshes) or sorted(MODELS_DIR.glob("*.msh"))
    if not meshes:
        parser.error(f"no meshes given and none found in {MODELS_DIR}")
    if args.output is not None and len(meshes) != 1:
        parser.error("--output requires exactly one input mesh")

    wp.init()
    for src in meshes:
        src = src.resolve()
        dst = args.output if args.output is not None else src.with_suffix(".npz")
        convert(
            src,
            dst,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            density=args.density,
        )


if __name__ == "__main__":
    main()
