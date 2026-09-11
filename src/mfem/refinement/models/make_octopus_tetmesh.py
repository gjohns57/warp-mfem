"""Tetrahedralize one tracked frame of the PlushOctopus_T1 episode with fTetWild.

This is the same pipeline as the dataset's ``tetrahedralize_frame0.py`` (export a
watertight surface from the fused surface tracking, run the local fTetWild build,
convert the ``.msh`` to the ``.npz`` consumed by :class:`MFEMRefinementModel`),
but the source frame is selectable instead of hard-wired to frame 0.

The default (``--frame 19``) is the frame ``sim_octopus.py`` uses as the
stress-free rest state (``OCTO_INITIAL_FRAME``): the raw episode opens mid-poke
(frames 0-12), the tool lifts off at frame 13, and by frame 19 (frame_id 20) the
plush has finished springing back and the fused surface tracking is at its
cleanest / roundest, so that surface tetrahedralizes into the best initial state
for the refinement solver. ``--frame auto`` instead picks the **first frame with
the poking tool out of contact** (``tool_trajectory.npz`` ``contact_active``, a
>=1.5 N tool-force threshold) -- tracked frame 13 for PlushOctopus_T1.

Resolution is governed almost entirely by the surface envelope ``--epsr`` (see
the dataset script's docstring). Rough counts for the frame-19 surface with
``--coarsen`` (fTetWild is non-deterministic, so counts wobble per run):

    epsr      envelope   vertices   tets     tag
    1e-3      0.4 mm        ~6400    ~19100   full  (near-full detail; --mesh fine)
    5e-3      1.9 mm         ~980     ~3000   medium
    1e-2      3.9 mm         ~600     ~2000   coarse  (--mesh coarse, the default)
    2e-2      7.7 mm         ~200      ~470

The .npz bakes in over-allocation ceilings (max_particles / max_tets / max_tris)
that become the RefinementSolver's *padded* system size -- these are computed by
``make_tetmesh_models._headroom`` as ``base * mult + slack`` (clamped to the old
flat 3x / 4x), tuned to the measured worst case of the production geometric
refinement config on the coarse mesh. Coarse is unchanged; medium/full tighten.

Examples
--------
Rebuild the three meshes sim_octopus.py's --mesh flag selects::

    python -m mfem.refinement.models.make_octopus_tetmesh --epsr 1e-2 --tag coarse
    python -m mfem.refinement.models.make_octopus_tetmesh --epsr 5e-3 --tag medium
    python -m mfem.refinement.models.make_octopus_tetmesh --epsr 1e-3 --tag full

A different frame at full detail::

    python -m mfem.refinement.models.make_octopus_tetmesh --frame 0 --epsr 1e-3
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from mfem.refinement.models.make_tetmesh_models import convert
from mfem.refinement.pokeflex_episodes import EPISODES, get_episode

# ``src/mfem/refinement/models/make_octopus_tetmesh.py`` -> repo root is 4 up.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = WORKSPACE_ROOT / "models"

# Local fTetWild build (same one tetrahedralize_frame0.py shells out to).
FTETWILD_BIN = Path("/home/gabriel/auras/fTetWild/build/FloatTetwild_bin")
FTETWILD_LIBDIR = FTETWILD_BIN.parent / "gnu_13.3_cxx11_64_release"

# Fused surface tracking file names, identical across every PokeFlex episode
# (found under the episode's dataset_root(); metres, PokeFlex world frame,
# frame 0 shares the rest pose). ``template_canonical.obj`` is in mm.
TRAJECTORY_NPY = "mesh_trajectories_canonical.npy"
VALID_MASK_NPY = "valid_mask_canonical.npy"
TEMPLATE_OBJ = "template_canonical.obj"


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    lines += [f"f {a} {b} {c}" for a, b, c in (faces + 1)]
    path.write_text("\n".join(lines) + "\n")


def _vertex_adjacency(faces: np.ndarray, n_vertices: int):
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.unique(np.concatenate([edges, edges[:, ::-1]], axis=0), axis=0)
    src, dst = edges[:, 0], edges[:, 1]
    degree = np.maximum(np.bincount(src, minlength=n_vertices), 1)
    return src, dst, degree


def _fill_invalid(positions, invalid, src, dst, degree, init, iters=20):
    """Jacobi-relax untracked vertices toward their tracked neighbours, seeded
    from the rest-pose template -- identical to the dataset's
    ``fill_invalid_vertices`` so the exported surface matches what the tracking
    overlay draws."""
    filled = positions.astype(np.float64).copy()
    filled[invalid] = init[invalid]
    n = degree.shape[0]
    for _ in range(iters):
        neighbour_avg = np.stack(
            [np.bincount(src, weights=filled[dst, d], minlength=n) for d in range(3)],
            axis=1,
        ) / degree[:, None]
        filled[invalid] = neighbour_avg[invalid]
    return filled


def first_tool_free_frame(tool_trajectory: Path) -> int:
    z = np.load(tool_trajectory, allow_pickle=True)
    contact = np.asarray(z["contact_active"], dtype=bool)
    free = np.flatnonzero(~contact)
    if free.size == 0:
        raise SystemExit(f"{tool_trajectory}: tool is in contact on every frame")
    return int(free[0])


def export_frame_surface(dataset_root: Path, frame: int, out_obj: Path) -> np.ndarray:
    trajectory = np.load(dataset_root / TRAJECTORY_NPY, mmap_mode="r")
    valid_mask = np.load(dataset_root / VALID_MASK_NPY, mmap_mode="r")
    template, faces = _read_obj(dataset_root / TEMPLATE_OBJ)
    template = template / 1000.0  # obj authored in mm; tracking is in m

    n_frames = trajectory.shape[0]
    if not -n_frames <= frame < n_frames:
        raise SystemExit(f"frame {frame} out of range (episode has {n_frames} frames)")

    src, dst, degree = _vertex_adjacency(faces, trajectory.shape[1])
    verts = _fill_invalid(
        np.asarray(trajectory[frame]),
        ~np.asarray(valid_mask[frame], dtype=bool),
        src, dst, degree, template,
    )
    _write_obj(out_obj, verts, faces)
    span = verts.max(axis=0) - verts.min(axis=0)
    print(
        f"frame {frame}: {len(verts)} verts / {len(faces)} faces -> {out_obj}\n"
        f"  bbox span (m) = {np.round(span, 4)}  diag = {float(np.linalg.norm(span)):.4f}"
    )
    return verts


def run_ftetwild(surface_obj: Path, out_msh: Path, epsr: float, lr: float,
                 coarsen: bool) -> None:
    if not FTETWILD_BIN.exists():
        raise SystemExit(f"fTetWild binary not found at {FTETWILD_BIN}")
    cmd = [
        str(FTETWILD_BIN),
        "-i", str(surface_obj),
        "-o", str(out_msh),
        "-l", str(lr),
        "-e", str(epsr),
    ]
    if coarsen:
        cmd.append("--coarsen")
    env = {"LD_LIBRARY_PATH": str(FTETWILD_LIBDIR), "PATH": "/usr/bin:/bin"}
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--episode", choices=tuple(EPISODES), default="octopus",
        help="which PokeFlex tracked episode to tetrahedralize (default "
             "octopus). Sets --dataset-root, --tool-trajectory, --density, the "
             "default --frame and the models/<episode>_initial_* output names.",
    )
    ap.add_argument(
        "--frame", default=None,
        help="tracked frame to tetrahedralize (default: the --episode's "
             "initial_frame); 'auto' = first frame with the tool out of "
             "contact (tool_trajectory.npz contact_active).",
    )
    ap.add_argument("--epsr", type=float, default=1e-2,
                    help="surface envelope = epsr * bbox diag (default 1e-2).")
    ap.add_argument("--lr", type=float, default=0.05,
                    help="ideal edge length = lr * bbox diag (default 0.05).")
    ap.add_argument("--no-coarsen", dest="coarsen", action="store_false",
                    help="do not pass --coarsen to fTetWild.")
    ap.add_argument("--tag", default="coarse",
                    help="output name suffix: 'coarse' -> <episode>_initial_"
                         "coarse.{msh,npz} (default: coarse; '' for no suffix).")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="override the --episode's tracked-data directory.")
    ap.add_argument("--tool-trajectory", type=Path, default=None,
                    help="override the --episode's tool_trajectory.npz.")
    ap.add_argument("--density", type=float, default=None,
                    help="rho baked into the .npz (default: the --episode's "
                         "identified rho).")
    ap.add_argument("--k-mu", type=float, default=6.0,
                    help="placeholder Lame mu in the .npz (sim_octopus overrides).")
    ap.add_argument("--k-lambda", type=float, default=1.0,
                    help="placeholder Lame lambda in the .npz (sim_octopus overrides).")
    args = ap.parse_args()

    episode = get_episode(args.episode)
    if args.dataset_root is None:
        args.dataset_root = episode.dataset_root()
    if args.tool_trajectory is None:
        args.tool_trajectory = Path(episode.tool_trajectory_npz)
    if args.density is None:
        args.density = episode.density
    if args.frame is None:
        args.frame = str(episode.initial_frame)

    if str(args.frame).lower() == "auto":
        frame = first_tool_free_frame(args.tool_trajectory)
        print(f"auto-selected first tool-free frame: {frame} "
              f"(frame_id {frame + 1}) from {args.tool_trajectory.name}")
    else:
        frame = int(args.frame)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    # The exported surface depends only on --frame, not the fTetWild resolution,
    # so it is shared across tags.
    surface_obj = MODELS_DIR / f"{episode.key}_initial_f{frame}_surface.obj"
    out_msh = MODELS_DIR / f"{episode.key}_initial{suffix}.msh"
    out_npz = MODELS_DIR / f"{episode.key}_initial{suffix}.npz"

    export_frame_surface(args.dataset_root, frame, surface_obj)
    run_ftetwild(surface_obj, out_msh, args.epsr, args.lr, args.coarsen)
    convert(
        out_msh, out_npz,
        k_mu=args.k_mu, k_lambda=args.k_lambda, k_damp=0.0, density=args.density,
    )
    print(f"\nwrote {out_msh}\nwrote {out_npz}")


if __name__ == "__main__":
    main()
