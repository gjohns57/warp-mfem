"""Build a coarse tetrahedral liver model from the SOFA ``liver_smooth.obj``
surface, plus the small extras (cylindrical UV, procedural tissue texture)
``soft_body_refinement.py``'s optional ``--tissue-texture`` rendering path consumes.

Pipeline
--------
1. Load the smooth surface (``models/liver_smooth.obj``, a copy of SOFA's
   ``liver-smoothUV.obj`` -- the fine visual mesh from the classic SOFA liver
   demo; its own ``vt`` coordinates are a degenerate 3-entry placeholder, not a
   real unwrap, so they are ignored).
2. Rescale it to a realistic liver size (bbox diagonal = ``--target-size``
   metres; the SOFA source mesh is authored in unlabelled Blender units,
   bbox diagonal ~8.8, which is a spatial scale, not metres).
3. Decimate to a coarse triangle count with libigl's quadric edge-collapse
   (``igl.decimate``).
4. Tetrahedralize the decimated surface with tetgen (``igl.copyleft.tetgen``),
   quality-constrained with a volume cap derived from the target edge length
   -- this is what actually keeps the *interior* coarse too (decimation alone
   only coarsens the boundary).
5. Convert to the ``.npz`` :class:`MFEMRefinementModel` format via
   ``make_tetmesh_models.build_refinement_model`` (same padding/headroom
   convention as the other pipelines).
6. Compute a simple cylindrical UV (wrap around the up axis) from the final
   tet-mesh vertex positions and synthesize a mottled reddish-brown "tissue"
   texture (base color + multi-octave value noise + faint vein streaks +
   glossy sheen) -- consumed by ``soft_body_refinement.py --tissue-texture/--tissue-uv``.

Fixing a Dirichlet boundary (e.g. "the back") is *not* baked in here --
:class:`MFEMRefinementModel` has no per-particle mass/flags field, and
``load_sim_model`` always rebuilds the newton ``Model`` fresh from a uniform
density. Instead ``soft_body_refinement.py --fix-axis/--fix-side/--fix-fraction`` zeroes
``model.particle_inv_mass`` for a coordinate-thresholded slab right after the
model is built, which is what ``RefinementSolver`` reads to pin particles
(see the commented-out predicate this mirrors in ``refinement_model.py``).

Example
-------
    python -m mfem.refinement.models.make_liver_tetmesh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import igl
import igl.copyleft.tetgen as tetgen
import numpy as np
from PIL import Image

from mfem.refinement.models.make_tetmesh_models import build_refinement_model

# ``src/mfem/refinement/models/make_liver_tetmesh.py`` -> repo root is 4 up.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = WORKSPACE_ROOT / "models"


def load_and_rescale(obj_path: Path, target_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Read the surface and uniformly rescale it so its bbox diagonal is
    ``target_size`` metres, centred at the origin."""
    V, F = igl.read_triangle_mesh(str(obj_path))
    V = V - V.mean(axis=0, keepdims=True)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    V = V * (target_size / diag)
    return V, F.astype(np.int64)


def decimate_surface(V: np.ndarray, F: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    U, G, _J, _I = igl.decimate(
        np.asfortranarray(V), np.asfortranarray(F.astype(np.int32)), target_faces
    )
    return U, G.astype(np.int64)


def tetrahedralize(V: np.ndarray, F: np.ndarray, edge_length_frac: float,
                    min_quality: float = 1.414) -> tuple[np.ndarray, np.ndarray]:
    """Quality + volume constrained tetgen call. ``edge_length_frac`` is the
    ideal interior edge length as a fraction of the bbox diagonal -- this is
    what keeps the tet mesh coarse (the decimated boundary alone doesn't
    constrain how finely tetgen fills the interior)."""
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    edge_len = edge_length_frac * diag
    tet_vol = edge_len**3 / (6.0 * np.sqrt(2.0))  # regular-tet volume of that edge length
    flags = f"pq{min_quality}a{tet_vol:.9f}"
    out = tetgen.tetrahedralize(V.astype(np.float64), F.astype(np.int64), flags=flags)
    TV, TT = out[0], out[1]
    return np.ascontiguousarray(TV, dtype=np.float32), np.ascontiguousarray(TT, dtype=np.int32)


def cylindrical_uv(points: np.ndarray, up_axis: int = 1) -> np.ndarray:
    """Simple wrap-around UV: angle around ``up_axis`` -> u, height -> v.
    Good enough for a "simple texture" on an organic blob -- it has a seam
    at the u=0/1 wrap, same as any cylindrical projection."""
    axes = [a for a in range(3) if a != up_axis]
    a, b = points[:, axes[0]], points[:, axes[1]]
    u = (np.arctan2(b, a) + np.pi) / (2.0 * np.pi)
    h = points[:, up_axis]
    v = (h - h.min()) / max(h.max() - h.min(), 1e-9)
    return np.stack([u, v], axis=1).astype(np.float32)


def make_tissue_texture(size: int, seed: int = 0) -> np.ndarray:
    """Procedural mottled reddish-brown "liver tissue" texture: multi-octave
    value noise blended between a darker maroon and a lighter pink-red, a
    handful of faint darker vein streaks, and a soft radial sheen. Returns an
    (size, size, 3) uint8 RGB image."""
    rng = np.random.default_rng(seed)

    def value_noise(shape, cell):
        """Bilinearly-upsampled random grid -- cheap stand-in for Perlin noise."""
        gy, gx = shape[0] // cell + 2, shape[1] // cell + 2
        grid = rng.random((gy, gx))
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
            (shape[1], shape[0]), Image.BICUBIC
        )
        return np.asarray(img, dtype=np.float32) / 255.0

    noise = np.zeros((size, size), dtype=np.float32)
    amp_total = 0.0
    for octave, amp in enumerate([1.0, 0.6, 0.3]):
        cell = max(size // (5 * (2**octave)), 2)
        noise += amp * value_noise((size, size), cell)
        amp_total += amp
    noise /= amp_total
    # Push the blend toward its extremes (and enlarge the blotches via the
    # coarser cell size above) so the mottling survives being minified when
    # the mesh is small on screen -- plain soft noise just averages out to a
    # near-uniform mid-tone once the camera backs off.
    t = np.clip((noise - 0.5) * 2.6 + 0.5, 0.0, 1.0)[..., None]

    dark = np.array([46.0, 10.0, 10.0])     # deep maroon
    light = np.array([212.0, 104.0, 84.0])  # lighter pink-red parenchyma
    color = dark[None, None, :] * (1.0 - t) + light[None, None, :] * t

    # A few faint darker streaks (vessels / connective tissue).
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    streaks = np.zeros((size, size), dtype=np.float32)
    for _ in range(5):
        cx, cy = rng.random(2)
        angle = rng.uniform(0, 2 * np.pi)
        dx, dy = np.cos(angle), np.sin(angle)
        dist_along = (xx - cx) * dx + (yy - cy) * dy
        dist_perp = -(xx - cx) * dy + (yy - cy) * dx
        wobble = 0.03 * np.sin(dist_along * rng.uniform(8, 16) + rng.uniform(0, 6))
        streaks += np.exp(-((dist_perp - wobble) ** 2) / (2 * 0.006**2)) * (np.abs(dist_along) < 0.4)
    streaks = np.clip(streaks, 0.0, 1.0)
    color = color * (1.0 - 0.35 * streaks[..., None]) + (dark * 0.6)[None, None, :] * (0.35 * streaks[..., None])

    # Soft radial sheen (organs read as "wet"/glossy under a key light).
    cy, cx = size * 0.35, size * 0.4
    r = np.hypot(xx * size - cx, yy * size - cy) / size
    sheen = np.clip(1.0 - r * 1.6, 0.0, 1.0) ** 2
    color = color + sheen[..., None] * 40.0

    return np.clip(color, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--obj", type=Path, default=MODELS_DIR / "liver_smooth.obj",
                    help="Source SOFA liver surface (default: models/liver_smooth.obj).")
    ap.add_argument("--target-size", type=float, default=0.20,
                    help="Rescale the surface so its bbox diagonal is this many metres "
                         "(default 0.20, a typical adult liver long-axis length).")
    ap.add_argument("--decimate-faces", type=int, default=600,
                    help="Target boundary triangle count after decimation (default 600; "
                         "300 over-smooths away the right/left lobe notch).")
    ap.add_argument("--edge-length-frac", type=float, default=1.0 / 9.0,
                    help="Ideal interior tet edge length as a fraction of the bbox "
                         "diagonal -- the knob that keeps the interior coarse (default 1/9).")
    ap.add_argument("--k-mu", type=float, default=6.0,
                    help="Placeholder Lame mu baked into the .npz (soft_body_refinement.py's --mu "
                         "overrides this at load time; default 6.0).")
    ap.add_argument("--k-lambda", type=float, default=1.0,
                    help="Lame lambda baked into the .npz -- NOT overridden by soft_body_refinement.py "
                         "at runtime, so this is the one that actually matters (default 1.0).")
    ap.add_argument("--density", type=float, default=1000.0,
                    help="Mass density, kg/m^3 (default 1000, close to real liver tissue ~1060).")
    ap.add_argument("--texture-size", type=int, default=512,
                    help="Tissue texture resolution in pixels (default 512).")
    ap.add_argument("--seed", type=int, default=0, help="Texture noise seed.")
    ap.add_argument("--tag", default="coarse",
                    help="Output name suffix: 'coarse' -> liver_coarse.{npz,uv.npy,tissue.png}.")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_npz = MODELS_DIR / f"liver{suffix}.npz"
    out_uv = MODELS_DIR / f"liver{suffix}_uv.npy"
    out_texture = MODELS_DIR / "liver_tissue.png"

    V, F = load_and_rescale(args.obj, args.target_size)
    print(f"{args.obj.name}: {V.shape[0]} verts / {F.shape[0]} faces, "
          f"rescaled to bbox diag {args.target_size} m")

    U, G = decimate_surface(V, F, args.decimate_faces)
    print(f"decimated: {U.shape[0]} verts / {G.shape[0]} faces")

    TV, TT = tetrahedralize(U, G, args.edge_length_frac)
    print(f"tetrahedralized: {TV.shape[0]} verts / {TT.shape[0]} tets")

    import warp as wp  # noqa: PLC0415 (only needed once we build the newton model)
    wp.init()
    refinement_model = build_refinement_model(
        TV, TT, k_mu=args.k_mu, k_lambda=args.k_lambda, k_damp=0.0, density=args.density,
    )
    refinement_model.save(str(out_npz))
    print(f"wrote {out_npz} (max_particles={refinement_model.max_particles}, "
          f"max_tets={refinement_model.max_tets}, max_tris={refinement_model.max_tris})")

    uv = cylindrical_uv(TV, up_axis=1)
    np.save(out_uv, uv)
    print(f"wrote {out_uv}")

    texture = make_tissue_texture(args.texture_size, seed=args.seed)
    Image.fromarray(texture).save(out_texture)
    print(f"wrote {out_texture}")


if __name__ == "__main__":
    main()
