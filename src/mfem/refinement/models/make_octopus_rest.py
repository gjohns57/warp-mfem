"""Estimate a stress-free **rest (reference) configuration** for the octopus.

``sim_octopus.py`` loads a tetrahedralization of one tracked frame (the first
tool-free frame, 13) and -- with ``--rest-shape none`` -- uses those very
vertices as the elastic energy's zero-strain reference. But that tracked shape
is not undeformed: it is the octopus sitting on the table under its own weight
(plus a small residual dent from the poke that just ended). Feeding it as its
own rest pose makes the solver treat that sagged state as zero strain, so once
the sim runs with gravity the body sags *further* away from the tracked shape.

This script recovers the shape the material would have with **no** load, such
that its static equilibrium under this sim's gravity + table contact reproduces
the tracked shape. It is the classic inverse / reference-configuration
fixed-point iteration (Twigg & Kacic-Alesic 2010-style):

    X_0 = x_tracked
    repeat:
        x_eq = static_equilibrium(rest = X_k, start = x_tracked)
        X_{k+1} = X_k + relax * (x_tracked - x_eq)

``static_equilibrium`` is a quasi-static settle with the real
:class:`RefinementSolver`: velocities are zeroed every step and a small
``--settle-dt`` is used so the implicit solve creeps to the elastic + contact +
gravity equilibrium. A handful of outer iterations drives ``x_eq`` onto
``x_tracked`` to well under a millimetre.

The solver / material / contact configuration is pulled straight from
``sim_octopus`` (its argument defaults + ``_refinement_solver_kwargs``), so the
rest shape matches the sim it will be used in. Override any physics knob that
you also override on the sim (``--gravity``, ``--mu``, ``--per-tet-material``,
``--contact-d1`` ...) so the two stay consistent.

Output: ``models/octopus_rest_<mesh>.npz`` with ``rest_particles`` (n, 3), a
copy of ``tet_indices``, and the settings it was built with. Load it via
``sim_octopus.py --rest-shape auto`` (or ``--rest-shape <path>``).

Examples
--------
Coarse mesh, all defaults (matches ``sim_octopus.py`` defaults)::

    python -m mfem.refinement.models.make_octopus_rest --mesh coarse

Match a sim you run softer and at zero gravity::

    python -m mfem.refinement.models.make_octopus_rest --mesh coarse \
        --gravity -9.81 --mu 600 --per-tet-material none
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import warp as wp
import newton

from mfem.refinement.models import MFEMRefinementModel
from mfem.refinement.solver import RefinementSolver
from mfem.refinement import sim_octopus as so
from mfem.refinement.pokeflex_episodes import EPISODES, get_episode

# ``src/mfem/refinement/models/make_octopus_rest.py`` -> repo root is 4 up.
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

# Physics knobs that must agree with the sim the rest shape is used in. Each is
# forwarded onto the sim_octopus argument namespace before the solver is built.
# ``episode`` selects the identified material / table height / mesh set.
_SIM_OVERRIDE_KEYS = (
    "episode",
    "gravity", "mu", "lmbda", "density", "energy",
    "per_tet_material", "per_tet_material_kind",
    "shell_thickness", "shell_mu", "iterations",
    "contact_d0", "contact_d1", "contact_stiffness",
    "friction_mu", "friction_mu_tool", "friction_mu_ground", "friction_eps_v",
    "line_search",
)


def _rigid_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best-fit rigid transform (Kabsch, no scale) mapping ``src`` onto ``dst``,
    returned applied to ``src``.

    The quasi-static settle also lets the free body **seat** onto the table
    (a rigid drop of a millimetre or two, plus any lateral creep under
    friction). That rigid component is not elastic deformation and no change of
    rest shape can remove it -- gravity always seats the body -- so feeding it
    into the fixed-point update makes the rest guess run away. Removing the
    rigid part here leaves only the genuine elastic sag for the iteration to
    invert.
    """
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (src - sc) @ R.T + dc


def _sim_args(overrides: dict):
    """A fully-populated ``sim_octopus`` argument namespace (all its defaults)
    with ``overrides`` applied -- so ``_refinement_solver_kwargs`` and
    ``_load_sim_model_world_frame`` see exactly what the sim would."""
    parser = so.MFEMRefinementSim.create_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    # Fill in the episode-derived defaults (material, table height, tool path,
    # ...) so _refinement_solver_kwargs / _load_sim_model_world_frame see exactly
    # what the sim would for this --episode.
    so.resolve_episode_defaults(args)
    return args


def _build_model(refinement_model: MFEMRefinementModel,
                 rest_particles: np.ndarray, args, episode):
    """A minimal free-body model: the soft mesh (rest pose = ``rest_particles``),
    the table plane, and the poker capsule parked far overhead so it never
    contacts during the settle. Shape order (capsule, plane) matches the sim so
    the per-shape friction list lines up."""
    builder = newton.ModelBuilder(up_axis=so.OCTO_UP_AXIS, gravity=args.gravity)

    parked = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 100.0, 0.0), wp.quat_identity(wp.float32))
    )
    builder.add_shape_capsule(
        parked,
        radius=episode.tip_r,
        half_height=0.5 * episode.tip_len,
    )
    builder.add_shape_plane(
        plane=(0.0, 1.0, 0.0, -episode.table_y),
        width=0.0,
        length=0.0,
    )

    model, init_particles = so._load_sim_model_world_frame(
        refinement_model, builder, args, rest_particles=rest_particles
    )
    return model, init_particles


def _settle(refinement_model: MFEMRefinementModel, rest_particles: np.ndarray,
            x_start: np.ndarray, args, episode, settle_dt: float,
            settle_steps: int, settle_tol: float, verbose: bool = False):
    """Quasi-static equilibrium of the body whose stress-free reference is
    ``rest_particles``, started from ``x_start``. Returns ``(x_eq (n,3), steps,
    converged)``."""
    n = refinement_model.particles.shape[0]
    model, _ = _build_model(refinement_model, rest_particles, args, episode)
    solver = RefinementSolver(
        model=model,
        iterations=int(args.iterations),
        max_tets=refinement_model.resolve_max_tets(args.max_tets_mult),
        max_particles=refinement_model.resolve_max_particles(args.max_particles_mult),
        max_tris=refinement_model.resolve_max_tris(args.max_tris_mult, model),
        refine_every_n_steps=1_000_000_000,
        max_new_vertices_per_refine=0,
        enable_refinement=False,
        **so._refinement_solver_kwargs(args),
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    q = np.zeros((refinement_model.resolve_max_particles(args.max_particles_mult), 3), dtype=np.float32)
    q[:n] = np.asarray(x_start, dtype=np.float32)[:n]
    state_0.particle_q.assign(q)
    state_1.particle_q.assign(q)

    prev = q[:n].copy()
    steps = settle_steps
    converged = False
    for it in range(settle_steps):
        state_0.particle_qd.zero_()
        state_1.particle_qd.zero_()
        solver.step(state_0, state_1, control, contacts, settle_dt)
        state_0, state_1 = state_1, state_0
        cur = state_0.particle_q.numpy()[:n]
        move = float(np.abs(cur - prev).max())
        prev = cur.copy()
        if verbose and (it % 50 == 0 or move < settle_tol):
            print(f"    settle step {it:4d}  max|dq| = {move * 1e3:8.4f} mm")
        if move < settle_tol:
            steps = it + 1
            converged = True
            break

    x_eq = prev.copy()
    # A fresh model + RefinementSolver is built every outer iteration; on a
    # large mesh (``--mesh fine``, ~19k tets) the CUDA mempool caches each
    # freed block and the next allocation OOMs by iteration 3 on an 8 GB card.
    # Drop every device reference and hand the cached blocks back to the driver
    # before returning so the next outer iteration starts from a clean pool.
    del solver, model, state_0, state_1, control, contacts
    gc.collect()
    try:
        wp.synchronize_device()
        dev = wp.get_device()
        if wp.is_mempool_supported(dev):
            wp.set_mempool_release_threshold(dev, 0)
    except Exception:
        pass
    return x_eq, steps, converged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", choices=tuple(EPISODES), default="octopus",
                    help="which PokeFlex tracked episode to build a rest shape "
                         "for (default octopus; same choices as sim_octopus "
                         "--episode).")
    ap.add_argument("--mesh", choices=tuple(so.OCTO_MESH_PATHS),
                    default=so.OCTO_MESH_DEFAULT,
                    help="which initial tet mesh to build a rest shape for "
                         "(same choices as sim_octopus --mesh).")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .npz (default "
                         "models/<episode>_rest_<mesh>.npz).")
    ap.add_argument("--device", default="cuda",
                    help="warp device for the settle solves (default cuda).")

    # Fixed-point iteration.
    ap.add_argument("--outer-iters", type=int, default=8,
                    help="max reference-configuration updates (default 8).")
    ap.add_argument("--outer-tol-mm", type=float, default=0.3,
                    help="stop once max|x_tracked - x_eq| is below this "
                         "(default 0.3 mm).")
    ap.add_argument("--relax", type=float, default=0.8,
                    help="under-relaxation on the update X += relax*(x-x_eq); "
                         "drop further if the iteration oscillates (default 0.8).")
    ap.add_argument("--keep-rigid", action="store_true",
                    help="do NOT rigid-align the settled shape to the tracked "
                         "one before each update (by default the rigid seating "
                         "drop onto the table is factored out so only the "
                         "elastic sag is inverted).")

    # Quasi-static settle.
    ap.add_argument("--settle-dt", type=float, default=4.0e-3,
                    help="time step for the quasi-static settle (default 4 ms; "
                         "smaller = steadier creep to equilibrium).")
    ap.add_argument("--settle-steps", type=int, default=800,
                    help="max settle steps per outer iteration (default 800).")
    ap.add_argument("--settle-tol", type=float, default=2.0e-5,
                    help="settle is converged when the largest per-step vertex "
                         "move drops below this, in metres (default 2e-5).")
    ap.add_argument("--verbose-settle", action="store_true",
                    help="print the settle residual as it converges.")

    # Physics knobs mirrored from sim_octopus (override the ones you also
    # override on the sim so the rest shape is consistent with it).
    ap.add_argument("--gravity", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--lmbda", type=float, default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--energy", choices=["arap", "neohookean"], default=None)
    ap.add_argument("--per-tet-material", default=None)
    ap.add_argument("--per-tet-material-kind",
                    choices=["field", "dyn", "sota"], default=None)
    ap.add_argument("--shell-thickness", type=float, default=None)
    ap.add_argument("--shell-mu", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--contact-d0", type=float, default=None)
    ap.add_argument("--contact-d1", type=float, default=None)
    ap.add_argument("--contact-stiffness", type=float, default=None)
    ap.add_argument("--friction-mu", type=float, default=None)
    ap.add_argument("--friction-mu-tool", type=float, default=None)
    ap.add_argument("--friction-mu-ground", type=float, default=None)
    ap.add_argument("--friction-eps-v", type=float, default=None)
    ap.add_argument("--line-search", dest="line_search", action="store_true",
                    default=None)
    args = ap.parse_args()

    wp.init()
    if args.device:
        wp.set_device(args.device)
    # Each outer iteration builds a fresh solver; keep the CUDA mempool from
    # hoarding freed blocks across them (OOM on --mesh fine otherwise).
    try:
        _dev = wp.get_device()
        if wp.is_mempool_supported(_dev):
            wp.set_mempool_release_threshold(_dev, 0)
    except Exception:
        pass

    episode = get_episode(args.episode)

    overrides = {k: getattr(args, k) for k in _SIM_OVERRIDE_KEYS}
    sim_args = _sim_args(overrides)

    mesh_path = WORKSPACE_ROOT / episode.mesh_paths[args.mesh]
    refinement_model = MFEMRefinementModel.load(
        str(mesh_path), 1.0, wp.vec3(0.0, 0.0, 0.0)
    )
    n = refinement_model.particles.shape[0]
    x_tracked = np.asarray(refinement_model.particles, dtype=np.float64)[:n].copy()

    out = args.out or (WORKSPACE_ROOT / episode.rest_npz(args.mesh))

    print(
        f"rest-shape solve: mesh '{args.mesh}' ({mesh_path.name}), {n} verts / "
        f"{refinement_model.tet_indices.shape[0]} tets\n"
        f"  gravity {sim_args.gravity} m/s^2, energy {sim_args.energy}, "
        f"per-tet material {sim_args.per_tet_material} "
        f"({sim_args.per_tet_material_kind})\n"
        f"  contact d0/d1 {sim_args.contact_d0}/{sim_args.contact_d1} m, "
        f"settle dt {args.settle_dt * 1e3:g} ms x <= {args.settle_steps} steps, "
        f"relax {args.relax}"
    )

    X = x_tracked.copy()
    t0 = time.perf_counter()
    history = []
    converged_outer = False
    for outer in range(args.outer_iters):
        x_eq, steps, settled = _settle(
            refinement_model, X.astype(np.float32), x_tracked.astype(np.float32),
            sim_args, episode, args.settle_dt, args.settle_steps, args.settle_tol,
            verbose=args.verbose_settle,
        )
        # Invert only the elastic sag, not the rigid seating drop onto the table.
        x_eq_aligned = x_eq if args.keep_rigid else _rigid_align(x_eq, x_tracked)
        resid = x_tracked - x_eq_aligned
        rn = np.linalg.norm(resid, axis=1)
        # How far the current rest guess sits from the tracked shape.
        corr = np.linalg.norm(X - x_tracked, axis=1)
        history.append((float(np.median(rn)), float(rn.max())))
        print(
            f"  outer {outer + 1}/{args.outer_iters}: settle {steps} steps"
            f"{'' if settled else ' (NOT converged)'}  "
            f"|x_tracked - x_eq| median {np.median(rn) * 1e3:7.3f} mm / "
            f"max {rn.max() * 1e3:7.3f} mm   "
            f"(rest offset median {np.median(corr) * 1e3:.2f} mm / "
            f"max {corr.max() * 1e3:.2f} mm)"
        )
        if rn.max() * 1e3 < args.outer_tol_mm:
            converged_outer = True
            break
        X = X + args.relax * resid

    dt_s = time.perf_counter() - t0
    final_med, final_max = history[-1]
    print(
        f"done in {dt_s:.1f} s  ->  residual median {final_med * 1e3:.3f} mm / "
        f"max {final_max * 1e3:.3f} mm"
        + ("" if converged_outer else "  (outer tol not reached; raise "
                                      "--outer-iters or lower --relax)")
    )

    rest_particles = X.astype(np.float32)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rest_particles=rest_particles,
        tet_indices=np.asarray(refinement_model.tet_indices, dtype=np.int32),
        tracked_particles=x_tracked.astype(np.float32),
        source_mesh=str(episode.mesh_paths[args.mesh]),
        episode=str(episode.key),
        mesh_key=args.mesh,
        gravity=np.float64(sim_args.gravity),
        energy=str(sim_args.energy),
        mu=np.float64(sim_args.mu),
        lmbda=np.float64(sim_args.lmbda),
        density=np.float64(sim_args.density),
        per_tet_material=str(sim_args.per_tet_material),
        per_tet_material_kind=str(sim_args.per_tet_material_kind),
        shell_thickness=np.float64(sim_args.shell_thickness),
        contact_d0=np.float64(sim_args.contact_d0),
        contact_d1=np.float64(sim_args.contact_d1),
        contact_stiffness=np.float64(sim_args.contact_stiffness),
        outer_iters_run=np.int64(len(history)),
        residual_median_mm=np.float64(final_med * 1e3),
        residual_max_mm=np.float64(final_max * 1e3),
        settle_dt=np.float64(args.settle_dt),
        relax=np.float64(args.relax),
    )
    drift = np.linalg.norm(rest_particles - x_tracked.astype(np.float32), axis=1)
    print(
        f"wrote {out}\n"
        f"  rest reference vs tracked shape: median "
        f"{np.median(drift) * 1e3:.2f} mm / max {drift.max() * 1e3:.2f} mm  "
        f"(this is the gravity sag + poke dent removed from the rest pose)\n"
        f"  use it with:  python -m mfem.refinement.sim_octopus "
        f"--episode {episode.key} --mesh {args.mesh} --rest-shape auto"
    )


if __name__ == "__main__":
    main()
