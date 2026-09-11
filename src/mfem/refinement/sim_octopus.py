"""Octopus poke sim, set up in the PokeFlex "world" reference frame.

This is a copy of :mod:`mfem.refinement.sim` re-parameterised so the simulation
runs directly in the capture rig's frame instead of the generic Z-up sim frame.

Reference frame (from the octopus dataset ``manifest.json`` / physics caches)
--------------------------------------------------------------------------
* Fixed rig/room frame, right-handed, **metres**. Not object-centred: the
  octopus sits out at X~0.36-0.62, Y~0.28-0.41, Z~0.02-0.31, so the origin is
  in a corner of the capture volume.
* **Up is +Y.** ``physics/setup_cache.npz`` stores ``table_y = 0.278365`` plus a
  per-vertex floor clamp in Y ~ [0.2756, 0.2784]; the poking rod axis
  (``physics/wrench_measured_world.npz``) is ~ (0, -0.9999, 0.011) -- straight
  down along -Y -- and the mean contact wrench is dominated by Fy ~ -18 N.
* **Ground / table plane:** the horizontal plane Y = 0.278365 m. X and Z are the
  two in-plane (horizontal) axes; Z is the longer horizontal extent.
* Vertex correspondence / rest pose come from tracked frame 19 (frame_id 20):
  the raw episode opens mid-poke, the tool lifts off at frame 13, and by frame
  19 the plush has finished springing back and the fused surface tracking is at
  its cleanest, so that frame's surface is used as the stress-free rest state.
  ``models/octopus_initial_coarse.npz`` is its fTetWild tetrahedralization,
  already in this frame and unit; regenerate it with
  ``python -m mfem.refinement.models.make_octopus_tetmesh --frame 19``.

Consequences vs :mod:`mfem.refinement.sim`
-----------------------------------------
* ``ModelBuilder`` is built with ``up_axis=Axis.Y``. Gravity defaults to -9.81:
  the reference forward sims are quasi-static (``gravity: [0, 0, 0]``) but pin
  the mesh boundary to the tracked data, whereas this octopus is a free body on
  the table and needs its weight to resist the laterally-sweeping tool. Pass
  ``--gravity 0`` for the quasi-static regime.
* The soft mesh is loaded **without** the +90-deg-about-X rotation the original
  applies, so particle positions stay in the world frame verbatim.
* The static collision plane is Y = ``table_y`` with normal +Y.
* The keyboard/GUI "poker" capsule stands vertically along world +Y (its local
  long axis is +Z, so it is rotated -90 deg about X). Its size is the fitted
  indenter tip (``tip_r`` / ``tip_len``) and it starts above the frame-0 poke
  site, ready to press down along -Y.
* Polyscope is put in ``y_up`` mode.

Parameters copied from ``PokeFlex_Tracked/PlushOctopus_T1``
----------------------------------------------------------
==========================  ===================  =====================================
quantity                    value                source
==========================  ===================  =====================================
capture fps                 30                   forward-sim ``frame_dt`` 0.03333 s
table plane / floor clamp   Y = 0.278365 m       ``physics/setup_cache.npz``
Young's modulus E           2832 Pa              ``physics/physics_params.json`` field
Poisson's ratio nu          0.42625              ``physics/physics_params.json``
density rho                 80 kg/m^3            ``physics/physics_params.json``
-> Lame mu / lambda         ~993 / ~5739 Pa      derived from (E, nu)
tool tip radius / length    14.8 / 17.3 mm       ``physics/physics_params.json``
tool axis (world)           ~ -Y                 ``physics/wrench_measured_world.npz``
tool/body friction mu       0.297                ``physics/sota_dx10mm/hybrid_identification.json``
measured contact Fy         mean -17.8, peak -96 N   ``physics/wrench_measured_world.npz``
coarse tet mesh             ~600 v / ~2000 tets  fTetWild ``--epsr 1e-2`` on frame 19
tool path                   155 poses @ 30 fps   ``tool_trajectory.npz`` (``--tool-trajectory``)
start frame                 19 (frame_id 20)     relaxed post-poke rest frame (``--start-frame``)
==========================  ===================  =====================================

Driving the tool
----------------
``--tool-trajectory`` (default: the PlushOctopus_T1 ``tool_trajectory.npz`` under
``models/``) kinematically drives the poker capsule along the recorded
tool_to_world path. The path is sampled by wall-clock ``sim_time`` (scaled by
``--tool-playback-rate``, 1.0 = one trajectory frame per sim frame at 30 fps)
**plus an offset to ``--start-frame``**, so at ``sim_time`` 0 the capsule is at
the recorded pose for the frame the initial mesh was built from (frame 19, the
relaxed post-poke rest frame), not the episode's frame 0. Translation is linearly
interpolated and rotation is slerped. The capsule (local +Z long axis, same as
the tool) is offset back along its axis so its tip tracks the recorded contact
point.

Starting from frame 19 (well after the tool lifts off at frame 13) the recorded
poker pose already clears the reconstructed rest surface, so little or no start
retract is needed. If a chosen ``--start-frame`` still has the fitted indenter
capsule overlapping the surface, imposing that verbatim flings the free body;
``--tool-start-standoff`` (default 0 mm) then retracts the whole path along the
tool axis just until the capsule kisses the start-frame mesh, and a negative
value keeps the raw path.

The recorded pose is imposed on the solver exactly by default, so the capsule
follows the measured path smoothly. ``--tool-clamp`` routes it through the
manual poker's contact-safe limiter instead -- safer against barrier spikes,
but it lags a fast scripted plunge and then snaps the capsule down to the
commanded pose when the body yields (a visible jump once per poke).

With ``--substeps N`` the recorded tool path is sampled at N sub-frame times
(``sim_time + k*sim_dt``) per frame, so the poker advances in N gradual
increments per frame rather than one jump; raising it is the way to smooth a
fast plunge and ease the contact barrier. Each sample is a real
``np.interp`` / ``Slerp`` of the ~30 fps capture, so ``--substeps`` genuinely
refines the trajectory (not a straight chord across the frame) and a low
``--fps`` / high ``--substeps`` run follows the same tool path as a high
``--fps`` / low ``--substeps`` one. In the default (verbatim) mode the host
uploads the N poses once per frame and a small warp kernel
(``_set_tool_pose_kernel``) copies pose ``k`` into ``body_q`` each substep on
device, so it works unchanged under CUDA-graph capture (``-g``); with
``--tool-clamp`` the Python limiter can only sub-step without ``-g`` (graph +
clamp falls back to one update per frame). ``--tool-playback-rate`` still
scales overall speed. Pass ``--tool-trajectory none`` for the manual
GUI/keyboard poker.

Per-tet material field
----------------------
The reference identification fits a *spatially varying* Young's modulus: one E
per tet of its own identification mesh (``physics/setup_cache.npz`` +
``physics/stiffness_field*.npy``, 3450 tets, E in 1542-5723 Pa for the
quasi-static "field" fit). ``--per-tet-material`` (default: the bundled
``models/octopus_stiffness_field.npz``) transfers that field onto this sim's
coarse mesh: for every coarse tet, its centroid is matched to the nearest
identification-mesh tet centroids (inverse-distance blend of the 4 nearest) and
the borrowed E is converted to per-tet Lame ``k_mu`` / ``k_lambda`` with the
identified nu. ``--per-tet-material-kind`` picks the field variant
(``field`` = quasi-static, ``dyn`` = viscoelastic dyn-fit, ``sota``). Pass
``--per-tet-material none`` for a single homogeneous ``--mu`` / ``--lmbda``.

Not copied: the reference uses a linear-elastic constitutive model with
Kelvin-Voigt viscoelastic damping (eta ~ 27-300 Ns/m); this sim uses the
solver's ARAP (or ``--energy neohookean``) model with no damping. Contact here
is an IPC-style log-barrier (``--contact-d0/d1/stiffness``) rather than the
reference's surface-traction / flat-punch contact.
"""

from mfem.refinement.solver import RefinementSolver
import newton
import newton.examples
import numpy as np
import nvtx
import warp as wp
import time
from typing import Callable
import math
import ast
import os
from mfem.refinement.models import MFEMRefinementModel
from mfem.refinement.tracked_surface import TrackedSurfaceOverlay
from mfem.refinement.surface_loss import (
    TrackedCorrespondenceLoss, TrackedSurfaceLoss, merge_correspondence_row,
    surface_triangles_from_tets,
)
from mfem.refinement.pokeflex_episodes import add_episode_arg, get_episode
from newton import Axis

import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# PlushOctopus_T1 setup, copied from the pokeflex-tracking episode
#   PokeFlex_Tracked/PlushOctopus_T1/  (manifest.json, physics/*.json, *.npz)
# All lengths in metres, PokeFlex world frame, +Y up. T = 155 frames.
#
# These OCTO_* constants are now just the "octopus" entry of the per-episode
# table in mfem.refinement.pokeflex_episodes (EPISODES["octopus"]). --episode
# selects a different tracked scenario (dice / turtle / tp_roll); everything
# episode-specific is then read from the chosen PokeflexEpisode via
# resolve_episode_defaults() / self.ep. The constants are kept for the default
# path and for external importers (make_octopus_rest).
# ---------------------------------------------------------------------------
OCTO_UP_AXIS = Axis.Y
OCTO_FRAME_COUNT = 155                       # manifest.json "T"
OCTO_CAPTURE_FPS = 30                        # forward-sim frame_dt = 0.03333 s
# Tracked frame the sim starts from, used as the stress-free rest state. The raw
# episode opens mid-poke (frames 0-12 in contact) and the tool lifts off at
# frame 13, but by frame 19 the plush has finished springing back and the fused
# surface tracking is at its cleanest / roundest (that frame's reference surface
# reads best when scrubbing the replay), so models/octopus_initial_*.npz are
# fTetWild'd from frame 19 and the recorded tool path + tracked-surface overlay
# are both advanced to it at sim_time 0. Keep in sync with the mesh
# (mfem.refinement.models.make_octopus_tetmesh --frame 19).
OCTO_INITIAL_FRAME = 19

# --- Table / floor (physics/setup_cache.npz) -------------------------------
OCTO_TABLE_Y = 0.278365                      # table_y; ground plane is Y = this
OCTO_FLOOR_CLAMP_Y = (0.275587, 0.278365)   # per-vertex floor clamp range, m

# --- Rest pose (frame 0) axis-aligned bounding box, metres ---------------
OCTO_REST_MIN = np.array([0.3632, 0.2790, 0.0173])
OCTO_REST_MAX = np.array([0.6130, 0.3891, 0.2896])
OCTO_REST_CENTROID = np.array([0.4811, 0.3198, 0.1452])
# Full 155-frame trajectory bounding box (raw mesh_trajectories.npy), metres.
OCTO_TRAJ_MIN = np.array([0.3613, 0.2763, 0.0168])
OCTO_TRAJ_MAX = np.array([0.6230, 0.4116, 0.3053])

# --- Identified material (physics/physics_params.json, quasi-static "field"
#     fit): linear-elastic, isotropic. -----------------------------------
OCTO_YOUNGS_PA = 2832.182628857769          # E   (field range 1542 - 5723 Pa)
OCTO_POISSON = 0.42625                       # nu
OCTO_DENSITY = 80.0                          # rho, kg/m^3
OCTO_LATTICE_DX = 0.02                       # identification lattice spacing, m
OCTO_DAMP_ETA = 26.84                        # dyn_fit viscoelastic eta (Ns/m)
# Lame parameters from (E, nu):  mu ~ 992.9 Pa,  lambda ~ 5738.5 Pa.
OCTO_MU_PA = OCTO_YOUNGS_PA / (2.0 * (1.0 + OCTO_POISSON))
OCTO_LAMBDA_PA = (
    OCTO_YOUNGS_PA * OCTO_POISSON
    / ((1.0 + OCTO_POISSON) * (1.0 - 2.0 * OCTO_POISSON))
)

# --- Shell (surface membrane) elements ---------------------------------
# The PlushOctopus_T1 identification never fit a separate shell material:
# every "shell" block in physics/sota_dx10mm/*.json is zeroed
# (membrane_stiffness = bending_stiffness = areal_density = 0). So the
# membrane shell over the surface tris falls back to the identified BULK
# material (E, nu above) times a fabric-thickness scale: the solver's
# shell_mu is a surface shear modulus in Pa*m, mu(E, nu) * thickness.
# The 2 mm default thickness is a plush-fabric guess, not an identified
# value; --shell-thickness 0 disables the shell entirely.
OCTO_SHELL_THICKNESS = 0.002                 # m (guess; not identified)

# Per-tet identified stiffness field (E per identification-mesh tet), bundled
# from PokeFlex_Tracked/PlushOctopus_T1 physics/{setup_cache.npz,
# stiffness_field*.npy,physics_params.json}. Transferred onto the sim's coarse
# mesh by --per-tet-material (see _per_tet_lame_from_field / the module
# docstring). Keys: nodes (N,3), tets (M,4), E_field / E_dyn / E_sota (M,),
# nu (scalar), eta_dyn (scalar).
OCTO_STIFFNESS_FIELD_DEFAULT = "models/octopus_stiffness_field.npz"

# Initial (frame-19, OCTO_INITIAL_FRAME) tet meshes at three resolutions, all
# fTetWild'd from the same tracked rest surface by
# mfem.refinement.models.make_octopus_tetmesh --frame 19 (varying --epsr /
# --tag). Selected with --mesh; "fine" is the near-full-detail "full" tag.
# fTetWild is non-deterministic, so counts vary slightly per rebuild.
#   coarse  ~600 v / ~2000 tets    (--epsr 1e-2)
#   medium  ~980 v / ~3000 tets    (--epsr 5e-3)
#   fine    ~6400 v / ~19100 tets  (--epsr 1e-3, near full surface detail)
OCTO_MESH_PATHS = {
    "coarse": "models/octopus_initial_coarse.npz",
    "medium": "models/octopus_initial_medium.npz",
    "fine": "models/octopus_initial_full.npz",
}
OCTO_MESH_DEFAULT = "coarse"

# --- Poking tool (physics/physics_params.json + wrench_measured_world.npz) -
OCTO_TIP_R = 0.014842664490570312           # fitted indenter tip radius, m
OCTO_TIP_LEN = 0.017289941100636492         # fitted indenter tip length, m
OCTO_POKER_AXIS_WORLD = np.array([0.0012, -0.99994, 0.0107])  # T_WT tool +z
OCTO_POKE_XZ_FRAME0 = (0.49361, 0.11301)    # tool origin (X, Z) at frame 0, m
OCTO_FRICTION_MU = 0.297                     # fitted tool/body friction coeff.
OCTO_TABLE_FRICTION_MU = 0.3                 # fitted body/table friction coeff.
# The solver's Contact tracks, per particle, which rigid shape it is nearest
# to (the poker capsule = shape 0, the table plane = shape 1, in the order
# they're added below) and looks up that shape's own Coulomb coefficient, so
# the tool and the table can be tuned independently (--friction-mu-tool /
# --friction-mu-ground; each falls back to --friction-mu when unset). At the
# fitted ~0.3 for both, the free octopus skates on the table and rides up
# under the laterally-sweeping tool, so the shared default is cranked to a
# higher value: enough table grip to keep the body planted while the tool
# drags across it. Pass --friction-mu 0.297 to get back to the identified
# coefficient for both, or split them (e.g. lower --friction-mu-ground so the
# body can slide, higher --friction-mu-tool so the tool grabs and drags it)
# to reproduce the tracked data's lateral slide under the tool.
OCTO_SIM_FRICTION_MU = 0.9
OCTO_CONTACT_THRESHOLD_N = 1.5              # qc tool-contact force threshold
# Measured contact wrench (world): mean Fy = -17.8 N, peak Fy = -96.0 N.
OCTO_WRENCH_FY_MEAN = -17.78
OCTO_WRENCH_FY_PEAK = -96.0

# Recorded tool trajectory for this episode (155 frames at 30 fps,
# tool_to_world 4x4 transforms in the PokeFlex world frame). Passed to
# --tool-trajectory to kinematically drive the poker along the measured path.
OCTO_TOOL_TRAJECTORY_DEFAULT = (
    "models/PokeFlex_Tracked-tool-trajectories-4objects/"
    "PokeFlex_Tracked-tool-trajectories-4objects/PlushOctopus_T1/tool_trajectory.npz"
)


# --- Episode selection ----------------------------------------------------
# argparse builds its defaults before it has seen --episode, so every option
# whose default depends on the episode is given a None sentinel and filled in
# here, straight after parsing, from the chosen PokeflexEpisode. An explicit CLI
# value is never None so it always wins. Idempotent -- safe to call from init(),
# __init__ and the make_octopus_rest precompute alike.
_EPISODE_DERIVED_DEFAULTS = (
    ("mu", lambda ep: ep.mu_pa),
    ("lmbda", lambda ep: ep.lambda_pa),
    ("density", lambda ep: ep.density),
    ("refine_density", lambda ep: ep.density),
    ("shell_thickness", lambda ep: ep.shell_thickness),
    ("per_tet_material", lambda ep: ep.stiffness_field_npz),
    ("fps", lambda ep: ep.capture_fps),
    ("start_frame", lambda ep: ep.initial_frame),
    ("tool_trajectory", lambda ep: ep.tool_trajectory_npz),
    ("friction_mu", lambda ep: ep.sim_friction_mu),
    ("tracked_surface", lambda ep: ep.tracked_surface_npy()),
)


def resolve_episode_defaults(args):
    """Fill any episode-derived option still at its ``None`` sentinel from
    ``args.episode`` and return the resolved
    :class:`~mfem.refinement.pokeflex_episodes.PokeflexEpisode`."""
    ep = get_episode(args)
    for name, getter in _EPISODE_DERIVED_DEFAULTS:
        if getattr(args, name, None) is None:
            setattr(args, name, getter(ep))
    return ep


@wp.kernel
def _translate_bodies_kernel(body_q: wp.array(dtype=wp.transform), offset: wp.vec3):
    tid = wp.tid()
    tf = body_q[tid]
    pos = wp.transform_get_translation(tf) + offset
    rot = wp.transform_get_rotation(tf)
    body_q[tid] = wp.transform(pos, rot)


@wp.kernel
def _set_tool_pose_kernel(
    pose_seq: wp.array(dtype=wp.transform),
    k: int,
    body_q: wp.array(dtype=wp.transform),
):
    """Write ``body_q[0] = pose_seq[k]``: the tool pose the host sampled from
    the recorded path for substep ``k`` of the current frame.

    Runs entirely on device, so it can sit *inside* the captured CUDA graph.
    ``k`` is baked into each captured launch; ``pose_seq`` (one pose per solver
    substep) is refreshed from the host once per frame, before the replay, in
    :meth:`OctopusSim._refresh_tool_path`.
    """
    body_q[0] = pose_seq[k]


def _lame_from_youngs(E, nu):
    """(E, nu) -> (mu, lambda). Works elementwise on arrays."""
    E = np.asarray(E, dtype=np.float64)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lmbda


def _per_tet_lame_from_field(coarse_particles: np.ndarray,
                             coarse_tets: np.ndarray, args):
    """Transfer the identified per-tet stiffness field onto the sim's coarse
    mesh and return ``(k_mu, k_lambda)`` float32 arrays, one entry per coarse
    tet (or ``(None, None)`` when ``--per-tet-material`` is disabled).

    ``--per-tet-material`` points at an ``.npz`` bundle with the identification
    mesh (``nodes`` (N,3), ``tets`` (M,4)), the per-identification-tet Young's
    modulus (``E_field`` / ``E_dyn`` / ``E_sota``, chosen by
    ``--per-tet-material-kind``), and the identified ``nu``. Both meshes are in
    the PokeFlex world frame (metres), so the transfer is a pure spatial
    lookup: each coarse-tet centroid borrows an inverse-distance-weighted blend
    of the Young's modulus at its 4 nearest identification-tet centroids, then
    that E is mapped to Lame parameters with the identified nu. The blended E is
    clamped to the field's own [min, max] so the interpolation can't overshoot.
    """
    path = getattr(args, "per_tet_material", None)
    if not path or str(path).lower() in ("", "none", "off"):
        return None, None

    from scipy.spatial import cKDTree  # noqa: PLC0415

    z = np.load(path, allow_pickle=True)
    id_nodes = np.asarray(z["nodes"], dtype=np.float64)
    id_tets = np.asarray(z["tets"], dtype=np.int64)
    kind = getattr(args, "per_tet_material_kind", "field")
    e_key = {"field": "E_field", "dyn": "E_dyn", "sota": "E_sota"}[kind]
    id_E = np.asarray(z[e_key], dtype=np.float64)
    nu = float(z["nu"]) if "nu" in z.files else get_episode(args).poisson

    id_centroids = id_nodes[id_tets].mean(axis=1)
    coarse_centroids = (
        np.asarray(coarse_particles, dtype=np.float64)[coarse_tets].mean(axis=1)
    )

    k = min(4, id_centroids.shape[0])
    dist, idx = cKDTree(id_centroids).query(coarse_centroids, k=k)
    dist = np.atleast_2d(dist.T).T.reshape(coarse_centroids.shape[0], k)
    idx = np.atleast_2d(idx.T).T.reshape(coarse_centroids.shape[0], k)
    w = 1.0 / np.maximum(dist, 1e-9) ** 2
    E_tet = (w * id_E[idx]).sum(axis=1) / w.sum(axis=1)
    E_tet = np.clip(E_tet, id_E.min(), id_E.max())

    mu, lmbda = _lame_from_youngs(E_tet, nu)

    print(
        f"per-tet material '{kind}' ({e_key}) transferred from {path}: "
        f"{id_tets.shape[0]} id tets -> {coarse_tets.shape[0]} coarse tets, "
        f"nearest-centroid dist mean {dist[:, 0].mean() * 1e3:.1f} mm / "
        f"max {dist[:, 0].max() * 1e3:.1f} mm; "
        f"E (Pa) min {E_tet.min():.0f} / med {np.median(E_tet):.0f} / "
        f"max {E_tet.max():.0f}, nu {nu:.4f} -> "
        f"mu (Pa) {mu.min():.0f}..{mu.max():.0f}, "
        f"lambda (Pa) {lmbda.min():.0f}..{lmbda.max():.0f}"
    )
    return mu.astype(np.float32), lmbda.astype(np.float32)


def _load_sim_model_world_frame(refinement_model: MFEMRefinementModel,
                                builder: newton.ModelBuilder, args,
                                rest_particles: np.ndarray | None = None):
    """Like :meth:`MFEMRefinementModel.load_sim_model` but **without** the
    +90-deg-about-X rotation, and with the material taken from ``args`` instead
    of the ``.npz`` (whose baked ``k_lambda`` is a placeholder 1.0).

    The octopus ``.npz`` is already stored in the PokeFlex world frame (metres,
    +Y up), so we add the soft mesh with an identity rotation and let it live in
    world space directly. By default (``--per-tet-material``) the elastic Lame
    parameters are a per-tet field transferred from the identified stiffness
    field (see :func:`_per_tet_lame_from_field`); with ``--per-tet-material
    none`` they collapse to the scalars ``args.mu`` / ``args.lmbda``, which
    default to the Lame parameters of the identified homogeneous (E, nu) for
    PlushOctopus_T1. ``args.density`` defaults to the identified rho = 80 kg/m^3.

    ``rest_particles`` (optional, ``--rest-shape``) is a **stress-free reference
    configuration**: the tracked mesh in ``refinement_model`` is itself a
    gravity-loaded (and slightly poke-dented) shape, so using it as its own
    rest pose makes the solver treat that deformed state as zero strain and the
    octopus then sags further under gravity. When ``rest_particles`` is given it
    is fed to ``add_soft_mesh`` instead, so the per-tet rest pose ``DmInv`` /
    rest volumes are built from the un-sagged shape; the caller still starts the
    simulation from the tracked positions (returned as the second element).
    Build it with ``python -m mfem.refinement.models.make_octopus_rest``.

    Returns ``(model, init_particle_q)`` where ``init_particle_q`` is the
    ``(max_particles, 3)`` tracked/start positions the caller should assign into
    the sim states (identical to the mesh vertices when ``rest_particles`` is
    ``None``).
    """
    n = refinement_model.particles.shape[0]
    max_particles = refinement_model.resolve_max_particles(getattr(args, "max_particles_mult", None))
    init_particles = np.zeros((max_particles, 3), dtype=np.float32)
    init_particles[:n, :] = refinement_model.particles

    if rest_particles is not None:
        rest_particles = np.asarray(rest_particles, dtype=np.float32)
        if rest_particles.shape[0] < n:
            raise ValueError(
                f"rest_particles has {rest_particles.shape[0]} rows but the "
                f"mesh has {n} particles"
            )
        build_particles = np.zeros_like(init_particles)
        build_particles[:n, :] = rest_particles[:n]
    else:
        build_particles = init_particles

    # The identified stiffness field is looked up by world position, so match
    # the tracked positions (init), not the rest reference.
    k_mu, k_lambda = _per_tet_lame_from_field(
        np.asarray(refinement_model.particles, dtype=np.float64),
        np.asarray(refinement_model.tet_indices),
        args,
    )
    if k_mu is None:
        k_mu, k_lambda = args.mu, args.lmbda

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(wp.float32),
        scale=wp.float32(1.0),
        vertices=build_particles,
        indices=refinement_model.tet_indices.flatten(),
        density=args.density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=refinement_model.tet_materials[:, 2],
        add_surface_mesh_edges=False,
        validate_mesh=False,
    )

    return builder.finalize(), init_particles


def _refinement_solver_kwargs(args) -> dict:
    """The :class:`RefinementSolver` tunables shared by the live sim and the
    offline rest-shape precompute (:mod:`mfem.refinement.models.make_octopus_
    rest`), so both build the solver identically. ``model`` / ``iterations`` /
    ``max_tets`` and the refinement on/off toggles are passed separately by
    each caller. Every field is read with a default so the precompute can pass
    a lighter argument namespace.
    """
    g = lambda name, default: getattr(args, name, default)
    friction_mu = g("friction_mu", OCTO_SIM_FRICTION_MU)
    mu_tool = g("friction_mu_tool", None)
    mu_ground = g("friction_mu_ground", None)
    shell_mu = g("shell_mu", None)
    if shell_mu is None:
        shell_mu = g("mu", OCTO_MU_PA) * g("shell_thickness", OCTO_SHELL_THICKNESS)
    return dict(
        energy=g("energy", "arap"),
        preconditioner=g("preconditioner", False),
        line_search=g("line_search", False),
        refine_density=g("refine_density", OCTO_DENSITY),
        refine_tet_score_weight=g("refine_tet_score_weight", 100.0),
        refine_vertex_score_weight=g("refine_vertex_score_weight", 200.0),
        penetrating_edge_score=g("penetrating_edge_score", 1.0e6),
        refine_conflict_iterations=g("refine_conflict_iterations", 10),
        refine_split_position=g("refine_split_position", 0.5),
        refine_hashmap_load_factor=g("refine_hashmap_load_factor", 0.5),
        refine_hashmap_edges_per_tet=g("refine_hashmap_edges_per_tet", 6),
        refine_scoring=g("refine_scoring", "legacy"),
        refine_min_edge_length=g("refine_min_edge_length", None),
        refine_elastic_weight=g("refine_elastic_weight", 1.0),
        refine_contact_weight=g("refine_contact_weight", 1.0),
        refine_tri_contact_weight=g("refine_tri_contact_weight", 1.0),
        refine_geometric_threshold=g("refine_geometric_threshold", 1.0),
        cg_max_iterations=g("cg_max_iterations", 5000),
        cg_tolerance=g("cg_tolerance", 1.0e-6),
        cg_check_every=g("cg_check_every", 0),
        preconditioner_singular_threshold=g("preconditioner_singular_threshold", 1.0e-20),
        line_search_max_iterations=g("line_search_max_iterations", 30),
        line_search_alpha0=g("line_search_alpha0", 1.0),
        line_search_tau=g("line_search_tau", 0.5),
        line_search_c=g("line_search_c", 0.01),
        line_search_threshold=g("line_search_threshold", 1.0e-8),
        attachment_stiffness=g("attachment_stiffness", 1.0e1),
        contact_d0=g("contact_d0", 2.0e-3),
        contact_d1=g("contact_d1", 8.0e-3),
        contact_stiffness=g("contact_stiffness", 1.0e5),
        friction_mu=[
            mu_tool if mu_tool is not None else friction_mu,
            mu_ground if mu_ground is not None else friction_mu,
        ],
        friction_eps_v=g("friction_eps_v", 5.0e-3),
        shell_mu=shell_mu,
        tri_contact=not g("no_tri_contact", False),
    )


def _load_rest_particles(args, refinement_model: MFEMRefinementModel,
                         mesh_key: str) -> np.ndarray | None:
    """Resolve ``--rest-shape`` to an ``(n, 3)`` stress-free reference vertex
    array, or ``None`` to use the tracked mesh as its own rest pose.

    ``none`` / ``off`` -> ``None``. ``auto`` -> ``models/octopus_rest_<mesh>.npz``
    if it exists (built by :mod:`mfem.refinement.models.make_octopus_rest`),
    else ``None`` with a hint. Anything else is treated as an explicit ``.npz``
    path with a ``rest_particles`` array.

    ``rest_particles`` is vertex-index-matched to the ``octopus_initial_<mesh>``
    tetrahedralization it was built from: entry ``i`` is the rest position of
    that mesh's vertex ``i``. fTetWild is non-deterministic, so **regenerating
    the initial mesh desynchronises the two** -- the rest file must be rebuilt
    alongside it (``make_octopus_rest --mesh <mesh>``). A stale pair pushes a
    scrambled zero-strain reference into the elastic energy and the solve
    diverges on the first step, so a mismatch is a hard error here rather than a
    silent slice. The check: vertex count must match exactly, and (when the rest
    npz carries ``tracked_particles``, the source mesh's vertices at build time)
    those must still coincide with this mesh's vertices.
    """
    spec = getattr(args, "rest_shape", "auto")
    s = str(spec).strip().lower()
    if s in ("", "none", "off"):
        return None
    episode = get_episode(args)
    rebuild_cmd = (
        f"python -m mfem.refinement.models.make_octopus_rest "
        f"--episode {episode.key} --mesh {mesh_key}"
    )
    rebuild_hint = (
        f"rebuild it with '{rebuild_cmd}' (the initial mesh was regenerated "
        f"without it)"
    )
    if s == "auto":
        path = episode.rest_npz(mesh_key)
        if not os.path.exists(path):
            print(
                f"--rest-shape auto: {path} not found; using the tracked mesh "
                f"as its own (deformed) rest pose. Build the un-sagged rest "
                f"with '{rebuild_cmd}'."
            )
            return None
    else:
        path = str(spec)
        if not os.path.exists(path):
            raise SystemExit(f"--rest-shape: {path} not found")

    z = np.load(path, allow_pickle=True)
    if "rest_particles" not in z.files:
        raise SystemExit(f"--rest-shape: {path} has no 'rest_particles' array")
    rp = np.asarray(z["rest_particles"], dtype=np.float32)
    mesh_particles = np.asarray(refinement_model.particles, dtype=np.float32)
    n = mesh_particles.shape[0]
    if rp.shape[0] != n:
        raise SystemExit(
            f"--rest-shape: {path} has {rp.shape[0]} rest vertices but the "
            f"'{mesh_key}' mesh has {n} -- {rebuild_hint}."
        )
    if "tracked_particles" in z.files:
        tp = np.asarray(z["tracked_particles"], dtype=np.float32)
        drift = (
            np.linalg.norm(tp[:n] - mesh_particles, axis=1)
            if tp.shape[0] == n else np.array([np.inf])
        )
        # The rest solve settles the tracked mesh by a few mm; anything past a
        # couple of cm means the npz was built from a different tetrahedra-
        # lization (scrambled vertex order), not this one.
        if not np.isfinite(drift.max()) or drift.max() > 0.02:
            raise SystemExit(
                f"--rest-shape: {path} was built from a different "
                f"tetrahedralization of '{mesh_key}' (its stored source "
                f"vertices are {drift.max() * 1e3:.0f} mm off this mesh) -- "
                f"{rebuild_hint}."
            )
    d = np.linalg.norm(rp - mesh_particles, axis=1)
    if np.median(d) > 0.02:
        raise SystemExit(
            f"--rest-shape: {path} rest reference sits median "
            f"{np.median(d) * 1e3:.0f} mm off the tracked start (expected a few "
            f"mm of gravity sag) -- {rebuild_hint}."
        )
    print(
        f"--rest-shape: loaded {path}  (rest reference vs tracked start: "
        f"median {np.median(d) * 1e3:.2f} mm / max {d.max() * 1e3:.2f} mm)"
    )
    return rp


class MFEMRefinementSim:
    def __init__(self,refinement_model: MFEMRefinementModel, args):
        self.sim_time = 0.0
        # Which PokeFlex tracked episode this run reproduces (--episode, default
        # "octopus"). All episode-specific constants -- material, table height,
        # tool geometry, tracked-surface overlay -- come from here. Also fills in
        # any episode-derived CLI default the user did not override.
        self.ep = resolve_episode_defaults(args)
        # Tracked episode frame the run starts from (self.ep.initial_frame unless
        # --start-frame overrode it). The initial mesh is built from this frame;
        # the recorded tool path and the tracked-surface overlay are both offset
        # so sim_time 0 lands here.
        self.start_frame = int(args.start_frame)
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.iterations = args.iterations
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.do_capture = args.graph_capture
        self.do_line_search = args.line_search

        # Stress-free reference configuration for the elastic energy (see
        # --rest-shape / _load_rest_particles). None -> the tracked mesh is its
        # own rest pose (so its gravity sag + residual poke dent are baked in as
        # zero strain); an (n, 3) array -> the un-sagged rest built offline by
        # mfem.refinement.models.make_octopus_rest, from which the per-tet rest
        # DmInv / volumes are computed while the sim still starts from the
        # tracked positions.
        self._rest_particles = _load_rest_particles(
            args, refinement_model, getattr(args, "mesh", OCTO_MESH_DEFAULT)
        )

        # self.sim_duration = args.sim_duration

        self.record_energy = args.record_energy



        # PokeFlex world frame: +Y up, gravity straight down along -Y (default
        # -9.81). The reference forward sims run quasi-static (gravity [0,0,0])
        # but there the mesh boundary is pinned by the tracked trajectory; our
        # octopus is a free body resting on the table, so it needs its weight to
        # resist being dragged along by the laterally-sweeping tool. Pass
        # --gravity 0 for the quasi-static regime (expect the free body to
        # slide if the tool pushes sideways).
        builder = newton.ModelBuilder(up_axis=OCTO_UP_AXIS, gravity=args.gravity)  # +Y up, all episodes

        # ---- Keyboard/GUI-controllable capsule -----------------------------
        # This sim keeps the soft mesh in the world frame verbatim (no +90-deg
        # rotation), so the octopus particles are already in world space:
        p = np.asarray(refinement_model.particles, dtype=np.float64)[:, :3]
        octo_hi = p.max(axis=0)          # world-frame AABB max; octo_hi[1] = top

        # Poker capsule = the fitted indenter tip geometry (physics_params.json):
        # a short, fat rounded punch, tip_r ~ 14.8 mm, tip_len ~ 17.3 mm. The
        # collision primitive is a capsule (radius + cylinder segment); use
        # tip_r for the radius and tip_len for the full segment length.
        self.capsule_radius = self.ep.tip_r
        self.capsule_half_height = 0.5 * self.ep.tip_len
        # Newton's capsule collision primitive is aligned to its local +Z axis,
        # so the render mesh is built along Z and we rotate the body to stand it
        # up along world +Y (see _capsule_rot_default).
        self.capsule_axis = Axis.Z

        # ---- Recorded tool trajectory (optional) -------------------------
        # When --tool-trajectory is given, the capsule is driven kinematically
        # along the measured tool_to_world path instead of the GUI/keyboard.
        #
        # The start-of-path retract (_load_tool_trajectory) checks the capsule
        # SDF against a point cloud. Bare vertices miss face punch-through: on
        # the coarse mesh the octopus crown has ~50 mm triangles, so the fitted
        # ~30 mm capsule can sit with every vertex a hair outside it while a
        # triangle interior is several mm inside -- exactly the frame-19 start
        # pose, which then launches the free body on the first contact solve.
        # Densify with the surface-triangle edge midpoints and centroids so the
        # retract sees the faces too.
        self._load_tool_trajectory(
            args, self._surface_sample_points(p, refinement_model.tet_indices)
        )

        if self._tool_traj_active:
            # Default / reset pose = the trajectory's first sample.
            self._capsule_pos_default, self._capsule_rot_default = (
                self._sample_tool_pose(0.0)
            )
        else:
            # Default pose: over the frame-0 poke site in the horizontal (X, Z)
            # plane (tool origin at frame 0), standing vertically along +Y with
            # its lower cap a small gap above the octopus top (Y-max), ready to
            # press down along -Y like the real rod.
            gap = 0.5 * self.capsule_radius
            poke_xz = self.ep.poke_xz_frame0
            self._capsule_pos_default = np.array(
                [poke_xz[0],
                 octo_hi[1] + gap + self.capsule_half_height + self.capsule_radius,
                 poke_xz[1]],
                dtype=np.float64,
            )
            # XYZ euler, degrees. -90 about X maps the capsule's local +Z long
            # axis onto world +Y so the poker is vertical in the PokeFlex frame.
            self._capsule_rot_default = np.array([-90.0, 0.0, 0.0], dtype=np.float64)

        # Live, user-controlled pose (mutated by the GUI / keyboard).
        self.capsule_pos = self._capsule_pos_default.copy()
        self.capsule_rot = self._capsule_rot_default.copy()
        # Discrete nudge increments for the GUI buttons.
        self.capsule_move_step = 0.25 * self.capsule_radius
        self.capsule_rot_step = 5.0
        # Continuous speeds for held-down hotkeys (per second).
        self.capsule_move_speed = 3.0 * self.capsule_radius
        self.capsule_rot_speed = 90.0
        self.capsule_visible = True

        # ---- Contact-safe kinematic driving ------------------------------
        # The capsule is a kinematically-scripted body: whatever pose we ask
        # for is imposed on the solver verbatim. If a keypress jumps it deep
        # into the octopus in a single frame, the contact barrier sees a huge
        # penetration and the step explodes.
        #
        # So every frame we evaluate the capsule SDF against the soft-body
        # particles at the *requested* new pose and, if it would come closer
        # than `contact_standoff`, bisect back along the move until it doesn't.
        # An extra `push_rate` per frame of deeper approach is allowed so the
        # user can still lean into the octopus and deform it -- just gradually.
        self._contact_d1 = float(args.contact_d1)
        # Keep the nearest particle no closer than this to the capsule surface.
        # A hair inside d1 so there is a little contact force to push with, but
        # never down into the stiff quadratic-extrapolation part of the barrier.
        self.capsule_contact_standoff = 0.7 * self._contact_d1
        self.capsule_push_rate = 1.5 * self._contact_d1   # extra approach / frame
        self.capsule_rot_clamp_far = 30.0                 # deg / frame when clear
        self.capsule_rot_clamp_near = 3.0                 # deg / frame while in contact
        self._capsule_gap = np.inf                        # nearest particle -> capsule (display)
        self._capsule_applied_pos = self.capsule_pos.copy()
        self._capsule_applied_rot = self.capsule_rot.copy()

        # ---- Early-out watchdog (see check_bail / --bail-unstable) ---------
        # Rest-pose AABB diagonal of the soft body, for the blow-up test.
        self._rest_bbox_diag = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
        self._bail_enabled = bool(getattr(args, "bail_unstable", False))
        self._bail_max_disp = float(getattr(args, "bail_max_disp", 0.15))
        self._bail_bbox_ratio = float(getattr(args, "bail_bbox_ratio", 4.0))
        self._bail_penetration = float(getattr(args, "bail_penetration", 0.015))
        self._prev_active_pts = None
        self._bail_reason = None

        capsule_body = builder.add_body(xform=self._capsule_transform())
        builder.add_shape_capsule(
            capsule_body,
            radius=self.capsule_radius,
            half_height=self.capsule_half_height,
        )

        # Static table plane at Y = table_y (normal +Y) so the octopus rests on
        # it exactly as in the capture rig. width/length = 0 makes it infinite
        # for collision.
        self.ground_height = self.ep.table_y
        self.ground_extent = 10.0
        builder.add_shape_plane(
            plane=(0.0, 1.0, 0.0, -self.ground_height),
            width=0.0,
            length=0.0,
        )

        self.model, self._initial_particle_q = _load_sim_model_world_frame(
            refinement_model, builder, args, rest_particles=self._rest_particles
        )


        # Per-shape Coulomb coefficient inside _refinement_solver_kwargs: shape 0
        # is the poker capsule (add_shape_capsule above), shape 1 is the table
        # plane (add_shape_plane) -- see Contact.friction_mu / --friction-mu-tool
        # / --friction-mu-ground. The full kwarg set is shared with the offline
        # rest-shape precompute (mfem.refinement.models.make_octopus_rest) so
        # both build the solver identically.
        self.solver = RefinementSolver(
            model=self.model,
            iterations=self.iterations,
            max_tets=refinement_model.resolve_max_tets(args.max_tets_mult),
            max_particles=refinement_model.resolve_max_particles(args.max_particles_mult),
            max_tris=refinement_model.resolve_max_tris(args.max_tris_mult, self.model),
            refine_every_n_steps=args.refine_every,
            max_new_vertices_per_refine=args.max_new_vertices,
            enable_refinement=args.refine,
            **_refinement_solver_kwargs(args),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        # With a separate stress-free rest reference (--rest-shape) the model
        # was built from the un-sagged shape, so model.particle_q holds the rest
        # positions (and the solver's rest_particle_q / DmInv are taken from
        # them). The simulation itself must still start from the tracked mesh,
        # so overwrite the live state positions here.
        if self._rest_particles is not None:
            self.state_0.particle_q.assign(self._initial_particle_q)
            self.state_1.particle_q.assign(self._initial_particle_q)

        particle_ct = self.solver._additional_state_0.active_particle_count.numpy()[0]
        # Surface triangulation, seeded from the initial tet mesh (newton's
        # add_soft_mesh computes this into model.tri_indices) and from then on
        # kept up to date by the solver's refine pass (additional_state.
        # tri_indices / active_tri_count), split in step with the tets --
        # see scatter_tris in refinement.py.
        tri_ct = self.solver._additional_state_0.active_tri_count.numpy()[0]
        self.ps_volume = ps.register_surface_mesh("Soft body", self.state_0.particle_q[:particle_ct].numpy(), self.solver._additional_state_0.tri_indices[:tri_ct].numpy())
        self._old_vertex_count = particle_ct

        capsule_mesh = newton.Mesh.create_capsule(
            self.capsule_radius, self.capsule_half_height, up_axis=self.capsule_axis
        )
        self.ps_capsule = ps.register_surface_mesh(
            "Capsule", capsule_mesh.vertices, capsule_mesh.indices.reshape(-1, 3)
        )
        self.ps_capsule.set_color([0.85, 0.55, 0.15])

        # The poker tip (tip_r ~ 15 mm) is fat next to the octopus and spends
        # most of a deep poke *inside* the opaque soft-body mesh, where it is
        # simply hidden from view. Two aids: (1) --soft-body-alpha makes the
        # octopus see-through so the buried capsule shows; (2) this thin shaft
        # marker sticks up out of the mesh so the tool's location is always
        # visible even when the capsule itself is submerged.
        self._soft_body_alpha = float(getattr(args, "soft_body_alpha", 1.0))
        self.ps_tool_shaft = ps.register_curve_network(
            "tool shaft", np.zeros((2, 3), dtype=np.float32),
            np.array([[0, 1]], dtype=np.int32),
        )
        self.ps_tool_shaft.set_radius(0.15 * self.capsule_radius, relative=False)
        self.ps_tool_shaft.set_color([0.95, 0.75, 0.2])

        self._sync_capsule()

        # Drive the capsule from the polyscope window (ImGui panel + hotkeys).
        ps.set_user_callback(self._capsule_gui_callback)

        # Table plane render quad: horizontal (X, Z) plane at Y = table_y.
        e = self.ground_extent
        h = self.ground_height
        ground_verts = np.array(
            [[-e, h, -e], [e, h, -e], [e, h, e], [-e, h, e]], dtype=np.float32
        )
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.ps_ground = ps.register_surface_mesh("Ground", ground_verts, ground_faces)
        # self.ps_ground.set_enabled(False)

        # See-through overlay of the dataset's fused surface tracking, played
        # back in lockstep with the sim so the measured surface can be compared
        # with the solver's mesh. sim_time 0 shows tracked frame ``start_frame``
        # (the frame the initial mesh was built from), not the episode's frame 0.
        self.tracked_surface = TrackedSurfaceOverlay(
            getattr(args, "tracked_surface", None),
            float(getattr(args, "tracked_surface_alpha", 0.3)),
            rate=float(getattr(args, "tool_playback_rate", 1.0) or 1.0),
            loop=bool(getattr(args, "tool_loop", False)),
            start_frame=self.start_frame,
            detrend=getattr(args, "tracked_surface_detrend", "rigid"),
        )

        # Optional surface-tracking loss: per-frame MSE of the sim surface
        # vertices against the nearest point on the tracked mesh (same overlay
        # geometry / frame mapping). Accumulated in self._loss_rows and
        # summarised at the end of run().
        self.surface_loss = None
        self._loss_rows: list[dict] = []
        self._surface_loss_every = max(int(getattr(args, "surface_loss_every", 10)), 1)
        if getattr(args, "surface_loss", False):
            loss_target = self.tracked_surface
            ref_path = getattr(args, "surface_loss_reference", None)
            if ref_path:
                from mfem.refinement.surface_loss import RecordingOverlay  # noqa: PLC0415
                loss_target = RecordingOverlay(ref_path)
                print(f"surface loss: scoring against reference recording {ref_path} "
                      f"({loss_target.T} frames, {loss_target.traj.shape[1]} verts)")
            self.surface_loss = TrackedSurfaceLoss(
                loss_target,
                project_to_faces=not getattr(args, "surface_loss_no_project", False),
                use_valid_mask=not getattr(args, "surface_loss_keep_untracked", False),
                symmetric=bool(getattr(args, "surface_loss_symmetric", False)),
                exclude_tool_margin=getattr(args, "surface_loss_exclude_tool", None),
            )
            if not self.surface_loss.active:
                print("surface loss: tracked surface disabled; --surface-loss ignored")
                self.surface_loss = None

        # Optional material-point loss (see surface_loss.TrackedCorrespondenceLoss):
        # bound once here, to the surface vertices of the *initial* state (the
        # tracked start-frame pose, before any step), and evaluated on that
        # same vertex prefix every frame -- refinement only appends vertices.
        self.corr_loss = None
        self._corr_surf_idx = None
        if self.surface_loss is not None and getattr(args, "surface_loss_correspondence", False):
            self.corr_loss = TrackedCorrespondenceLoss(
                loss_target,
                use_valid_mask=not getattr(args, "surface_loss_keep_untracked", False),
                exclude_tool_margin=getattr(args, "surface_loss_exclude_tool", None),
                relative=not getattr(args, "surface_loss_correspondence_absolute", False),
            )
            add0 = self.solver._additional_state_0
            n0 = int(add0.active_particle_count.numpy()[0])
            tris0 = add0.tri_indices[:int(add0.active_tri_count.numpy()[0])].numpy()
            surf0 = np.unique(tris0)
            self._corr_surf_idx = surf0[surf0 < n0]
            info = self.corr_loss.bind(
                self.state_0.particle_q[:n0].numpy()[self._corr_surf_idx], self.sim_time
            )
            print(f"[correspondence] bound {info['n_bound']} surface vertices to tracked "
                  f"frame {info['frame']} (bind residual RMS {info['bind_rmse_mm']:.2f} mm, "
                  f"max {info['bind_max_mm']:.2f} mm, unbound {info['n_unbound']})")

        if args.graph_capture:
            self.capture()
        else:
            self.graph = None

    # ------------------------------------------------------------------
    # Controllable capsule
    # ------------------------------------------------------------------
    def _capsule_pose7(self) -> np.ndarray:
        """Live capsule pose as a 7-vector [px, py, pz, qx, qy, qz, qw]."""
        quat = Rotation.from_euler("xyz", self.capsule_rot, degrees=True).as_quat()
        return np.concatenate((self.capsule_pos, quat)).astype(np.float32)

    def _capsule_transform(self) -> wp.transform:
        """World transform of the capsule rigid body from the live pose."""
        pose = self._capsule_pose7()
        return wp.transform(wp.vec3(*pose[:3].tolist()), wp.quat(*pose[3:].tolist()))

    # ------------------------------------------------------------------
    # Recorded tool trajectory
    # ------------------------------------------------------------------
    @staticmethod
    def _surface_sample_points(particles, tet_indices):
        """Vertices of the initial mesh plus the edge midpoints and centroids of
        its boundary triangles.

        Used only for the tool-path start retract: the capsule SDF is sampled at
        these points, so seeding it with face interior points (not just corners)
        lets the retract catch a capsule that punches through a large coarse
        triangle while all three of its vertices stay just outside.
        """
        p = np.asarray(particles, dtype=np.float64)[:, :3]
        try:
            tris = surface_triangles_from_tets(tet_indices)
        except Exception:
            return p
        if len(tris) == 0:
            return p
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        mids = np.concatenate([0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)], axis=0)
        cents = (a + b + c) / 3.0
        return np.concatenate([p, mids, cents], axis=0)

    def _load_tool_trajectory(self, args, octo_verts=None):
        """Load ``tool_trajectory.npz`` (155 tool_to_world transforms, metres,
        PokeFlex world frame) if ``--tool-trajectory`` points at one.

        Sets ``self._tool_traj_active``. When active, ``step()`` samples the
        path by wall-clock ``sim_time`` (plus the ``start_frame`` offset,
        ``_tool_time_offset``) and imposes it on the capsule instead of the
        GUI/keyboard pose.

        ``octo_verts`` (initial soft-body vertices *plus* surface-triangle edge
        midpoints and centroids -- see :meth:`_surface_sample_points`) is used to
        retract the whole path along the tool axis so the poker *starts* clear of
        the mesh (see ``--tool-start-standoff``). At the default rest frame (19)
        the recorded poker pose grazes the coarse crown by a few mm, so the
        retract pulls it back a touch; an earlier ``--start-frame`` needs a
        larger retract.
        """
        self._tool_traj_active = False
        self._tool_frame_idx = 0
        # Trajectory time (seconds, at the 30 fps capture rate) that sim_time 0
        # maps to: the sim starts from tracked frame ``start_frame``, so the
        # recorded tool path is sampled from there, not from its frame 0. Added
        # to every sample time in _sample_tool_pose().
        self._tool_time_offset = self.start_frame / float(self.ep.capture_fps)
        path = getattr(args, "tool_trajectory", None)
        if not path or str(path).lower() in ("", "none", "off"):
            return

        z = np.load(path, allow_pickle=True)
        T = np.asarray(z["tool_transform"], dtype=np.float64)   # (N, 4, 4) T_WT
        times = np.asarray(z["timestamps"], dtype=np.float64)   # (N,) seconds
        if times.ndim != 1 or times.shape[0] != T.shape[0]:
            times = np.arange(T.shape[0], dtype=np.float64) / float(self.fps)
        # Guard against any non-monotonic / duplicate stamps (Slerp needs strict).
        times = np.maximum.accumulate(times)
        eps = 1e-9
        times = times + eps * np.arange(times.shape[0])

        self._tool_traj_T = T
        self._tool_traj_time = times
        self._tool_traj_rot = Rotation.from_matrix(T[:, :3, :3])
        self._tool_traj_pos = T[:, :3, 3].copy()
        self._tool_slerp = Slerp(times, self._tool_traj_rot)
        self._tool_contact_active = (
            np.asarray(z["contact_active"], dtype=bool)
            if "contact_active" in z.files else np.ones(T.shape[0], dtype=bool)
        )

        # The capsule's local long axis is +Z; so is the tool's. Offset the
        # capsule centre back along local +Z so its lower (+Z) cap sits at the
        # tool's rigid tip point (contact_position, a fixed 12.1 mm along +Z
        # from the tool origin at every frame) at the first sample.
        local_z0 = T[0, :3, 2]
        if "contact_position" in z.files:
            cp0 = np.asarray(z["contact_position"], dtype=np.float64)[0]
            tip_along_z = float(np.dot(cp0 - T[0, :3, 3], local_z0))
        else:
            tip_along_z = self.ep.tip_len
        self._tool_center_offset = (
            tip_along_z - (self.capsule_half_height + self.capsule_radius)
        )

        # Retract the whole path along -tool-axis so the poker starts a fixed
        # clearance off the start-frame mesh. At the default rest frame (19) the
        # tool lifted off six frames earlier and the recorded poker pose already
        # clears the reconstructed surface, so the retract is typically ~0. An
        # earlier --start-frame can still have the fitted indenter capsule
        # overlapping the surface at the recorded pose, which launches the free
        # body if imposed verbatim. Default 0 mm = retract just to contact;
        # negative = raw path (no retract).
        self._tool_playback_rate = float(getattr(args, "tool_playback_rate", 1.0))
        self._tool_loop = bool(getattr(args, "tool_loop", False))
        standoff = float(getattr(args, "tool_start_standoff", 0.0)) / 1000.0
        self._tool_start_retract = 0.0
        if octo_verts is not None and len(octo_verts) and standoff >= 0.0:
            for _ in range(40):
                pos0, rot0 = self._sample_tool_pose(0.0)
                d0 = self._capsule_min_distance(pos0, rot0, np.asarray(octo_verts))
                if d0 >= standoff:
                    break
                step = max(standoff - d0, 1.0e-3)
                self._tool_center_offset -= step
                self._tool_start_retract += step
            if self._tool_start_retract > 1.0e-4:
                print(
                    f"tool path retracted {self._tool_start_retract * 1e3:.1f} mm "
                    f"along -tool-axis so the poker starts "
                    f"{standoff * 1e3:.0f} mm clear of the start-frame mesh"
                )
        # By default the recorded pose is imposed on the solver exactly, so the
        # capsule follows the measured path smoothly. --tool-clamp instead runs
        # it through the manual poker's contact-safe conservative-advancement
        # limiter, which caps the approach rate: on a scripted plunge the
        # capsule then lags behind and *snaps* down to the commanded pose the
        # moment the soft body yields -- a visible jump once per poke. (The
        # limiter also needs --gravity -9.81, kept as the default, or the
        # unpinned octopus gets shoved by the sweeping tool.)
        self._tool_verbatim = not bool(getattr(args, "tool_clamp", False))
        self._tool_traj_active = True

        # Device-side tool path for verbatim driving: one pose per solver
        # substep, each sampled by the host from the recorded trajectory at that
        # substep's sub-frame time (see _refresh_tool_path) and copied into
        # body_q by _set_tool_pose_kernel. Refreshed once per frame before the
        # substep loop / CUDA-graph replay, so per-substep tool motion follows
        # the true recorded path -- not a straight chord between frame endpoints
        # -- and still works under -g. Fixed size (--substeps never changes at
        # runtime); seeded with the start pose so a construction-time capture is
        # valid.
        n_sub = max(self.sim_substeps, 1)
        self._tool_pose_seq = wp.zeros(n_sub, dtype=wp.transform)
        self._tool_pose_seq.assign(np.tile(self._tool_pose7(0.0), (n_sub, 1)))

        start_contact = bool(
            self._tool_contact_active[min(self.start_frame, T.shape[0] - 1)]
        )
        span = self._tool_traj_pos.max(0) - self._tool_traj_pos.min(0)
        print(
            f"tool trajectory: {T.shape[0]} frames, "
            f"{times[-1] - times[0]:.2f} s, "
            f"start frame = {self.start_frame} "
            f"(t0 = {self._tool_time_offset:.3f} s, "
            f"{'in contact' if start_contact else 'tool-free'}), "
            f"origin span (mm) = {np.round(span * 1e3, 1)}, "
            f"contact frames = {int(self._tool_contact_active.sum())}, "
            f"playback x{self._tool_playback_rate:g}"
            + (", verbatim" if self._tool_verbatim else ", clamped")
            + (", loop" if self._tool_loop else "")
        )

    def _sample_tool_pose(self, t: float):
        """Capsule (pos, xyz-euler-deg) for a sim-relative trajectory time ``t``
        in seconds. ``t`` is measured from the start of the run; the offset to
        the ``start_frame``-th recorded pose (``_tool_time_offset``) is added
        here, so callers only ever pass ``sim_time * playback_rate``.

        Translation is linearly interpolated, rotation is slerped, and the
        capsule centre is shifted by ``_tool_center_offset`` along the current
        local +Z so the poker tip tracks the measured contact point.
        """
        t = t + self._tool_time_offset
        tt = self._tool_traj_time
        t0, t1 = float(tt[0]), float(tt[-1])
        if self._tool_loop and t1 > t0:
            t = t0 + ((t - t0) % (t1 - t0))
        tc = float(np.clip(t, t0, t1))

        self._tool_frame_idx = int(np.searchsorted(tt, tc))
        self._tool_frame_idx = min(self._tool_frame_idx, tt.shape[0] - 1)

        rot = self._tool_slerp(tc)
        local_z = rot.as_matrix()[:, 2]
        pos = np.array(
            [np.interp(tc, tt, self._tool_traj_pos[:, k]) for k in range(3)],
            dtype=np.float64,
        )
        pos = pos + self._tool_center_offset * local_z
        rot_deg = rot.as_euler("xyz", degrees=True).astype(np.float64)
        return pos, rot_deg

    def _tool_pose7(self, t: float) -> np.ndarray:
        """The recorded path pose at trajectory time ``t`` as a transform
        7-vector [px, py, pz, qx, qy, qz, qw] (warp / scipy quaternion order)."""
        pos, rot_deg = self._sample_tool_pose(t)
        quat = Rotation.from_euler("xyz", rot_deg, degrees=True).as_quat()
        return np.concatenate((pos, quat)).astype(np.float32)

    def _refresh_tool_path(self):
        """Sample the recorded tool path at each substep's sub-frame time and
        load the poses into ``_tool_pose_seq``, the device buffer
        :data:`_set_tool_pose_kernel` reads (verbatim driving).

        Done from the host once per frame, before the substep loop / graph
        replay, so the per-substep motion works identically with and without
        ``-g``. Substep ``k`` gets the true path pose at ``sim_time + k*sim_dt``
        (``np.interp`` + ``Slerp`` against the ~30 fps capture), so ``--substeps``
        genuinely refines the trajectory rather than subdividing a straight
        chord across the frame -- a low ``--fps`` / high ``--substeps`` run then
        follows the same tool path as a high ``--fps`` / low ``--substeps`` one.
        Also syncs the Python-side pose (GUI readout, gap cache) to the last
        substep of the frame.
        """
        rate = self._tool_playback_rate
        n = max(self.sim_substeps, 1)
        self._tool_pose_seq.assign(
            np.stack([
                self._tool_pose7((self.sim_time + k * self.sim_dt) * rate)
                for k in range(n)
            ])
        )
        self.capsule_pos, self.capsule_rot = self._sample_tool_pose(
            (self.sim_time + (n - 1) * self.sim_dt) * rate
        )
        self._capsule_applied_pos = self.capsule_pos.copy()
        self._capsule_applied_rot = self.capsule_rot.copy()

    def _advance_tool(self, t: float):
        """Sample the recorded path at trajectory time ``t`` seconds and drive
        the capsule through the contact-safe conservative-advancement limiter
        (the ``--tool-clamp`` path). Verbatim driving goes through
        :data:`_set_tool_pose_kernel` instead (see :meth:`simulate`).
        """
        pos, rot_deg = self._sample_tool_pose(t)
        self.capsule_pos = pos
        self.capsule_rot = rot_deg
        self._apply_capsule_control()

    def _sync_capsule(self):
        """Push the live capsule pose into the sim state and the render mesh.

        ``body_q`` is written into *both* state buffers so the value is picked
        up regardless of which buffer the (possibly graph-captured) solver
        step consumes as its input this frame.
        """
        pose = self._capsule_pose7().reshape(1, 7)
        self.state_0.body_q.assign(pose)
        self.state_1.body_q.assign(pose)

    def _nudge_capsule(self, d_pos=(0.0, 0.0, 0.0), d_rot=(0.0, 0.0, 0.0)):
        self.capsule_pos = self.capsule_pos + np.asarray(d_pos, dtype=np.float64)
        self.capsule_rot = self.capsule_rot + np.asarray(d_rot, dtype=np.float64)

    def _active_particles_np(self) -> np.ndarray:
        """World positions of the currently-active soft-body particles."""
        n = int(self.solver._additional_state_0.active_particle_count.numpy()[0])
        return self.state_0.particle_q[:max(n, 0)].numpy().astype(np.float64)

    def _capsule_signed_distances(self, pos, rot_deg, pts) -> np.ndarray:
        """Signed distance from every ``pts`` row to the capsule surface for the
        given pose (capsule long axis is local +Z). Negative = inside."""
        if pts.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        R = Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix()  # local->world
        local = (pts - np.asarray(pos, dtype=np.float64)) @ R             # world->local
        hh = self.capsule_half_height
        seg_z = np.clip(local[:, 2], -hh, hh)
        radial = np.hypot(local[:, 0], local[:, 1])
        return np.hypot(radial, local[:, 2] - seg_z) - self.capsule_radius

    def _capsule_min_distance(self, pos, rot_deg, pts) -> float:
        """Smallest signed distance from ``pts`` to the capsule surface."""
        d = self._capsule_signed_distances(pos, rot_deg, pts)
        return float(d.min()) if d.size else np.inf

    def _refresh_capsule_gap(self):
        """Update the displayed nearest-particle gap after a solver step."""
        try:
            self._capsule_gap = self._capsule_min_distance(
                self._capsule_applied_pos, self._capsule_applied_rot,
                self._active_particles_np(),
            )
        except Exception:
            self._capsule_gap = np.inf

    def _apply_capsule_control(self):
        """Advance the imposed capsule pose toward the user's commanded pose,
        but never let it come closer to the soft body than ``contact_standoff``
        (plus a small ``push_rate`` of extra approach per frame), then push the
        pose into the sim state.

        This is a conservative-advancement step: we evaluate the capsule SDF
        against the particles at the requested pose and, if it violates the
        floor, bisect back along the interpolation from the last applied pose.
        """
        prev_pos = np.asarray(self._capsule_applied_pos, dtype=np.float64)
        prev_rot = np.asarray(self._capsule_applied_rot, dtype=np.float64)

        want_pos = np.asarray(self.capsule_pos, dtype=np.float64)
        want_rot = np.asarray(self.capsule_rot, dtype=np.float64)

        # Rate-limit rotation (tighter once we are in contact).
        d_rot = want_rot - prev_rot
        max_d = self.capsule_rot_clamp_near if (
            np.isfinite(self._capsule_gap)
            and self._capsule_gap < 5.0 * self._contact_d1
        ) else self.capsule_rot_clamp_far
        biggest = float(np.max(np.abs(d_rot))) if d_rot.size else 0.0
        if biggest > max_d > 0.0:
            want_rot = prev_rot + d_rot * (max_d / biggest)

        pts = self._active_particles_np()

        d_prev = self._capsule_min_distance(prev_pos, prev_rot, pts)

        if d_prev < 0.0:
            # The soft body rebounded into the capsule. Ignore the user's
            # command for this frame and climb the SDF gradient (numerically)
            # to push the capsule straight back out to the standoff.
            eps = 0.25 * self._contact_d1
            grad = np.array([
                self._capsule_min_distance(prev_pos + d, prev_rot, pts)
                - self._capsule_min_distance(prev_pos - d, prev_rot, pts)
                for d in (np.array([eps, 0, 0]), np.array([0, eps, 0]), np.array([0, 0, eps]))
            ])
            gn = float(np.linalg.norm(grad))
            retreat = (grad / gn) * (self.capsule_contact_standoff - d_prev) if gn > 1e-12 \
                else np.array([0.0, 0.0, self.capsule_contact_standoff - d_prev])
            new_pos = prev_pos + retreat
            new_rot = prev_rot
            self.capsule_pos = new_pos
            self.capsule_rot = new_rot
            self._capsule_applied_pos = new_pos.copy()
            self._capsule_applied_rot = new_rot.copy()
            self._capsule_gap = self._capsule_min_distance(new_pos, new_rot, pts)
            self._sync_capsule()
            return

        d_want = self._capsule_min_distance(want_pos, want_rot, pts)

        # Closest approach the new pose may reach this frame: down to the
        # standoff when clear, or a bounded `push_rate` deeper when already
        # leaning on the body -- but never past actual contact (d = 0).
        floor = max(
            min(self.capsule_contact_standoff, d_prev - self.capsule_push_rate),
            0.0,
        )

        if d_want >= floor or d_want >= d_prev:
            # Requested pose is safe (or actually backs away): take it as-is.
            new_pos, new_rot = want_pos, want_rot
        else:
            # Bisect the interpolation prev -> want for the furthest fraction
            # whose closest approach still respects the floor.
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                d_mid = self._capsule_min_distance(
                    prev_pos + mid * (want_pos - prev_pos),
                    prev_rot + mid * (want_rot - prev_rot),
                    pts,
                )
                if d_mid >= floor:
                    lo = mid
                else:
                    hi = mid
            new_pos = prev_pos + lo * (want_pos - prev_pos)
            new_rot = prev_rot + lo * (want_rot - prev_rot)

        self.capsule_pos = new_pos
        self.capsule_rot = new_rot
        self._capsule_applied_pos = new_pos.copy()
        self._capsule_applied_rot = new_rot.copy()
        self._capsule_gap = self._capsule_min_distance(new_pos, new_rot, pts)
        self._sync_capsule()

    def reset_capsule(self):
        self.capsule_pos = self._capsule_pos_default.copy()
        self.capsule_rot = self._capsule_rot_default.copy()

    def _capsule_gui_callback(self):
        psim.TextUnformatted("Capsule control")
        psim.Separator()

        if self._tool_traj_active:
            n = self._tool_traj_time.shape[0]
            in_contact = bool(self._tool_contact_active[min(self._tool_frame_idx, n - 1)])
            psim.TextUnformatted(
                f"tool trajectory: frame {self._tool_frame_idx + 1}/{n}  "
                f"t={self.sim_time * self._tool_playback_rate:.2f}s  "
                f"{'contact' if in_contact else 'free'}"
                f"{'  [verbatim]' if self._tool_verbatim else '  [clamped]'}"
            )
            changed_rate, new_rate = psim.DragFloat(
                "playback rate", self._tool_playback_rate, 0.01, 0.05, 4.0, "%.2f"
            )
            if changed_rate:
                self._tool_playback_rate = float(new_rate)
            clamp_on = not self._tool_verbatim
            changed_clamp, clamp_on = psim.Checkbox("contact-safe clamp", clamp_on)
            if changed_clamp:
                self._tool_verbatim = not clamp_on
            _, self._tool_loop = psim.Checkbox("loop", self._tool_loop)
            if getattr(self, "_tool_start_retract", 0.0) > 1e-4:
                psim.TextUnformatted(
                    f"path retracted {self._tool_start_retract * 1e3:.1f} mm "
                    f"along -tool-axis (starts clear of coarse mesh)"
                )
            psim.TextUnformatted("pose below is driven by the path (edits ignored)")
            psim.Separator()

        s = self.capsule_move_step
        r = self.capsule_rot_step

        changed_p, new_p = psim.DragFloat3(
            "position", tuple(self.capsule_pos), 0.5 * s
        )
        if changed_p:
            self.capsule_pos = np.asarray(new_p, dtype=np.float64)

        changed_r, new_r = psim.DragFloat3(
            "rotation (deg)", tuple(self.capsule_rot), 1.0
        )
        if changed_r:
            self.capsule_rot = np.asarray(new_r, dtype=np.float64)

        psim.TextUnformatted(f"move step: {s:.4g}   rot step: {r:g} deg")

        # Translation nudges.
        if psim.Button("-X"):
            self._nudge_capsule(d_pos=(-s, 0.0, 0.0))
        psim.SameLine()
        if psim.Button("+X"):
            self._nudge_capsule(d_pos=(s, 0.0, 0.0))
        psim.SameLine()
        if psim.Button("-Y"):
            self._nudge_capsule(d_pos=(0.0, -s, 0.0))
        psim.SameLine()
        if psim.Button("+Y"):
            self._nudge_capsule(d_pos=(0.0, s, 0.0))
        psim.SameLine()
        if psim.Button("-Z"):
            self._nudge_capsule(d_pos=(0.0, 0.0, -s))
        psim.SameLine()
        if psim.Button("+Z"):
            self._nudge_capsule(d_pos=(0.0, 0.0, s))

        # Rotation nudges.
        if psim.Button("roll -"):
            self._nudge_capsule(d_rot=(-r, 0.0, 0.0))
        psim.SameLine()
        if psim.Button("roll +"):
            self._nudge_capsule(d_rot=(r, 0.0, 0.0))
        psim.SameLine()
        if psim.Button("pitch -"):
            self._nudge_capsule(d_rot=(0.0, -r, 0.0))
        psim.SameLine()
        if psim.Button("pitch +"):
            self._nudge_capsule(d_rot=(0.0, r, 0.0))
        psim.SameLine()
        if psim.Button("yaw -"):
            self._nudge_capsule(d_rot=(0.0, 0.0, -r))
        psim.SameLine()
        if psim.Button("yaw +"):
            self._nudge_capsule(d_rot=(0.0, 0.0, r))

        _, self.capsule_move_step = psim.DragFloat(
            "button move step", self.capsule_move_step, 1e-4, 1e-5, 10.0
        )
        _, self.capsule_rot_step = psim.DragFloat(
            "button rot step", self.capsule_rot_step, 0.1, 0.1, 90.0
        )
        _, self.capsule_move_speed = psim.DragFloat(
            "key move speed /s", self.capsule_move_speed, 1e-3, 0.0, 100.0
        )
        _, self.capsule_rot_speed = psim.DragFloat(
            "key rot speed deg/s", self.capsule_rot_speed, 1.0, 0.0, 720.0
        )

        if psim.Button("reset capsule"):
            self.reset_capsule()
        psim.SameLine()
        _, self.capsule_visible = psim.Checkbox("show capsule", self.capsule_visible)

        psim.Separator()
        gap_txt = "n/a" if not np.isfinite(self._capsule_gap) else f"{self._capsule_gap:+.2e}"
        psim.TextUnformatted(f"gap to soft body: {gap_txt}   (d1 = {self._contact_d1:.1e})")
        psim.TextUnformatted("contact-safe: capsule can't be driven into the body")
        _, self.capsule_contact_standoff = psim.DragFloat(
            "standoff", self.capsule_contact_standoff, 1e-5, 0.0, 1.0e-2, "%.2e"
        )
        _, self.capsule_push_rate = psim.DragFloat(
            "max push / frame", self.capsule_push_rate, 1e-5, 0.0, 1.0e-2, "%.2e"
        )

        psim.TextUnformatted("hold: WASD = X/Y, Q/E = Z, arrows = rotate, R = reset")

        self._handle_capsule_hotkeys()

    def _handle_capsule_hotkeys(self):
        """Move the capsule smoothly for as long as a hotkey is held down.

        Uses ``IsKeyDown`` (level, not edge) and scales every increment by the
        frame delta time so the motion is smooth and frame-rate independent.
        """
        # Ignore keys while an ImGui text field has focus.
        try:
            if psim.GetIO().WantCaptureKeyboard:
                return
            dt = float(psim.GetIO().DeltaTime)
        except Exception:
            dt = 1.0 / float(self.fps)
        if not (dt > 0.0) or dt > 0.25:
            dt = 1.0 / float(self.fps)

        def down(key):
            try:
                return psim.IsKeyDown(key)
            except Exception:
                return False

        move = self.capsule_move_speed * dt
        rot = self.capsule_rot_speed * dt

        d_pos = np.zeros(3)
        d_rot = np.zeros(3)
        if down(psim.ImGuiKey_D):
            d_pos[0] += move
        if down(psim.ImGuiKey_A):
            d_pos[0] -= move
        if down(psim.ImGuiKey_W):
            d_pos[1] += move
        if down(psim.ImGuiKey_S):
            d_pos[1] -= move
        if down(psim.ImGuiKey_E):
            d_pos[2] += move
        if down(psim.ImGuiKey_Q):
            d_pos[2] -= move
        if down(psim.ImGuiKey_RightArrow):
            d_rot[2] += rot
        if down(psim.ImGuiKey_LeftArrow):
            d_rot[2] -= rot
        if down(psim.ImGuiKey_UpArrow):
            d_rot[1] += rot
        if down(psim.ImGuiKey_DownArrow):
            d_rot[1] -= rot

        if d_pos.any() or d_rot.any():
            self._nudge_capsule(d_pos=d_pos, d_rot=d_rot)

        # Reset stays edge-triggered so it fires once per tap.
        try:
            if psim.IsKeyPressed(psim.ImGuiKey_R, False):
                self.reset_capsule()
        except Exception:
            pass

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
                # simulate() ends with self.state_0 on the buffer holding this
                # frame's result and self.state_1 on the other, after N trace-
                # time rebinds. step() then does exactly ONE more swap after
                # each replay, so which physical buffer self.state_0 names
                # alternates frame to frame -- and for N != 1 the "other"
                # buffer is a whole substep (sim_dt) stale, which shows up as
                # the recorded soft body lagging the tool (apparent contact
                # penetration) on every other frame. Mirror particle_q/qd into
                # the other buffer so both hold the latest state and it no
                # longer matters which one step()'s swap lands on; the graph's
                # baked "read-from" buffer is then always current too. (For
                # N == 1 this is the original odd-substeps fixup; body_q is
                # already written into both buffers by simulate()'s per-substep
                # kernel launches, so it does not need mirroring -- and copying
                # it here instead destabilises the solve.)
                wp.copy(self.state_1.particle_q, self.state_0.particle_q)
                wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.graph = capture.graph
        else:
            self.graph = None


    def simulate(self):
        for k in range(self.sim_substeps):
            # Advance the recorded tool path across the substeps so the poker
            # moves in N gradual increments over the frame instead of one jump
            # (raising --substeps then directly smooths a fast plunge and eases
            # the contact barrier).
            if self._tool_traj_active:
                if self._tool_verbatim:
                    # Device-side copy of the host-sampled per-substep pose ->
                    # works inside the captured CUDA graph. k is baked into each
                    # captured launch; _tool_pose_seq is refreshed from the host
                    # once per frame in step() (_refresh_tool_path).
                    wp.launch(
                        _set_tool_pose_kernel,
                        dim=1,
                        inputs=[self._tool_pose_seq, k],
                        outputs=[self.state_0.body_q],
                    )
                elif not self.do_capture:
                    # --tool-clamp limiter needs Python; only in the non-graph
                    # path (graph + clamp drives once per frame in step()).
                    self._advance_tool(
                        (self.sim_time + k * self.sim_dt) * self._tool_playback_rate
                    )

            with nvtx.annotate("solver_step", color="blue"):
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )

            self.state_0, self.state_1 = self.state_1, self.state_0

    @nvtx.annotate("Solver Frame", color="green")
    def step(self):
        if self._tool_traj_active and self._tool_verbatim:
            # Sample this frame's per-substep tool poses from the recorded path;
            # simulate() copies pose k into body_q each substep (inside the CUDA
            # graph when -g).
            self._refresh_tool_path()
        elif self._tool_traj_active and self.do_capture:
            # --tool-clamp + CUDA graph: the limiter needs Python, so run it
            # once per frame here, before the replay.
            self._advance_tool(self.sim_time * self._tool_playback_rate)
        elif not self._tool_traj_active:
            # Manual GUI/keyboard poker: advance the imposed pose toward the
            # commanded one within the contact-safe limiter, once per frame.
            self._apply_capsule_control()
        # else: --tool-clamp + no graph -> simulate() advances per substep.

        if self.graph:
            wp.capture_launch(self.graph)
            self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()

        # Cache the post-step gap so next frame's motion can be rate-limited.
        self._refresh_capsule_gap()

        self.sim_time += self.frame_dt

    def check_bail(self):
        """Return a short reason string if the run should stop early (mesh
        blowing up, or the poker punching clean through the contact barrier),
        else ``None``. No-op unless ``--bail-unstable`` was passed.

        Called once per frame by :func:`run` after :meth:`step`; the three
        symptoms it watches for:

        * non-finite particle positions (the solve diverged);
        * the active-mesh AABB diagonal past ``--bail-bbox-ratio`` x the rest
          pose, or any vertex jumping more than ``--bail-max-disp`` in a single
          frame (numerical blow-up);
        * the nearest particle more than ``--bail-penetration`` *inside* the
          capsule surface (contact barrier defeated -- a legitimate deep poke
          keeps this gap near zero).
        """
        if not self._bail_enabled or self._bail_reason is not None:
            return self._bail_reason

        try:
            pts = self._active_particles_np()
        except Exception:
            return None
        if pts.size == 0:
            return None

        reason = None
        if not np.isfinite(pts).all():
            reason = "non-finite particle positions"
        else:
            diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            if (self._rest_bbox_diag > 0.0
                    and diag > self._bail_bbox_ratio * self._rest_bbox_diag):
                reason = (f"mesh AABB diag {diag * 1e3:.0f} mm > "
                          f"{self._bail_bbox_ratio:g}x rest "
                          f"({self._rest_bbox_diag * 1e3:.0f} mm)")
            elif (self._prev_active_pts is not None
                  and self._prev_active_pts.shape == pts.shape):
                jump = float(np.linalg.norm(pts - self._prev_active_pts, axis=1).max())
                if jump > self._bail_max_disp:
                    reason = (f"vertex moved {jump * 1e3:.0f} mm in one frame "
                              f"(> {self._bail_max_disp * 1e3:.0f} mm)")

        if reason is None and np.isfinite(self._capsule_gap) \
                and self._capsule_gap < -self._bail_penetration:
            reason = (f"capsule punched {-self._capsule_gap * 1e3:.0f} mm into "
                      f"the mesh (> {self._bail_penetration * 1e3:.0f} mm)")

        self._prev_active_pts = pts
        self._bail_reason = reason
        return reason

    def render(self):
        # Lots of device to host copy but it is just in the rendering so I can ignore this part when presenting timings
        particle_count = self.solver._additional_state_0.active_particle_count.numpy()[0]
        tet_count = self.solver._additional_state_0.active_tet_count.numpy()[0]
        tri_count = self.solver._additional_state_0.active_tri_count.numpy()[0]

        if self._old_vertex_count != particle_count:
            print(f"new tet count {tet_count}, new tri count {tri_count}, new vert count {particle_count}")
        ps.remove_surface_mesh("Soft body")
        self.ps_volume = ps.register_surface_mesh("Soft body", self.state_0.particle_q[:particle_count].numpy(), self.solver._additional_state_0.tri_indices[:tri_count].numpy())
        self.ps_volume.set_edge_width(1.0)
        self.ps_volume.set_edge_color([0.0, 0.0, 0.0])
        if self._soft_body_alpha < 1.0:
            # See-through octopus so a buried poker capsule stays visible.
            self.ps_volume.set_transparency(self._soft_body_alpha)
        self._old_vertex_count = particle_count

        capsule_tf = self.state_0.body_q.numpy()[0]
        capsule_mat = np.eye(4)
        capsule_mat[:3, :3] = Rotation.from_quat(capsule_tf[3:7]).as_matrix()
        capsule_mat[:3, 3] = capsule_tf[:3]
        self.ps_capsule.set_transform(capsule_mat)
        self.ps_capsule.set_enabled(self.capsule_visible)

        # Thin shaft marker: from the capsule's back cap, extend up the tool
        # axis (-local_z, away from the poke) so it pokes out of the mesh.
        local_z = capsule_mat[:3, 2]
        back_cap = capsule_mat[:3, 3] - (
            self.capsule_half_height + self.capsule_radius
        ) * local_z
        shaft = np.array(
            [back_cap - 0.02 * local_z, back_cap - 0.14 * local_z], dtype=np.float32
        )
        self.ps_tool_shaft.update_node_positions(shaft)
        self.ps_tool_shaft.set_enabled(self.capsule_visible)

        # Advance the tracked-surface overlay to match the current sim time.
        if self.tracked_surface.active:
            self.tracked_surface.update_for_time(self.sim_time)

    def evaluate_surface_loss(self):
        """Score this frame's sim surface against the tracked mesh and append
        the result to ``self._loss_rows`` (no-op unless ``--surface-loss``).

        Only the *surface* vertices (those referenced by the active surface
        triangulation) are compared -- interior tet vertices are never near the
        tracked surface and would inflate the MSE. Returns the row dict, or
        ``None`` when the loss is disabled.
        """
        if self.surface_loss is None:
            return None
        add0 = self.solver._additional_state_0
        n = int(add0.active_particle_count.numpy()[0])
        tri_ct = int(add0.active_tri_count.numpy()[0])
        tris = add0.tri_indices[:tri_ct].numpy()
        surf = np.unique(tris)
        surf = surf[surf < n]
        pts = self.state_0.particle_q[:n].numpy()[surf]
        tool = lambda P: self._capsule_signed_distances(
            self._capsule_applied_pos, self._capsule_applied_rot, np.asarray(P, dtype=np.float64)
        )
        local_faces = np.searchsorted(surf, tris[(tris < n).all(axis=1)])
        row = self.surface_loss.evaluate(pts, self.sim_time, tool_signed_distance=tool,
                                         sim_faces=local_faces)
        if self.corr_loss is not None:
            pts0 = self.state_0.particle_q[:n].numpy()[self._corr_surf_idx]
            merge_correspondence_row(
                row, self.corr_loss.evaluate(pts0, self.sim_time, tool_signed_distance=tool))
        row["time"] = float(self.sim_time)

        # Tool-submersion diagnostics: how far the poker capsule has sunk into
        # the soft body this frame. capsule_gap = min signed particle->capsule
        # distance (<0 => a particle is inside the capsule); pen_* summarise the
        # set of particles that are inside. A clean poke keeps the barrier
        # holding so capsule_gap stays ~>= 0; a run where the tool submerges
        # drives it negative. Consumed by mfem.refinement.sweep_octopus to
        # penalise such runs.
        cap_d = self._capsule_signed_distances(
            self._capsule_applied_pos, self._capsule_applied_rot,
            self._active_particles_np(),
        )
        if cap_d.size:
            pen = np.clip(-cap_d, 0.0, None)
            row["capsule_gap"] = float(cap_d.min())
            row["capsule_pen_count"] = int((cap_d < 0.0).sum())
            row["capsule_pen_sum_mm"] = float(pen.sum() * 1e3)
            row["capsule_pen_max_mm"] = float(pen.max() * 1e3)
        else:
            row["capsule_gap"] = float("inf")
            row["capsule_pen_count"] = 0
            row["capsule_pen_sum_mm"] = 0.0
            row["capsule_pen_max_mm"] = 0.0

        self._loss_rows.append(row)
        return row


    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Which PokeFlex tracked episode to simulate. Options whose default is
        # episode-specific are given a None sentinel below and resolved by
        # resolve_episode_defaults() after parsing.
        add_episode_arg(parser)

        # --headless comes from newton.examples.create_parser(); init() maps it
        # onto polyscope's mock render backend (no window, no-op scene, frames
        # not throttled to --fps) for sweeps and batch --record runs.
        parser.add_argument(
            "--record-energy",
            help="Record energy during simulation",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--dimx",
            help="Dimension X",
            type=int,
            default=15,
        )

        parser.add_argument(
            "--dimy",
            help="Dimension Y",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--dimz",
            help="Dimension Z",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--energy",
            help="Which energy model to use (arap, neohookean)",
            type=str,
            default="arap",
            choices=["arap", "neohookean"],
        )

        parser.add_argument(
            "--mu",
            help="Lame's mu (shear modulus), Pa. Default = mu(E, nu) for the "
                 "selected --episode's identified material.",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--lmbda",
            help="Lame's lambda, Pa. Default = lambda(E, nu) for the selected "
                 "--episode's identified material. Note: ARAP energy ignores "
                 "this. Only used when --per-tet-material is 'none'.",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--shell-thickness",
            help="Membrane shell thickness, m. The surface tris get ARAP "
                 "membrane elements with surface modulus mu(E, nu) * this "
                 "thickness. The identification never fit a shell (all shell "
                 "params in the episode JSONs are 0), so the moduli fall back "
                 "to the identified bulk material; the default thickness "
                 "(2 mm) is a plush-fabric guess. Pass 0 to disable shell "
                 "elements.",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--shell-mu",
            help="Surface shear modulus of the shell, Pa*m, overriding "
                 "mu * --shell-thickness.",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--per-tet-material",
            help="Path to the bundled identified stiffness-field .npz "
                 "(identification mesh + per-tet Young's modulus). Its per-tet E "
                 "is transferred onto this sim's coarse mesh by nearest-centroid "
                 "inverse-distance blend and converted to per-tet Lame "
                 "mu/lambda, overriding --mu / --lmbda. Pass 'none' for a single "
                 "homogeneous --mu / --lmbda. Default: the selected --episode's "
                 "models/<episode>_stiffness_field.npz.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--per-tet-material-kind",
            help="Which identified stiffness field to transfer: 'field' "
                 "(quasi-static, E 1542-5723 Pa), 'dyn' (viscoelastic dyn-fit), "
                 "or 'sota' (homogeneous SOTA fit).",
            type=str,
            default="field",
            choices=["field", "dyn", "sota"],
        )

        parser.add_argument(
            "--density",
            help="Mass density, kg/m^3. Default = the selected --episode's "
                 "identified rho (80 for all four).",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--gravity",
            help="Gravitational acceleration along -Y (PokeFlex up). Default "
                 "-9.81: the free octopus needs its weight to stay put under "
                 "the laterally-sweeping tool. Pass 0 for the reference "
                 "quasi-static regime (the unpinned body may then slide).",
            type=float,
            default=-9.81,
        )

        parser.add_argument(
            "-l",
            "--line-search",
            help="Whether to use line search",
            action="store_true",
        )

        parser.add_argument(
            "-g",
            "--graph-capture",
            help="Whether to capture a Cuda graph",
            action="store_true",
        )

        parser.add_argument(
            "-p",
            "--preconditioner",
            help="Use the block-Jacobi preconditioner for the global CG solve",
            action="store_true",
        )

        parser.add_argument(
            "-r",
            "--refine",
            help="Enable adaptive mesh refinement during the sim",
            action="store_true",
        )

        parser.add_argument(
            "--iterations",
            help="SQP iterations per time step",
            type=int,
            default=12,
        )

        parser.add_argument(
            "--substeps",
            help="Solver steps per frame. With a recorded tool path the pose is "
                 "resampled from the true path at each substep's sub-frame time, "
                 "so raising this makes the poker move gradually within a frame "
                 "(smoother, eases the contact barrier) and lets a low --fps run "
                 "track the trajectory as tightly as a high --fps one.",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--fps",
            help="Frames per second. Default = the --episode capture rate "
                 "(30 for all four; forward-sim frame_dt = 0.03333 s).",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--mesh",
            choices=tuple(OCTO_MESH_PATHS),
            default=OCTO_MESH_DEFAULT,
            help="Initial tet-mesh resolution (default coarse): coarse ~600 v / "
                 "~2000 tets, medium ~980 v / ~3000 tets, fine ~6400 v / "
                 "~19100 tets. All three are fTetWild'd from the --episode's "
                 "rest frame (models/<episode>_initial_{coarse,medium,full}.npz), "
                 "so --start-frame is unaffected.",
        )

        parser.add_argument(
            "--start-frame",
            help="Tracked episode frame the run starts from and uses as the "
                 "stress-free rest state (default: the --episode's initial_frame "
                 "= the relaxed post-poke frame whose tracked surface reads "
                 "cleanest). The initial tet mesh "
                 "models/<episode>_initial_*.npz is fTetWild'd from this frame, "
                 "so the recorded tool path and the tracked-surface overlay are "
                 "both advanced to it at sim_time 0. Only change it together "
                 "with the mesh (python -m "
                 "mfem.refinement.models.make_octopus_tetmesh --episode X "
                 "--frame N).",
            type=int,
            default=None,
        )

        parser.add_argument(
            "--rest-shape",
            default="auto",
            help="Stress-free reference configuration for the elastic energy. "
                 "The tracked initial mesh is itself a gravity-loaded (and "
                 "slightly poke-dented) shape, so with 'none' the solver treats "
                 "that deformed state as zero strain and the octopus sags "
                 "further under gravity. 'auto' (default) loads "
                 "models/octopus_rest_<mesh>.npz if present -- an un-sagged rest "
                 "whose gravity + table equilibrium reproduces the tracked "
                 "shape -- else falls back to 'none' with a hint. Or give an "
                 "explicit .npz path (needs a 'rest_particles' array). Build it "
                 "with 'python -m mfem.refinement.models.make_octopus_rest'. "
                 "The sim still STARTS from the tracked mesh either way.",
        )

        # ---- Recorded tool trajectory -------------------------------------
        parser.add_argument(
            "--tool-trajectory",
            help="Path to a tool_trajectory.npz (tool_to_world transforms in "
                 "the PokeFlex world frame). Drives the poker capsule along the "
                 "measured path instead of the GUI/keyboard. Default: the "
                 "selected --episode's recorded path. Pass 'none' to disable.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--tool-playback-rate",
            help="Trajectory seconds advanced per sim second (1.0 = real time / "
                 "1 trajectory frame per sim frame at --fps 30). Lower it if "
                 "the contact barrier spikes on fast plunges.",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--tool-start-standoff",
            help="Retract the whole recorded path along the tool axis so the "
                 "poker starts this many mm clear of the start-frame mesh. "
                 "Starting from frame 19 (--start-frame), well after the tool "
                 "lifts off, the recorded poker pose already clears the "
                 "reconstructed rest surface, so the retract is usually ~0. If a "
                 "different --start-frame leaves the fitted indenter capsule "
                 "(~30 mm dia) overlapping the surface, imposing that verbatim "
                 "flings the free body. Default 0 = retract just until the "
                 "capsule kisses the mesh. Negative = raw path, no retract (may "
                 "fling).",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--tool-loop",
            help="Loop the tool trajectory when it ends (default: hold last pose)",
            action="store_true",
        )
        parser.add_argument(
            "--tool-clamp",
            help="Route the recorded tool pose through the manual poker's "
                 "contact-safe conservative-advancement limiter instead of "
                 "imposing it exactly (the default). Safer against barrier "
                 "spikes, but the limiter lags a fast scripted plunge and then "
                 "snaps the capsule down to the commanded pose when the body "
                 "yields -- a visible jump once per poke.",
            action="store_true",
        )
        parser.add_argument(
            "--soft-body-alpha",
            help="Opacity of the octopus render mesh (1.0 = opaque). Set below "
                 "1 (e.g. 0.4) to see the poker capsule while it is buried "
                 "inside the mesh during a deep poke.",
            type=float,
            default=1.0,
        )

        # Dataset surface-tracking overlay (see mfem.refinement.tracked_surface).
        # add_cli_args defaults --tracked-surface to the octopus trajectory;
        # override to a None sentinel so resolve_episode_defaults() can point it
        # at the selected --episode's mesh_trajectories_canonical.npy instead.
        TrackedSurfaceOverlay.add_cli_args(parser)
        parser.set_defaults(tracked_surface=None)

        # ---- Surface-tracking loss (mfem.refinement.surface_loss) ---------
        parser.add_argument(
            "--surface-loss",
            help="Each frame, measure the mean squared distance from the sim "
                 "surface vertices to the nearest point on the tracked mesh "
                 "(the --tracked-surface overlay, same frame mapping). Printed "
                 "every --surface-loss-every frames and summarised at exit; "
                 "--surface-loss-out saves the per-frame curve.",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-every",
            help="Print the running surface loss every N frames (still "
                 "accumulated every frame).",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--surface-loss-no-project",
            help="Measure distance to the nearest tracked vertex instead of the "
                 "nearest point on the tracked triangles (faster, ~1.7 mm bias).",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-keep-untracked",
            help="Include untracked tracked vertices / faces in the target "
                 "(default: drop them via valid_mask_canonical.npy).",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-symmetric",
            help="Also measure tracked -> sim and report the mean of both "
                 "directions (Chamfer-style).",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-correspondence",
            help="Also report the material-point loss: each sim surface vertex "
                 "is paired once, at the start, with the closest point on the "
                 "tracked template and thereafter compared with where that "
                 "point went (the template has fixed topology, so this is a "
                 "true Lagrangian error that also sees tangential motion). "
                 "Scored on the initial surface vertices only, so refinement "
                 "cannot move it by adding vertices. Added to the curve as "
                 "corr_* columns (corr_mse, corr_rmse_mm, corr_max_mm, ...).",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-correspondence-absolute",
            help="Correspondence loss on raw positions instead of displacement "
                 "from the first frame (the bind residual then becomes a floor).",
            action="store_true",
        )
        parser.add_argument(
            "--surface-loss-reference",
            help="Score the surface loss against this --record .npz (e.g. a "
                 "converged --mesh fine run of the same episode without "
                 "refinement) instead of the tracked dataset surface. Isolates "
                 "discretisation error from modelling/tracking error; the "
                 "reference sees under the tool, so --surface-loss-exclude-tool "
                 "is unnecessary with it.",
            type=str,
            default=None,
            metavar="REC.npz",
        )
        parser.add_argument(
            "--surface-loss-exclude-tool",
            help="Treat tracked vertices closer than this (m; negative = inside) "
                 "to the poker capsule as untracked and leave sim vertices that "
                 "close out of the sim -> tracked mean. The tracked mesh cannot "
                 "see under the poker and interpolates through it, so without "
                 "this the loss charges the sim for the dimple it gets right, "
                 "in proportion to how many vertices refinement puts there. "
                 "Also applies to the tracked -> sim direction.",
            type=float,
            default=None,
            metavar="MARGIN_M",
        )
        parser.add_argument(
            "--surface-loss-out",
            help="Write the per-frame surface-loss curve to this .npz "
                 "(keys: time, frame, mse, rmse_mm, max_mm, ...).",
            type=str,
            default=None,
        )

        # ---- Early-out watchdog (see MFEMRefinementSim.check_bail) --------
        parser.add_argument(
            "--bail-unstable",
            help="Stop the run early if the mesh goes unstable (non-finite "
                 "positions, AABB past --bail-bbox-ratio x rest, or a vertex "
                 "jumping more than --bail-max-disp in one frame) or the poker "
                 "capsule punches more than --bail-penetration through the "
                 "contact barrier. The recording / surface-loss curve gathered "
                 "so far are still written. Handy for parameter sweeps.",
            action="store_true",
        )
        parser.add_argument(
            "--bail-max-disp",
            help="--bail-unstable: max single-frame vertex displacement, m.",
            type=float,
            default=0.15,
        )
        parser.add_argument(
            "--bail-bbox-ratio",
            help="--bail-unstable: bail when the active-mesh AABB diagonal "
                 "exceeds this multiple of the rest-pose diagonal.",
            type=float,
            default=4.0,
        )
        parser.add_argument(
            "--bail-penetration",
            help="--bail-unstable: bail when the nearest particle is more than "
                 "this far inside the capsule surface, m (a real deep poke "
                 "keeps this near zero).",
            type=float,
            default=0.015,
        )

        parser.add_argument(
            "--refine-every",
            help="Evaluate refinement candidates every N solver steps",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--max-new-vertices",
            help="Maximum number of new vertices created per refinement pass",
            type=int,
            default=32,
        )

        # ---- Refinement headroom (padded buffer ceilings) -------------------
        # Each is a multiple of the mesh's own particle/tet/surface-tri count
        # (so the caller doesn't need to know those counts) and defaults to
        # the ceiling baked into the .npz at mesh-conversion time (see
        # make_tetmesh_models.py / make_octopus_tetmesh.py); pass one of these
        # to override it.
        parser.add_argument(
            "--max-particles-mult",
            help="Padded particle-buffer ceiling as a multiple of the mesh's particle count (default: baked into the .npz)",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--max-tets-mult",
            help="Padded tet-buffer ceiling as a multiple of the mesh's tet count (default: baked into the .npz)",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--max-tris-mult",
            help="Padded surface-tri ceiling as a multiple of the mesh's surface-tri count (default: baked into the .npz)",
            type=float,
            default=None,
        )

        parser.add_argument(
            "--record",
            help="Record the active soft-body mesh (tet indices + vertex "
                 "positions) and the poker pose at every frame to a .npz file "
                 "(replay it with replay_octopus.py). Defaults to "
                 "simulation.npz if the flag is given without a path.",
            nargs="?",
            const="simulation.npz",
            default=None,
        )

        parser.add_argument(
            "--record-frames",
            help="Stop after recording this many frames (0 = until the window is closed)",
            type=int,
            default=0,
        )

        # ---- Solver tunables (forwarded to RefinementSolver) ---------------
        # Refinement candidate scoring / selection.
        parser.add_argument(
            "--refine-density",
            help="Density used for mass recomputation during a refine pass "
                 "(default: the --episode identified rho).",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--refine-tet-score-weight",
            help="Weight on per-tet elastic energy in the edge refinement score (0 disables it)",
            type=float,
            default=100.0,
        )
        parser.add_argument(
            "--refine-vertex-score-weight",
            help="Weight on the min endpoint vertex score (contact barrier energy) in the edge refinement score",
            type=float,
            default=200.0,
        )
        parser.add_argument(
            "--penetrating-edge-score",
            help="Score forced onto edges whose midpoint penetrates a rigid shape (always beats the threshold)",
            type=float,
            default=1.0e6,
        )
        parser.add_argument(
            "--refine-conflict-iterations",
            help="Number of conflict-resolution sweeps for parallel edge-split selection",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--refine-split-position",
            help="Parametric position of the new vertex along a split edge (0.5 = midpoint)",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--refine-hashmap-load-factor",
            help="Load factor for sizing the refinement candidate hash table",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--refine-hashmap-edges-per-tet",
            help="Estimated edges per tet for sizing the refinement candidate hash table",
            type=int,
            default=6,
        )
        parser.add_argument(
            "--refine-scoring",
            help="Edge refinement scoring: 'legacy' (energy sum over incident tets + "
                 "midpoint-penetration override) or 'geometric' (dimensionless per-edge "
                 "score from edge length, longest-edge ratio, elastic energy density and "
                 "tri/vertex contact proximity; see refinement.populate_candidates_geometric)",
            choices=["legacy", "geometric"],
            default="legacy",
        )
        parser.add_argument(
            "--refine-min-edge-length",
            help="geometric scoring: shortest edge refinement may create, m (edges "
                 "shorter than twice this are never split). Default: --contact-d1",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--refine-elastic-weight",
            help="geometric scoring: weight on tet strain-energy density / mu",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--refine-contact-weight",
            help="geometric scoring: weight on per-vertex tool-contact proximity",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--refine-tri-contact-weight",
            help="geometric scoring: weight on surface-tri tool-contact proximity",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--refine-geometric-threshold",
            help="geometric scoring: split edges whose score exceeds this (dimensionless; "
                 "an edge of 2x the min length in full contact scores 2 x weight)",
            type=float,
            default=1.0,
        )

        # Global CG solve + block-Jacobi preconditioner.
        parser.add_argument(
            "--cg-max-iterations",
            help="Maximum iterations for the global CG solve",
            type=int,
            default=5000,
        )
        parser.add_argument(
            "--cg-tolerance",
            help="Relative residual tolerance for the global CG solve",
            type=float,
            default=1.0e-6,
        )
        parser.add_argument(
            "--cg-check-every",
            help="Convergence-check cadence for the global CG solve (0 = only at end)",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--preconditioner-singular-threshold",
            help="|det| below which a block-Jacobi preconditioner block falls back to identity",
            type=float,
            default=1.0e-20,
        )

        # Backtracking line search.
        parser.add_argument(
            "--line-search-max-iterations",
            help="Maximum backtracking iterations in the line search",
            type=int,
            default=30,
        )
        parser.add_argument(
            "--line-search-alpha0",
            help="Initial step length for the line search",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--line-search-tau",
            help="Step-shrink factor per backtracking iteration",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--line-search-c",
            help="Armijo sufficient-decrease constant",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--line-search-threshold",
            help="Minimum step / merit-slack tolerance for the line search",
            type=float,
            default=1.0e-8,
        )

        # Attachments.
        parser.add_argument(
            "--attachment-stiffness",
            help="Stiffness of the Dirichlet constraint pinning zero-inv-mass particles",
            type=float,
            default=1.0e1,
        )

        # Contact barrier. Defaults sized for the octopus (metres): the rest
        # pose is only ~0.11 m tall in Y, so the barrier band is a few mm.
        parser.add_argument(
            "--contact-d0",
            help="Inner contact barrier distance; for d <= d0 the log-barrier is replaced by its quadratic extrapolation",
            type=float,
            default=5.0e-4,
        )
        parser.add_argument(
            "--contact-d1",
            help="Outer contact barrier distance; the barrier is zero for d >= d1 and active in (d0, d1)",
            type=float,
            default=2.0e-3,
        )
        parser.add_argument(
            "--contact-stiffness",
            help="Scaling coefficient on the contact barrier energy",
            type=float,
            default=1.0e5,
        )
        parser.add_argument(
            "--no-tri-contact",
            action="store_true",
            help="Vertex-only contact: disable the per-surface-triangle barrier against the capsule tool",
        )

        # Contact friction (lagged / semi-implicit Coulomb friction).
        # Applied per-shape: shape 0 is the poker capsule, shape 1 is the table
        # plane (see the add_shape_capsule / add_shape_plane calls in
        # MFEMRefinementSim.__init__). The fitted PlushOctopus_T1 values are
        # ~0.3 for both (tool/body physics/sota_dx10mm/hybrid_identification
        # .json: 0.297; body/table ~0.3), but at that level the free octopus
        # slides on the table and pops up under the sweeping tool, so the
        # shared default is cranked to OCTO_SIM_FRICTION_MU to keep the body
        # anchored. --friction-mu-tool / --friction-mu-ground override the
        # tool and the table independently (each falling back to
        # --friction-mu when unset) -- e.g. drop --friction-mu-ground so the
        # body can slide across the table while --friction-mu-tool stays high
        # enough for the tool to drag it along, matching the tracked data
        # (the mesh slides in the direction of the tool's sweep there, which
        # a single shared coefficient tends to suppress by anchoring the body
        # to the table instead).
        parser.add_argument(
            "--friction-mu",
            help="Coulomb friction coefficient for soft-body/rigid contact "
                 "(0 disables friction), used for any shape not overridden by "
                 "--friction-mu-tool / --friction-mu-ground. Default (0.9) is "
                 "raised above the identified ~0.3 so the free body stays "
                 "planted on the table.",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--friction-mu-tool",
            help="Coulomb friction coefficient for the poker capsule / soft-body "
                 "contact only. Overrides --friction-mu for the tool so it can be "
                 "tuned independently of the table (default: --friction-mu).",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--friction-mu-ground",
            help="Coulomb friction coefficient for the table plane / soft-body "
                 "contact only. Overrides --friction-mu for the ground so it can "
                 "be tuned independently of the tool (default: --friction-mu).",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--friction-eps-v",
            help="Sliding speed (m/s) below which contact friction is treated as static; "
            "scaled by dt into a per-step sliding-distance threshold. Raised so "
            "the resting contact with the table sticks instead of creeping.",
            type=float,
            default=5.0e-3,
        )

        return parser


def _apply_warp_config(parser, args):
    """Apply ``--warp-config`` overrides to :obj:`warp.config`.

    Each entry in ``args.warp_config`` must have the form ``KEY=VALUE``.  The
    key is validated to be an existing attribute of :obj:`warp.config`.  The
    value is parsed with :func:`ast.literal_eval`; if that fails the raw
    string is kept.

    Args:
        parser: The argument parser, used for error reporting.
        args: Parsed argument namespace containing ``warp_config``.
    """
    if not args.warp_config:
        return

    for entry in args.warp_config:
        if "=" not in entry:
            parser.error(f"invalid --warp-config format '{entry}': expected KEY=VALUE")

        key, value_str = entry.split("=", 1)

        if not hasattr(wp.config, key):
            parser.error(f"invalid --warp-config key '{key}': not a recognized warp.config setting")

        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        setattr(wp.config, key, value)

def init(parser):
    """Initialize Newton example components from parsed arguments.

    Args:
        parser: An argparse.ArgumentParser instance (should include arguments from
              create_parser()). If None, a default parser is created.

    Returns:
        tuple: (viewer, args) where viewer is configured based on args.viewer

    Raises:
        ValueError: If invalid viewer type or missing required arguments
    """
    import warp as wp  # noqa: PLC0415

    # parse args (before polyscope starts, so --headless can pick the backend)
    args = parser.parse_args()
    # Fill episode-derived option defaults (--fps, --tool-trajectory, material,
    # ...) from --episode before anything reads them.
    resolve_episode_defaults(args)

    if getattr(args, "headless", False):
        ps.init("openGL_mock")          # no window, no GPU render, no-op scene
        ps.set_frame_tick_limit_fps_mode("ignore_limits")
    else:
        ps.init()
        ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
        ps.set_max_fps(args.fps)


    # Apply --warp-config overrides before any Warp API calls
    _apply_warp_config(parser, args)

    # Suppress Warp compilation messages if requested
    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)

    # Set device if specified
    if args.device:
        wp.set_device(args.device)


    return args


class _MeshRecorder:
    """Accumulate the active soft-body mesh (tet indices + vertex positions) and
    the poker pose at every rendered frame, then write them all to a single
    ``.npz`` for offline playback with :mod:`mfem.refinement.replay_octopus`.

    Adaptive refinement changes the vertex / tet counts over a run, so frames
    are held in a list and only stacked into dense arrays on :meth:`close` when
    every frame shares a shape; otherwise they are stored as ``dtype=object``
    arrays (the replay script loads with ``allow_pickle=True``). An unchanging
    tet topology is collapsed to a single ``(num_tets, 4)`` array.
    """

    def __init__(self, path: str, sim: "MFEMRefinementSim"):
        self.path = path
        self.fps = int(sim.fps)
        self.episode = str(getattr(getattr(sim, "ep", None), "key", "octopus"))
        self.start_frame = int(getattr(sim, "start_frame", 0))
        self.capsule_radius = float(sim.capsule_radius)
        self.capsule_half_height = float(sim.capsule_half_height)
        self._times: list[float] = []
        self._positions: list[np.ndarray] = []
        self._tets: list[np.ndarray] = []
        self._capsule: list[np.ndarray] = []

    def capture(self, sim: "MFEMRefinementSim"):
        """Snapshot the same active mesh that :meth:`MFEMRefinementSim.render`
        draws this frame, plus the current capsule transform."""
        n = int(sim.solver._additional_state_0.active_particle_count.numpy()[0])
        m = int(sim.solver._additional_state_0.active_tet_count.numpy()[0])
        self._positions.append(
            sim.state_0.particle_q[:n].numpy().astype(np.float32)
        )
        self._tets.append(
            sim.solver._additional_state_0.tet_indices[:m].numpy().astype(np.int32)
        )
        self._capsule.append(sim.state_0.body_q.numpy()[0].astype(np.float32))
        self._times.append(float(sim.sim_time))

    @staticmethod
    def _pack(frames: "list[np.ndarray]"):
        """Stack same-shaped per-frame arrays into one dense array; fall back to
        a 1-D object array when the shapes differ. Returns (array, is_ragged)."""
        if len({a.shape for a in frames}) == 1:
            return np.stack(frames), False
        packed = np.empty(len(frames), dtype=object)
        packed[:] = frames
        return packed, True

    def close(self):
        if not self._positions:
            print(f"--record: no frames captured, {self.path} not written")
            return

        positions, ragged_p = self._pack(self._positions)
        tets, ragged_t = self._pack(self._tets)
        # Collapse an unchanging topology (refinement off, or no split fired) to
        # a single (num_tets, 4) array so the file stays small.
        tets_constant = not ragged_t and all(
            np.array_equal(t, self._tets[0]) for t in self._tets[1:]
        )
        if tets_constant:
            tets = self._tets[0]

        np.savez_compressed(
            self.path,
            fps=np.float64(self.fps),
            episode=np.str_(self.episode),
            start_frame=np.int64(self.start_frame),
            frame_count=np.int64(len(self._positions)),
            times=np.asarray(self._times, dtype=np.float64),
            positions=positions,
            tet_indices=tets,
            capsule_transform=np.stack(self._capsule),
            capsule_radius=np.float64(self.capsule_radius),
            capsule_half_height=np.float64(self.capsule_half_height),
        )
        pos_desc = "ragged" if ragged_p else "x".join(map(str, positions.shape))
        tet_desc = "constant" if tets_constant else ("ragged" if ragged_t else "per-frame")
        print(
            f"--record: wrote {len(self._positions)} frames to {self.path} "
            f"(positions {pos_desc}, tets {tet_desc})"
        )


def frame_camera_on_soft_body(sim: MFEMRefinementSim, *, fill_fraction: float = 0.7,
                              view_dir=(1.0, -0.5, 1.0)):
    """Aim the camera at the soft body and back it off so it fills the view.

    The camera target is set to the current bounding-box center of the active
    soft-body particles, and the camera is placed along ``view_dir`` at a
    distance such that the body's bounding sphere spans roughly
    ``fill_fraction`` of the vertical field of view (``0.7`` -> the body fills
    ~70% of the screen height, leaving a margin around it).

    Args:
        sim: The running simulation (provides the current particle positions).
        fill_fraction: Fraction of the vertical FOV the body should span (0-1).
        view_dir: Direction from the camera toward the target, in world space.
            The default is a 3/4 view for the +Y-up PokeFlex frame, looking
            down slightly toward the octopus.
    """
    count = int(sim.solver._additional_state_0.active_particle_count.numpy()[0])
    pts = sim.state_0.particle_q[:count].numpy()

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0

    try:
        fov_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
    except Exception:
        fov_deg = 45.0
    half_fov = math.radians(fov_deg) * 0.5

    # radius / sin(half_fov) is the distance at which the bounding sphere exactly
    # fills the vertical FOV; dividing by fill_fraction backs the camera off so
    # the body occupies that fraction of the view instead.
    distance = radius / math.sin(half_fov) / max(fill_fraction, 1e-3)

    d = np.asarray(view_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    camera_pos = center - d * distance

    ps.look_at(tuple(camera_pos), tuple(center))


def run(sim: MFEMRefinementSim, args):
    # Edge width/color are reapplied every frame in render() itself, since
    # render() re-registers a fresh surface mesh object each frame.
    ps.set_up_dir('y_up')
    # Hide polyscope's built-in ground plane.
    ps.set_ground_plane_mode('none')
    # Enable order-independent transparency if anything is drawn see-through:
    # the octopus itself (so a buried poker capsule stays visible) or the
    # tracked-surface overlay.
    want_transparency = getattr(args, "soft_body_alpha", 1.0) < 1.0 or (
        getattr(sim, "tracked_surface", None) is not None
        and sim.tracked_surface.active
    )
    if want_transparency:
        try:
            ps.set_transparency_mode('pretty')
        except Exception:
            pass
    # Frame the camera on the soft body automatically.
    frame_camera_on_soft_body(sim)

    recorder = _MeshRecorder(args.record, sim) if args.record else None

    # For a --record run with a finite (non-looping) tool trajectory, stop a
    # beat after the path ends so the recording matches the poke instead of
    # holding the last pose forever. Interactive runs keep going (close the
    # window). An explicit --record-frames still wins.
    auto_stop_frames = 0
    if (recorder is not None and getattr(sim, "_tool_traj_active", False)
            and not sim._tool_loop and not args.record_frames):
        # Remaining path after the start-frame offset, not the whole trajectory.
        traj_span = float(
            sim._tool_traj_time[-1] - sim._tool_traj_time[0]
            - getattr(sim, "_tool_time_offset", 0.0)
        )
        auto_stop_frames = int(
            math.ceil(traj_span / max(sim._tool_playback_rate, 1e-6) * sim.fps)
        ) + int(0.5 * sim.fps)

    recorded_frames = 0
    stepped = 0
    try:
        while not ps.window_requests_close():
            frame_start_time = time.perf_counter()

            with wp.ScopedTimer("Step and readback"):
                sim.step()

                sim.render()

            ps.frame_tick()
            stepped += 1

            bail = sim.check_bail()
            if bail is not None:
                print(f"[bail] frame {stepped} t={sim.sim_time:.2f}s: {bail}; "
                      "stopping early")
                break

            row = sim.evaluate_surface_loss()
            if row is not None and stepped % sim._surface_loss_every == 0:
                extra = (
                    f"  sym {row['rmse_symmetric_mm']:.2f}"
                    if "rmse_symmetric_mm" in row else ""
                )
                if "corr_rmse_mm" in row:
                    extra += f"  corr {row['corr_rmse_mm']:.2f} (n={row['corr_n_sim']})"
                print(
                    f"[surface loss] frame {stepped} t={row['time']:.2f}s "
                    f"tracked #{row['frame']}: RMSE {row['rmse_mm']:.2f} mm "
                    f"(max {row['max_mm']:.2f} mm, n={row['n_sim']}){extra}"
                )

            if recorder is not None:
                recorder.capture(sim)
                recorded_frames += 1
                if args.record_frames and recorded_frames >= args.record_frames:
                    break

            if auto_stop_frames and stepped >= auto_stop_frames:
                break

            # _throttle_render_fps(frame_start_time, sim.fps)
    finally:
        if recorder is not None:
            recorder.close()
        if sim._loss_rows:
            keys = sim._loss_rows[0].keys()
            curve = {k: np.array([r[k] for r in sim._loss_rows]) for k in keys}
            rmse_mm = curve["rmse_mm"]
            print(
                f"[surface loss] {len(rmse_mm)} frames  RMSE mm  "
                f"mean {rmse_mm.mean():.2f}  min {rmse_mm.min():.2f}  "
                f"max {rmse_mm.max():.2f}   (frame-mean MSE "
                f"{curve['mse'].mean():.3e} m^2)"
            )
            if "corr_rmse_mm" in curve:
                c = curve["corr_rmse_mm"]
                print(
                    f"[surface loss] correspondence RMSE mm  mean {c.mean():.2f}  "
                    f"min {c.min():.2f}  max {c.max():.2f}   (frame-mean MSE "
                    f"{curve['corr_mse'].mean():.3e} m^2)"
                )
            if getattr(args, "surface_loss_out", None):
                np.savez_compressed(args.surface_loss_out, **curve)
                print(f"[surface loss] wrote {args.surface_loss_out}")

if __name__ == "__main__":
    parser = MFEMRefinementSim.create_parser()
    args = init(parser)

    # Loaded verbatim in the PokeFlex world frame (metres, +Y up), no scale or
    # translation. --mesh picks the resolution: octopus_initial_coarse.npz is
    # fTetWild (--epsr 1e-2) on the tracked rest surface (frame 19, frame_id 20):
    # ~600 vertices / ~2000 tets (fTetWild is non-deterministic; counts vary
    # slightly per run); "medium" / "fine" are the same frame at finer
    # envelopes. Built by mfem.refinement.models.make_octopus_tetmesh --frame 19;
    # the episode opens mid-poke and the tool lifts off at frame 13, and by
    # frame 19 the plush has sprung back and the surface tracking is cleanest.
    # The npz's baked k_mu / k_lambda are placeholders and are overridden by the
    # per-tet identified stiffness field (--per-tet-material) or, with 'none', by
    # the scalar --mu / --lmbda.
    episode = get_episode(args)
    mesh_path = episode.mesh_paths[args.mesh]
    print(f"episode {episode.key}: --mesh {args.mesh} -> {mesh_path}")
    refinement_model = MFEMRefinementModel.load(mesh_path, 1.0, wp.vec3(0.0, 0.0, 0.0))

    sim = MFEMRefinementSim(refinement_model, args)

    run(sim, args)
    sim.solver.write_timings("timings.json")
