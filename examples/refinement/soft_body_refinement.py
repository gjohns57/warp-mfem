from mfem.refinement.solver import RefinementSolver
from mfem.refinement.refinement import edge_refinement_scores
import newton
import newton.examples
import numpy as np
import nvtx
import warp as wp
import time
import shutil
import subprocess
from typing import Callable
import math
import ast
from mfem.refinement.models import MFEMRefinementModel
from newton import Axis

import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import igl


@wp.kernel
def _translate_bodies_kernel(body_q: wp.array(dtype=wp.transform), offset: wp.vec3):
    tid = wp.tid()
    tf = body_q[tid]
    pos = wp.transform_get_translation(tf) + offset
    rot = wp.transform_get_rotation(tf)
    body_q[tid] = wp.transform(pos, rot)


_NOISE_GRID_CACHE: dict = {}


def _fbm_noise3(points: np.ndarray, feature_size: float, seed: int, octaves: int = 3) -> np.ndarray:
    """Smooth 3D fractal value noise in roughly [0, 1] at ``points`` (N, 3).
    The lowest octave has blobs about ``feature_size`` across; each further
    octave halves that at half the amplitude. Cubic-interpolated random
    lattices with wraparound, so there are no seams and no dependence on any
    surface parameterization."""
    n = 32
    total = np.zeros(points.shape[0], dtype=np.float64)
    amp_sum = 0.0
    for o in range(octaves):
        key = (seed, o)
        if key not in _NOISE_GRID_CACHE:
            rng = np.random.default_rng(seed * 1009 + o)
            _NOISE_GRID_CACHE[key] = rng.random((n, n, n))
        coords = (points.astype(np.float64) * (2.0**o / feature_size)).T
        amp = 0.5**o
        total += amp * map_coordinates(_NOISE_GRID_CACHE[key], coords, order=3, mode="grid-wrap")
        amp_sum += amp
    return total / amp_sum


def tissue_noise_colors(points: np.ndarray, feature_size: float = 0.03, seed: int = 0,
                        amplitude: float = 0.4) -> np.ndarray:
    """RGB (N, 3) float32 in [0, 1]: mottled reddish-brown "liver" color with
    soft darker vein bands, evaluated as a *solid* texture at 3D ``points``.

    Feeding it each surface point's *rest* position makes the pattern a fixed
    property of the material -- it follows the deforming surface, refined
    vertices get the right color for free, and (unlike wrapping a 2D image
    through a cylindrical UV) there are no seams, no polar streaks and no
    stretching where the surface is steep or faces the projection axis."""
    blotch = _fbm_noise3(points, feature_size, seed, octaves=3)
    t = np.clip((blotch - 0.5) * 3.8 + 0.5, 0.0, 1.0)[:, None]
    dark = np.array([70.0, 22.0, 20.0])     # deep maroon (a bit lifted from the old texture's near-black)
    light = np.array([205.0, 104.0, 88.0])  # lighter pink-red parenchyma
    color = dark * (1.0 - t) + light * t

    # Soft vein bands: where an independent low-frequency field crosses 0.5.
    vein_field = _fbm_noise3(points, feature_size * 2.0, seed + 101, octaves=2)
    veins = np.exp(-(((vein_field - 0.5) / 0.03) ** 2))[:, None]
    color = color * (1.0 - 0.35 * veins) + (dark * 0.6) * (0.35 * veins)
    # ``amplitude`` scales all the variation (blotches and veins) about the
    # mean tissue color: 1 = full contrast, 0 = flat color.
    mid = 0.5 * (dark + light)
    color = mid + amplitude * (color - mid)
    return np.clip(color / 255.0, 0.0, 1.0).astype(np.float32)


def make_surgical_tool_mesh(tip_radius: float, half_height: float, *,
                            shaft_radius_frac: float = 0.35, neck_frac: float = 0.6,
                            shaft_extra_frac: float = 3.0,
                            n_theta: int = 28, n_hemi: int = 14, n_neck: int = 8, n_cap: int = 8):
    """A cosmetic surface-of-revolution mesh standing in for the plain capsule
    the poker's physics still use: a rounded ball tip -- exactly the bottom
    hemisphere of a capsule of the given ``tip_radius``/``half_height``, so
    the visible part that actually indents the mesh matches the real contact
    geometry pixel-for-pixel -- tapering through a short neck into a long
    thin shaft, reading as a surgical probe/dissector instead of a pill.

    Both ends are centered on local +Z with the tip at
    ``z = -(half_height + tip_radius)``, matching ``newton.Mesh.create_capsule``'s
    convention (so it drops in as a straight visual swap, positioned/oriented
    by the same ``capsule_pos``/``capsule_rot``).

    Returns ``(vertices, faces)`` as float32 (N, 3) / int32 (M, 3) arrays.
    """
    shaft_radius = shaft_radius_frac * tip_radius
    neck_length = neck_frac * half_height
    shaft_extra = shaft_extra_frac * half_height

    z_hemi_top = -half_height  # equator of the bottom hemisphere (tip apex is -half_height - tip_radius)
    z_neck_end = z_hemi_top + neck_length
    z_top = half_height + tip_radius + shaft_extra
    z_cap_start = z_top - shaft_radius

    zs, rs = [], []
    for i in range(n_hemi + 1):  # bottom hemisphere: apex -> equator
        phi = -math.pi / 2.0 + (math.pi / 2.0) * (i / n_hemi)
        zs.append(z_hemi_top + tip_radius * math.sin(phi))
        rs.append(tip_radius * math.cos(phi))
    for i in range(1, n_neck + 1):  # linear taper down to the shaft radius
        t = i / n_neck
        zs.append(z_hemi_top + t * (z_neck_end - z_hemi_top))
        rs.append(tip_radius + t * (shaft_radius - tip_radius))
    zs.append(z_cap_start)
    rs.append(shaft_radius)
    for i in range(1, n_cap + 1):  # rounded cap closing the far (handle) end
        phi = (math.pi / 2.0) * (i / n_cap)
        zs.append(z_cap_start + shaft_radius * math.sin(phi))
        rs.append(shaft_radius * math.cos(phi))
    rs[-1] = 0.0  # exact pole, in case of floating-point drift

    zs = np.asarray(zs, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    n_p = zs.shape[0]
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    verts = np.empty((n_p, n_theta, 3), dtype=np.float32)
    verts[:, :, 0] = rs[:, None] * cos_t[None, :]
    verts[:, :, 1] = rs[:, None] * sin_t[None, :]
    verts[:, :, 2] = zs[:, None]
    verts = verts.reshape(-1, 3)

    faces = []
    for i in range(n_p - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a, b = i * n_theta + j, i * n_theta + j2
            c, d = (i + 1) * n_theta + j2, (i + 1) * n_theta + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, np.asarray(faces, dtype=np.int32)


class MFEMRefinementSim:
    def __init__(self,refinement_model: MFEMRefinementModel, args):
        self.sim_time = 0.0
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.iterations = args.iterations
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.do_capture = args.graph_capture
        self.do_line_search = args.line_search

        # self.sim_duration = args.sim_duration

        self.record_energy = args.record_energy



        builder = newton.ModelBuilder(gravity=args.gravity)

        # ---- Keyboard/GUI-controllable capsule -----------------------------
        # load_sim_model() rotates the soft mesh by +90 deg about X, so mirror
        # that here to know where the octopus actually sits in world space:
        #   (x, y, z) -> (x, -z, y)
        p = np.asarray(refinement_model.particles, dtype=np.float64)
        octo_world = np.column_stack((p[:, 0], -p[:, 2], p[:, 1]))
        octo_lo = octo_world.min(axis=0)
        octo_hi = octo_world.max(axis=0)
        octo_ext = octo_hi - octo_lo
        octo_center = 0.5 * (octo_lo + octo_hi)

        # Size the capsule relative to the mesh so it reads as a small
        # vertical "poker" hovering just above it. The fractions are tuned for
        # the octopus; --capsule-radius-frac/--capsule-half-height-frac
        # override them for meshes at a very different scale.
        self.capsule_radius = args.capsule_radius_frac * float(max(octo_ext[0], octo_ext[1]))
        self.capsule_half_height = args.capsule_half_height_frac * float(octo_ext[0])
        self.capsule_axis = Axis.Z

        # Default pose: centred over the octopus, standing vertically (the
        # Z-aligned capsule needs no rotation) with its lower cap a small
        # gap above the octopus top. That default gap is only a fraction of
        # the capsule's own (mesh-scaled) radius, so on a small mesh it can
        # land well inside the *absolute* --contact-d1 barrier distance,
        # meaning the "parked" tool still repels the body at rest even with
        # no poke commanded -- --capsule-standoff pushes it further out to
        # clear that (0 default: identical to the old behavior).
        gap = 0.5 * self.capsule_radius + args.capsule_standoff

        # Poke target: mesh center by default, or an explicit (--poke-x, --poke-z)
        # point in the model's own local frame (before the +90deg-about-X world
        # rotation -- i.e. the same frame make_liver_tetmesh.py's vertices are in),
        # e.g. picked to land on a thick part of the mesh instead of a thin edge.
        # world_xy = (local_x, -local_z) per the (x, y, z) -> (x, -z, y) rotation.
        target_x = octo_center[0] if args.poke_x is None else float(args.poke_x)
        target_y = octo_center[1] if args.poke_z is None else -float(args.poke_z)
        target_xy = np.array([target_x, target_y])

        # Local surface height at the target column (not the mesh's global max --
        # a thick interior point may sit lower than the tallest extremity).
        col_radius = 3.0 * self.capsule_radius
        col_mask = np.linalg.norm(octo_world[:, :2] - target_xy, axis=1) < col_radius
        local_top = float(octo_world[col_mask, 2].max()) if col_mask.any() else float(octo_hi[2])
        self._poke_contact_point = np.array([target_xy[0], target_xy[1], local_top], dtype=np.float64)

        # Approach direction: straight down world -Z (i.e. local +Y) by default,
        # or an explicit outward surface normal at the poke target in the
        # model's own local frame (--poke-normal-x/y/z), e.g. computed offline
        # with igl.per_vertex_normals so the capsule presses perpendicular to a
        # tilted patch instead of just straight down into it.
        if args.poke_normal_x is not None and args.poke_normal_y is not None and args.poke_normal_z is not None:
            local_normal = np.array([args.poke_normal_x, args.poke_normal_y, args.poke_normal_z], dtype=np.float64)
        else:
            local_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        local_normal = local_normal / np.linalg.norm(local_normal)
        # Same (x, y, z) -> (x, -z, y) map as the position rotation above -- a
        # pure rotation, so a direction vector transforms the same way.
        world_normal = np.array([local_normal[0], -local_normal[2], local_normal[1]], dtype=np.float64)
        self._poke_normal_world = world_normal / np.linalg.norm(world_normal)

        # Capsule center when its lower cap exactly touches the mesh's rest-pose
        # surface (zero gap) -- the scripted poke trajectory's press target is
        # measured from here, independent of the parked --capsule-standoff gap.
        self._capsule_touch_pos = (
            self._poke_contact_point + self._poke_normal_world * (self.capsule_half_height + self.capsule_radius)
        )
        self._capsule_pos_default = self._capsule_touch_pos + self._poke_normal_world * gap

        # Orient the capsule (whose local long axis is world +Z at rot=(0,0,0))
        # so it points along the approach normal instead.
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(self._poke_normal_world, z_axis):
            capsule_rot_euler = np.zeros(3)
        else:
            align_rot, _ = Rotation.align_vectors([self._poke_normal_world], [z_axis])
            capsule_rot_euler = align_rot.as_euler("xyz", degrees=True)
        self._capsule_rot_default = np.asarray(capsule_rot_euler, dtype=np.float64)

        # ---- Scripted tool trajectory (--scripted-tool poke) ----------------
        # A simple canned motion for non-interactive demos: hover, descend
        # straight down until pressed --poke-depth past the rest-pose surface,
        # hold, then retract back to the hover pose. Overrides self.capsule_pos
        # each frame in step() -- the existing contact-safe interpolation in
        # _apply_capsule_control() still governs how fast it's actually allowed
        # to move, so a too-aggressive schedule just gets rate-limited, not
        # exploded.
        self.scripted_tool = args.scripted_tool
        self.poke_depth = args.poke_depth
        self.poke_start_time = args.poke_start_time
        self.poke_descend_time = max(args.poke_descend_time, 1.0e-6)
        self.poke_hold_time = args.poke_hold_time
        self.poke_retract_time = max(args.poke_retract_time, 1.0e-6)
        self.poke_count = max(args.poke_count, 1)
        self.poke_rest_time = max(args.poke_rest_time, 0.0)

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

        capsule_body = builder.add_body(xform=self._capsule_transform())
        builder.add_shape_capsule(
            capsule_body,
            radius=self.capsule_radius,
            half_height=self.capsule_half_height,
        )

        # Static ground plane at z = 0 (normal +Z) so the soft body can rest on it.
        # width/length = 0 makes it infinite for collision -- only the visual quad
        # below is sized by --ground-extent, which matters because a 10-unit quad
        # next to a ~0.1m mesh confuses polyscope's auto scene-scale (min zoom
        # distance, clip planes, ...), which is why close-up shots on a small
        # mesh wouldn't actually zoom in until this was shrunk to match.
        self.ground_height = 0.0
        self.ground_extent = args.ground_extent
        builder.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_height),
            width=0.0,
            length=0.0,
        )

        max_particles = refinement_model.resolve_max_particles(args.max_particles_mult)
        max_tets = refinement_model.resolve_max_tets(args.max_tets_mult)

        self.model = refinement_model.load_sim_model(builder, args.mu, max_particles=max_particles)
        max_tris = refinement_model.resolve_max_tris(args.max_tris_mult, self.model)

        # Dirichlet boundary: pin a coordinate-thresholded slab of particles
        # (e.g. "the back" of a mesh) by zeroing their inv-mass before the
        # solver is built -- RefinementSolver reads model.particle_inv_mass
        # once, at construction, to derive its fixed-particle attachment set.
        self._n_model_particles = refinement_model.particles.shape[0]
        if args.fix_axis is not None:
            self._apply_fix_predicate(args.fix_axis, args.fix_side, args.fix_fraction)

        # Optional tissue texture: wrap an RGB image onto the mesh's boundary
        # surface via a precomputed per-vertex UV (see make_liver_tetmesh.py).
        # The boundary triangulation and per-vertex UV are recomputed fresh in
        # _register_tissue_surface() every frame rather than cached here,
        # because --refine changes both: new tets change the boundary, and new
        # vertices have no entry in the loaded UV file (they borrow their
        # nearest original neighbor's UV -- see _tissue_uv_base_count below).
        self.tissue_texture = None
        self.tissue_uv = None
        self._tissue_uv_base_count = 0
        self.tissue_edge_radius = args.tissue_edge_radius
        # --tissue-noise: color the surface with a solid 3D noise at each
        # point's rest position instead of a UV-mapped image (no UV needed).
        self.tissue_noise = args.tissue_noise
        self.tissue_noise_scale = args.tissue_noise_scale
        self.tissue_noise_seed = args.tissue_noise_seed
        self.tissue_noise_amplitude = args.tissue_noise_amplitude
        self.tissue_subdivide = args.tissue_subdivide
        self.tissue_enabled = args.tissue_noise or args.tissue_texture is not None
        if args.tissue_texture is not None and not args.tissue_noise:
            from PIL import Image  # noqa: PLC0415
            self.tissue_texture = np.asarray(Image.open(args.tissue_texture).convert("RGB"), dtype=np.float32) / 255.0
            uv = np.load(args.tissue_uv).astype(np.float32)
            self._tissue_uv_base_count = uv.shape[0]
            if uv.shape[0] < max_particles:
                padded = np.zeros((max_particles, 2), dtype=np.float32)
                padded[:uv.shape[0]] = uv
                uv = padded
            self.tissue_uv = uv


        self.solver = RefinementSolver(
            model=self.model,
            iterations=self.iterations,
            max_tets=max_tets,
            max_particles=max_particles,
            max_tris=max_tris,
            refine_every_n_steps=args.refine_every,
            max_new_vertices_per_refine=args.max_new_vertices,
            preconditioner=args.preconditioner,
            line_search=self.do_line_search,
            enable_refinement=args.refine,
            refine_density=args.refine_density,
            refine_tet_score_weight=args.refine_tet_score_weight,
            refine_vertex_score_weight=args.refine_vertex_score_weight,
            penetrating_edge_score=args.penetrating_edge_score,
            refine_conflict_iterations=args.refine_conflict_iterations,
            refine_split_position=args.refine_split_position,
            refine_hashmap_load_factor=args.refine_hashmap_load_factor,
            refine_hashmap_edges_per_tet=args.refine_hashmap_edges_per_tet,
            refine_scoring=args.refine_scoring,
            refine_min_edge_length=args.refine_min_edge_length,
            refine_elastic_weight=args.refine_elastic_weight,
            refine_contact_weight=args.refine_contact_weight,
            refine_tri_contact_weight=args.refine_tri_contact_weight,
            refine_curvature_weight=args.refine_curvature_weight,
            refine_geometric_threshold=args.refine_geometric_threshold,
            cg_max_iterations=args.cg_max_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_check_every=args.cg_check_every,
            cg_use_cuda_graph=args.graph_capture,
            preconditioner_singular_threshold=args.preconditioner_singular_threshold,
            line_search_max_iterations=args.line_search_max_iterations,
            line_search_alpha0=args.line_search_alpha0,
            line_search_tau=args.line_search_tau,
            line_search_c=args.line_search_c,
            line_search_threshold=args.line_search_threshold,
            attachment_stiffness=args.attachment_stiffness,
            contact_d0=args.contact_d0,
            contact_d1=args.contact_d1,
            contact_stiffness=args.contact_stiffness,
            friction_mu=args.friction_mu,
            friction_eps_v=args.friction_eps_v,
            tri_contact=not args.no_tri_contact,
            shell_mu=args.shell_mu,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        particle_ct = self.solver._additional_state_0.active_particle_count.numpy()[0]
        self.ps_volume = ps.register_volume_mesh("Soft body", self.model.particle_q[:particle_ct].numpy(), self.model.tet_indices.numpy())
        self.ps_volume.add_scalar_quantity("elastic energy", self.solver._elastic_energy[:self.solver._additional_state_0.active_tet_count.numpy()[0]].numpy(), defined_on='cells', vminmax=(0.0, 10.0), cmap='blues', enabled=True)
        self._old_vertex_count = particle_ct

        self.ps_tissue = None
        if self.tissue_enabled:
            # The tet volume mesh's own surface would otherwise z-fight with
            # (and hide) the textured skin; keep it around for the elastic-
            # energy overlay but don't show it by default.
            self.ps_volume.set_enabled(False)
            self._register_tissue_surface(self.model.particle_q[:particle_ct].numpy())

        # Cosmetic only: the physics collider (added below) is still the plain
        # capsule this mesh's ball tip exactly matches -- see
        # make_surgical_tool_mesh's docstring.
        tool_verts, tool_faces = make_surgical_tool_mesh(self.capsule_radius, self.capsule_half_height)
        self.ps_capsule = ps.register_surface_mesh("Capsule", tool_verts, tool_faces)
        self.ps_capsule.set_color((0.72, 0.74, 0.78))  # polished steel
        self.ps_capsule.set_material("candy")  # glossy/specular -- reads as metal, not clay
        self._sync_capsule()

        # Drive the capsule from the polyscope window (ImGui panel + hotkeys).
        ps.set_user_callback(self._capsule_gui_callback)

        e = self.ground_extent
        h = self.ground_height
        ground_verts = np.array(
            [[-e, -e, h], [e, -e, h], [e, e, h], [-e, e, h]], dtype=np.float32
        )
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.ps_ground = None
        if not args.no_ground:
            self.ps_ground = ps.register_surface_mesh("Ground", ground_verts, ground_faces)
        # self.ps_ground.set_enabled(False)

        if args.graph_capture:
            self.capture()
        else:
            self.graph = None

    def _tissue_uv_for(self, particle_positions: np.ndarray) -> np.ndarray:
        """Per-vertex UV for the current (possibly refined) vertex set. Original
        vertices use their precomputed UV; vertices --refine has since added
        (indices >= _tissue_uv_base_count, absent from the loaded UV file)
        borrow the UV of whichever original vertex is currently nearest them,
        which is at least locally consistent even though it isn't a real
        parameterization of the new triangles."""
        n = particle_positions.shape[0]
        base_n = min(n, self._tissue_uv_base_count)
        uv = np.empty((n, 2), dtype=np.float32)
        uv[:base_n] = self.tissue_uv[:base_n]
        if n > self._tissue_uv_base_count:
            tree = cKDTree(particle_positions[:self._tissue_uv_base_count])
            _dist, nn = tree.query(particle_positions[self._tissue_uv_base_count:n])
            uv[self._tissue_uv_base_count:n] = self.tissue_uv[nn]
        return uv

    def _register_tissue_surface(self, particle_positions: np.ndarray):
        """(Re-)register the boundary-surface mesh wearing the tissue texture,
        wired up with its UV parameterization, plus a thin-radius wireframe
        overlay of its edges. Both the boundary triangulation and the UV are
        recomputed from the *current* tet_indices/vertex set every call (not
        cached), since --refine changes them; render() calls this fresh each
        frame, mirroring how it re-registers ``ps_volume``."""
        tet_count = int(self.solver._additional_state_0.active_tet_count.numpy()[0])
        tet_indices = self.solver._additional_state_0.tet_indices[:tet_count].numpy()
        boundary_faces, _j, _k = igl.boundary_facets(tet_indices.astype(np.int64))
        boundary_faces = boundary_faces.astype(np.int32)

        ps.remove_surface_mesh("Tissue", error_if_absent=False)
        if self.tissue_noise:
            # Solid texture at each surface point's *rest* position. The
            # display mesh is midpoint-subdivided first (positions and rest
            # positions linearly, same connectivity) so the coloring is sampled
            # finer than the coarse sim mesh -- the geometry is unchanged.
            n = particle_positions.shape[0]
            rest = self.solver._additional_state_0.rest_particle_q[:n].numpy()
            verts, faces = particle_positions, boundary_faces
            if self.tissue_subdivide > 0:
                sv, faces = igl.upsample(
                    np.hstack([particle_positions, rest]).astype(np.float64),
                    boundary_faces.astype(np.int64), self.tissue_subdivide,
                )
                verts, rest = sv[:, :3].astype(np.float32), sv[:, 3:]
                faces = faces.astype(np.int32)
            self.ps_tissue = ps.register_surface_mesh("Tissue", verts, faces)
            self.ps_tissue.add_color_quantity(
                "tissue",
                tissue_noise_colors(rest, self.tissue_noise_scale, self.tissue_noise_seed,
                                    self.tissue_noise_amplitude),
                defined_on="vertices", enabled=True,
            )
        else:
            uv = self._tissue_uv_for(particle_positions)
            self.ps_tissue = ps.register_surface_mesh(
                "Tissue", particle_positions, boundary_faces
            )
            self.ps_tissue.add_parameterization_quantity("uv", uv, enabled=True)
            self.ps_tissue.add_color_quantity(
                "tissue", self.tissue_texture, defined_on="texture", param_name="uv",
                enabled=True,
            )

        # Mesh-edge wireframe as its own thin curve network: SurfaceMesh's
        # built-in edge_width is relative to an auto length-scale calibrated
        # for much bigger meshes and renders as comically thick tubes here.
        edges = np.concatenate(
            [boundary_faces[:, [0, 1]], boundary_faces[:, [1, 2]], boundary_faces[:, [2, 0]]],
            axis=0,
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        ps.remove_curve_network("Tissue edges", error_if_absent=False)
        edge_net = ps.register_curve_network("Tissue edges", particle_positions, edges)
        edge_net.set_radius(self.tissue_edge_radius, relative=False)
        edge_net.set_color((0.05, 0.05, 0.05))

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)

    def _update_scripted_tool(self):
        """Overwrite ``self.capsule_pos`` (the *commanded* pose) from the
        canned --scripted-tool poke schedule. No-op when --scripted-tool is
        'none', leaving the GUI/hotkeys in control as before.

        With --poke-count > 1, --poke-start-time only gates the very first
        cycle; repeats follow immediately, each separated by --poke-rest-time
        of hovering. The whole trajectory runs along --poke-normal-x/y/z (the
        world -Z world axis by default), so this interpolates a single scalar
        ``s`` from 0 (hover) to 1 (fully pressed) along that fixed line rather
        than a bare Z coordinate."""
        if self.scripted_tool != "poke":
            return

        # The capsule's orientation is fixed for the whole run (set once in
        # __init__ from --poke-normal-x/y/z) -- reassert it every frame so an
        # interactive nudge can't leave it pointing the wrong way mid-script.
        self.capsule_rot = self._capsule_rot_default.copy()

        t = self.sim_time
        hover_pos = self._capsule_pos_default
        press_pos = self._capsule_touch_pos - self._poke_normal_world * self.poke_depth

        cycle_time = self.poke_descend_time + self.poke_hold_time + self.poke_retract_time
        period = cycle_time + self.poke_rest_time

        t_since_start = t - self.poke_start_time
        if t_since_start < 0.0:
            s = 0.0
        else:
            cycle_idx = min(int(t_since_start // period), self.poke_count - 1)
            t_local = t_since_start - cycle_idx * period

            t_descend_end = self.poke_descend_time
            t_hold_end = t_descend_end + self.poke_hold_time
            t_retract_end = t_hold_end + self.poke_retract_time

            if t_local < t_descend_end:
                s = self._smoothstep(t_local / self.poke_descend_time)
            elif t_local < t_hold_end:
                s = 1.0
            elif t_local < t_retract_end:
                s = 1.0 - self._smoothstep((t_local - t_hold_end) / self.poke_retract_time)
            else:
                s = 0.0

        self.capsule_pos = hover_pos + (press_pos - hover_pos) * s

    def _apply_fix_predicate(self, axis: str, side: str, fraction: float):
        """Zero ``self.model.particle_inv_mass`` for particles within
        ``fraction`` of the mesh's extent along ``axis``, measured from
        ``side`` ('max' or 'min'). This is what pins them as a Dirichlet
        boundary -- see the comment at the call site."""
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        n = self._n_model_particles
        coord = self.model.particle_q.numpy()[:n, axis_idx]
        lo, hi = float(coord.min()), float(coord.max())
        span = hi - lo
        mask = coord >= hi - fraction * span if side == "max" else coord <= lo + fraction * span

        inv_mass = self.model.particle_inv_mass.numpy()
        inv_mass[:n][mask] = 0.0
        self.model.particle_inv_mass.assign(inv_mass)
        print(f"--fix-axis {axis} --fix-side {side} --fix-fraction {fraction}: "
              f"pinned {int(mask.sum())}/{n} particles as a Dirichlet boundary")

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

    def _capsule_min_distance(self, pos, rot_deg, pts) -> float:
        """Smallest signed distance from ``pts`` to the capsule surface for the
        given capsule pose (capsule long axis is local +Z)."""
        if pts.shape[0] == 0:
            return np.inf
        R = Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix()  # local->world
        local = (pts - np.asarray(pos, dtype=np.float64)) @ R             # world->local
        hh = self.capsule_half_height
        seg_z = np.clip(local[:, 2], -hh, hh)
        radial = np.hypot(local[:, 0], local[:, 1])
        d = np.hypot(radial, local[:, 2] - seg_z) - self.capsule_radius
        return float(d.min())

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
                # With an odd number of substeps the graph reads from buffer A and writes
                # to buffer B, but A is never updated — every replay steps from the same
                # state.  Copy the output (state_0 after simulate's final swap) back to
                # the input buffer (state_1) so A always holds the latest state before the
                # next replay.
                if self.sim_substeps % 2 == 1:
                    wp.copy(self.state_1.particle_q, self.state_0.particle_q)
                    wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.graph = capture.graph
        else:
            self.graph = None


    def simulate(self):
        for _ in range(self.sim_substeps):
            with nvtx.annotate("solver_step", color="blue"):
                self.solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
                )

            self.state_0, self.state_1 = self.state_1, self.state_0

    @nvtx.annotate("Solver Frame", color="green")
    def step(self):
        # Scripted demos (--scripted-tool) drive the commanded pose here;
        # interactive runs leave capsule_pos as the GUI/hotkeys last set it.
        self._update_scripted_tool()

        # Advance the imposed capsule pose toward what the user asked for, but
        # only as fast as the contact solver can take, then push it into the
        # sim state before this frame's solver step.
        self._apply_capsule_control()

        if self.graph:
            wp.capture_launch(self.graph)
            self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()

        # Cache the post-step gap so next frame's motion can be rate-limited.
        self._refresh_capsule_gap()

        self.sim_time += self.frame_dt

    def render(self):
        # Lots of device to host copy but it is just in the rendering so I can ignore this part when presenting timings
        particle_count = self.solver._additional_state_0.active_particle_count.numpy()[0]
        tet_count = self.solver._additional_state_0.active_tet_count.numpy()[0]

        if self._old_vertex_count != particle_count:
            print(f"new tet count {tet_count}, new vert count {particle_count}")
        ps.remove_volume_mesh("Soft body")
        self.ps_volume = ps.register_volume_mesh("Soft body", self.state_0.particle_q[:particle_count].numpy(), self.solver._additional_state_0.tet_indices[:tet_count].numpy())
        self.ps_volume.set_edge_width(1.0)
        self.ps_volume.set_edge_color([0.0, 0.0, 0.0])
        self.ps_volume.add_scalar_quantity("elastic energy", self.solver._elastic_energy[:tet_count].numpy(), defined_on='cells', vminmax=(0.0, 0.01), cmap='blues', enabled=True)
        self._old_vertex_count = particle_count

        if self.tissue_enabled:
            self.ps_volume.set_enabled(False)
            self._register_tissue_surface(self.state_0.particle_q[:particle_count].numpy())

        # Debug overlay: every mesh edge colored by its refinement score (from
        # the most recent refine() scoring pass) -- VolumeMesh itself only
        # supports 'vertices'/'cells' scalar quantities, not per-edge, so this
        # needs a separate curve network drawn over the same vertex positions.
        ps.remove_curve_network("Refinement scores", error_if_absent=False)
        if self.solver._refine_enabled:
            edges, scores = edge_refinement_scores(self.solver._additional_state_0, self.solver._refine_buffers)
            self.ps_refine_scores = ps.register_curve_network(
                "Refinement scores", self.state_0.particle_q[:particle_count].numpy(), edges,
            )
            self.ps_refine_scores.set_radius(0.001, relative=False)
            # When the tissue-texture path is active it already draws its own
            # thin boundary wireframe ("Tissue edges"); this ALL-tet-edges
            # overlay (denser, and a fixed 1mm radius that reads as chunky
            # tubes once the camera is zoomed in close) would just double up
            # on top of it, so leave it registered (toggleable in the GUI)
            # but hidden by default rather than skipping it outright.
            self.ps_refine_scores.set_enabled(not self.tissue_enabled)
            self.ps_refine_scores.add_scalar_quantity(
                "score", scores, defined_on="edges", enabled=True, cmap="reds",
            )

        capsule_tf = self.state_0.body_q.numpy()[0]
        capsule_mat = np.eye(4)
        capsule_mat[:3, :3] = Rotation.from_quat(capsule_tf[3:7]).as_matrix()
        capsule_mat[:3, 3] = capsule_tf[:3]
        self.ps_capsule.set_transform(capsule_mat)
        self.ps_capsule.set_enabled(self.capsule_visible)


    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

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
            "--model",
            help="Path to the .npz soft-body model to load (e.g. fem_beam.npz or models/octopus_frame0_coarse.npz)",
            type=str,
            default="models/octopus_frame0_coarse.npz",
        )

        parser.add_argument(
            "--translate-y",
            help="Shift the loaded model along its own local +Y (= world up, since "
                 "load_sim_model rotates +90 deg about X) before simulating, e.g. to "
                 "clear the z=0 ground plane for a model authored centered at the origin",
            type=float,
            default=0.0,
        )

        parser.add_argument(
            "--mu",
            help="First Lame's parameter for elasticity",
            type=float,
            default=1.0e2,
        )

        parser.add_argument(
            "--capsule-radius-frac",
            type=float, default=0.09,
            help="Poker capsule radius as a fraction of max(mesh x-extent, y-extent) "
                 "(default 0.09, tuned for the octopus; shrink this on a small mesh).",
        )
        parser.add_argument(
            "--capsule-half-height-frac",
            type=float, default=0.183,
            help="Poker capsule half-height as a fraction of the mesh's x-extent "
                 "(default 0.183, tuned for the octopus).",
        )

        parser.add_argument(
            "--gravity",
            help="Gravitational acceleration along world -Z, m/s^2 (default -9.81; 0 disables gravity)",
            type=float,
            default=-9.81,
        )

        parser.add_argument(
            "--no-ground",
            action="store_true",
            help="Don't draw the ground-plane quad. Visual only: the collision plane at "
                 "z=0 is still in the physics model (harmless when the body sits well above it).",
        )
        parser.add_argument(
            "--ground-extent",
            type=float, default=10.0,
            help="Half-width of the visual ground-plane quad, m (default 10, tuned for "
                 "the octopus). On a much smaller mesh this dwarfs it enough to confuse "
                 "polyscope's auto scene-scale (in particular, close-up camera shots stop "
                 "actually zooming in) -- shrink this to roughly the mesh's own scale.",
        )

        parser.add_argument(
            "--scripted-tool",
            choices=["none", "poke"], default="none",
            help="Drive the capsule with a canned motion instead of the GUI/hotkeys: "
                 "'poke' hovers, descends straight down --poke-depth past the mesh's "
                 "rest-pose surface, holds, then retracts (default: none, interactive).",
        )
        parser.add_argument(
            "--poke-depth",
            type=float, default=0.03,
            help="How far past the mesh's rest-pose surface the scripted poke presses, m (default 0.03).",
        )
        parser.add_argument(
            "--poke-start-time",
            type=float, default=0.5,
            help="Seconds to hold at the hover pose before the scripted poke starts descending (default 0.5).",
        )
        parser.add_argument(
            "--poke-descend-time",
            type=float, default=1.0,
            help="Seconds for the scripted poke's descend phase (default 1.0).",
        )
        parser.add_argument(
            "--poke-hold-time",
            type=float, default=0.5,
            help="Seconds the scripted poke holds at full depth (default 0.5).",
        )
        parser.add_argument(
            "--poke-retract-time",
            type=float, default=1.0,
            help="Seconds for the scripted poke's retract phase (default 1.0).",
        )
        parser.add_argument(
            "--poke-count",
            type=int, default=1,
            help="Number of descend/hold/retract cycles at the same spot (default 1). "
                 "Cycles repeat back-to-back with --poke-rest-time hovering in between.",
        )
        parser.add_argument(
            "--poke-rest-time",
            type=float, default=0.4,
            help="Seconds spent hovering at rest between repeated pokes (default 0.4; "
                 "only matters when --poke-count > 1).",
        )
        parser.add_argument(
            "--poke-x",
            type=float, default=None,
            help="Poke target's X coordinate in the model's own local frame (before the "
                 "sim's +90deg-about-X world rotation), e.g. picked to land on a thick "
                 "part of the mesh instead of the bbox-center default. Default: mesh center.",
        )
        parser.add_argument(
            "--poke-z",
            type=float, default=None,
            help="Poke target's Z coordinate in the model's own local frame (see --poke-x). "
                 "Default: mesh center.",
        )
        parser.add_argument(
            "--poke-normal-x", type=float, default=None,
            help="Outward surface normal at the poke target, X component, in the model's "
                 "own local frame (e.g. from igl.per_vertex_normals at the chosen vertex). "
                 "All three components are required together; the capsule is oriented "
                 "along this normal and its trajectory presses straight along it, instead "
                 "of always straight down world -Z. Default: (0, 1, 0) local (= world -Z), "
                 "the old straight-down behavior.",
        )
        parser.add_argument("--poke-normal-y", type=float, default=None)
        parser.add_argument("--poke-normal-z", type=float, default=None)

        parser.add_argument(
            "--camera-zoom-contact",
            action="store_true",
            help="Frame the initial camera on the poke/contact point instead of the "
                 "whole mesh -- e.g. to watch refinement happen close up (default: off).",
        )
        parser.add_argument(
            "--camera-zoom-radius",
            type=float, default=None,
            help="Radius (m) of the close-up view when --camera-zoom-contact is set "
                 "(default: 4x the poker capsule's radius).",
        )
        parser.add_argument(
            "--camera-zoom-fill",
            type=float, default=0.6,
            help="Fraction of the vertical FOV the close-up radius should fill (default 0.6).",
        )
        parser.add_argument(
            "--camera-zoom-azimuth-deg",
            type=float, default=0.0,
            help="Orbit the close-up camera this many degrees around the poke's surface "
                 "normal (after --camera-zoom-tilt-deg), to view the contact patch from a "
                 "different side (default 0).",
        )
        parser.add_argument(
            "--camera-zoom-tilt-deg",
            type=float, default=0.0,
            help="Tilt the close-up camera this many degrees away from looking straight "
                 "down the surface normal (default 0 = straight-on). A straight-on view "
                 "looks down the poker's own axis, so the tool itself hides the dimple "
                 "forming under it; tilting off-axis lets you see under/beside it.",
        )

        parser.add_argument(
            "--capsule-standoff",
            help="Extra clearance (m) added above the poker capsule's default resting "
                 "gap above the mesh -- use this to clear --contact-d1 on a small mesh "
                 "where the default gap would otherwise sit inside the contact barrier "
                 "at rest (default 0, unchanged from before).",
            type=float,
            default=0.0,
        )

        parser.add_argument(
            "--lmbda",
            help="Second Lame's parameter for elasticity",
            type=float,
            default=4.0e2,
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
            default=8,
        )

        parser.add_argument(
            "--substeps",
            help="Time steps per frame",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--fps",
            help="Frames per second",
            type=int,
            default=24,
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
        # make_tetmesh_models.py); pass one of these to override it.
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
            help="Record the polyscope view to a video file (encoded with ffmpeg)",
            nargs="?",
            const="simulation.mp4",
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
            help="Density used for mass recomputation during a refine pass",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--refine-tet-score-weight",
            help="Weight on per-tet elastic energy in the edge refinement score (0 disables it)",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--refine-vertex-score-weight",
            help="Weight on the min endpoint vertex score (contact barrier energy) in the edge refinement score",
            type=float,
            default=0.1,
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
            default=5,
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
            "--refine-curvature-weight",
            help="geometric scoring: scale the contact terms by 1 + w * L * kappa, with "
                 "kappa the tool distance-field curvature (SDF Hessian) along the edge, "
                 "so edges that wrap too far around the poker split first (0 disables)",
            type=float,
            default=0.0,
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
            default=1000,
        )
        parser.add_argument(
            "--cg-tolerance",
            help="Relative residual tolerance for the global CG solve",
            type=float,
            default=1.0e-6,
        )
        parser.add_argument(
            "--cg-check-every",
            help="Convergence-check cadence for the global CG solve. 0 = no host-side check: "
                 "on CUDA the loop is a captured conditional graph that still exits on "
                 "convergence, but on the CPU it runs all --cg-max-iterations every Newton "
                 "step. Default: 0 with -g on CUDA (captured CG), otherwise 10.",
            type=int,
            default=None,
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
            default=20,
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

        # Contact barrier.
        parser.add_argument(
            "--contact-d0",
            help="Inner contact barrier distance; for d <= d0 the log-barrier is replaced by its quadratic extrapolation",
            type=float,
            default=1.0e-2,
        )
        parser.add_argument(
            "--contact-d1",
            help="Outer contact barrier distance; the barrier is zero for d >= d1 and active in (d0, d1)",
            type=float,
            default=1.0e-1,
        )
        parser.add_argument(
            "--contact-stiffness",
            help="Scaling coefficient on the contact barrier energy",
            type=float,
            default=1.0e7,
        )
        parser.add_argument(
            "--no-tri-contact",
            action="store_true",
            help="Vertex-only contact: disable the per-surface-triangle barrier against the capsule tool",
        )

        # Contact friction (lagged / semi-implicit Coulomb friction).
        parser.add_argument(
            "--friction-mu",
            help="Coulomb friction coefficient for soft-body/rigid contact (0 disables friction)",
            type=float,
            default=0.3,
        )
        parser.add_argument(
            "--friction-eps-v",
            help="Sliding speed (m/s) below which contact friction is treated as static; "
            "scaled by dt into a per-step sliding-distance threshold",
            type=float,
            default=1.0e-3,
        )

        # Membrane shell over the surface triangulation (e.g. a capsule/skin
        # layer distinct from the bulk tet elasticity -- a real liver has
        # Glisson's capsule, a thin fibrous membrane around the parenchyma).
        parser.add_argument(
            "--shell-mu",
            type=float, default=0.0,
            help="Surface shear modulus of a membrane-shell layer over the mesh's "
                 "boundary triangles, in Pa*m (bulk shell mu times its thickness); "
                 "0 disables it (default). Adds ARAP membrane resistance on top of "
                 "the volumetric tet elasticity, independent of --mu/--k-lambda.",
        )

        # Dirichlet boundary: pin a coordinate-thresholded slab of particles.
        parser.add_argument(
            "--fix-axis",
            choices=["x", "y", "z"],
            default=None,
            help="Pin (Dirichlet-fix, zero inv-mass) particles within --fix-fraction of "
                 "the mesh's extent along this axis, measured from --fix-side. E.g. "
                 "'--fix-axis z --fix-side max' fixes the mesh's +z face. Default: no fixing.",
        )
        parser.add_argument(
            "--fix-side",
            choices=["min", "max"],
            default="max",
            help="Which end of --fix-axis's range to pin (default: max).",
        )
        parser.add_argument(
            "--fix-fraction",
            type=float,
            default=0.15,
            help="Fraction of the --fix-axis extent to pin, measured from --fix-side "
                 "(default 0.15).",
        )

        # Optional tissue texture (see make_liver_tetmesh.py).
        parser.add_argument(
            "--tissue-texture",
            type=str,
            default=None,
            help="Path to an RGB image wrapped onto the soft body's boundary surface "
                 "via --tissue-uv (e.g. models/liver_tissue.png). Default: no texture, "
                 "just the plain volume-mesh rendering.",
        )
        parser.add_argument(
            "--tissue-uv",
            type=str,
            default=None,
            help="Path to an (N,2) .npy of per-vertex UV coordinates, N >= the model's "
                 "initial particle count, in the same vertex order (e.g. "
                 "models/liver_coarse_uv.npy). Required with --tissue-texture.",
        )
        parser.add_argument(
            "--tissue-noise",
            action="store_true",
            help="Color the tissue surface with procedural 3D noise evaluated at each "
                 "point's rest position (a solid texture) instead of wrapping "
                 "--tissue-texture through --tissue-uv. No image/UV files needed, and no "
                 "seams/stretching/polar streaks, since nothing is being parameterized.",
        )
        parser.add_argument(
            "--tissue-noise-scale", type=float, default=0.03,
            help="Size (m) of the largest color blotches for --tissue-noise (default 0.03).",
        )
        parser.add_argument(
            "--tissue-noise-amplitude", type=float, default=0.4,
            help="Strength of the --tissue-noise color variation about the mean tissue "
                 "color: 1 = full contrast, 0 = flat (default 0.4, subtle).",
        )
        parser.add_argument("--tissue-noise-seed", type=int, default=0,
                            help="Random seed for --tissue-noise (default 0).")
        parser.add_argument(
            "--tissue-subdivide", type=int, default=2,
            help="Midpoint-subdivision levels applied to the *display* surface only for "
                 "--tissue-noise, so the color is sampled finer than the coarse sim mesh "
                 "(geometry unchanged; default 2).",
        )
        parser.add_argument(
            "--tissue-edge-radius",
            type=float, default=0.0003,
            help="Absolute radius (m) of the mesh-edge wireframe drawn over the tissue "
                 "surface (default 0.0003m = 0.3mm). SurfaceMesh's own edge_width is "
                 "relative to an auto length-scale and looks comically thick on a small "
                 "mesh, so this is a separate thin-radius curve-network overlay instead.",
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

    ps.init()
    ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
    # parse args

    args = parser.parse_args()
    ps.set_max_fps(args.fps)


    # Apply --warp-config overrides before any Warp API calls
    _apply_warp_config(parser, args)

    # Suppress Warp compilation messages if requested
    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)

    # Set device if specified
    if args.device:
        wp.set_device(args.device)

    # CG convergence checks: Warp's check_every=0 path only terminates early inside a CUDA
    # conditional graph; on the CPU it would spin through every allowed iteration.
    if getattr(args, "cg_check_every", None) is None:
        captured_cg = wp.get_device().is_cuda and bool(getattr(args, "graph_capture", False))
        args.cg_check_every = 0 if captured_cg else 10
        if not captured_cg:
            why = "CPU device" if not wp.get_device().is_cuda else "no graph capture (-g)"
            print(f"--cg-check-every: {why}, defaulting to {args.cg_check_every} "
                  "(0 only exits early inside a captured CUDA graph)")


    return args


class _VideoRecorder:
    """Pipe polyscope screenshots into ffmpeg to produce a video file.

    Frames are grabbed with :func:`polyscope.screenshot_to_buffer` and streamed
    as raw RGB to an ffmpeg subprocess, so no Python video-encoding package is
    required -- only an ``ffmpeg`` binary on ``PATH``.
    """

    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = fps
        self.proc = None

    def _start(self, width: int, height: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; cannot record the simulation")
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(self.fps),
            "-i", "-",
            "-an",
            # libx264 + yuv420p needs even dimensions; crop off a stray odd row/col.
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "medium",
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add_frame(self):
        buf = ps.screenshot_to_buffer(transparent_bg=False)  # (H, W, 4) uint8
        rgb = np.ascontiguousarray(buf[:, :, :3])
        if self.proc is None:
            h, w = rgb.shape[:2]
            self._start(w, h)
        self.proc.stdin.write(rgb.tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


def tilt_view_dir(view_dir, tilt_deg: float, reference=(0.0, 0.0, 1.0)):
    """Tilt ``view_dir`` (camera-to-target direction) by ``tilt_deg`` toward
    ``reference``, e.g. so a close-up that would otherwise look straight down
    a surface normal (the poker hiding the dimple forming right under it)
    instead comes in at an angle. ``reference`` is only used to build a
    rotation axis perpendicular to ``view_dir``; if it's nearly parallel to
    ``view_dir`` (e.g. looking straight along world Z) this falls back to
    world X so the rotation axis stays well defined."""
    view_dir = np.asarray(view_dir, dtype=np.float64)
    view_dir = view_dir / np.linalg.norm(view_dir)
    reference = np.asarray(reference, dtype=np.float64)
    if abs(np.dot(view_dir, reference)) > 0.99:
        reference = np.array([1.0, 0.0, 0.0])
    tangent = reference - np.dot(reference, view_dir) * view_dir
    tangent = tangent / np.linalg.norm(tangent)
    axis = np.cross(view_dir, tangent)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(math.radians(tilt_deg) * axis).apply(view_dir)


def frame_camera_on_point(center, radius: float, *, fill_fraction: float = 0.7,
                          view_dir=(1.0, -1.0, -0.4)):
    """Aim the camera at ``center`` and back it off so a sphere of ``radius``
    spans roughly ``fill_fraction`` of the vertical field of view. Shared by
    :func:`frame_camera_on_soft_body` (whole-mesh framing) and any close-up
    shot (e.g. zooming in on a contact region -- see ``--camera-zoom-contact``)."""
    center = np.asarray(center, dtype=np.float64)
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0

    try:
        fov_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
    except Exception:
        fov_deg = 45.0
    half_fov = math.radians(fov_deg) * 0.5

    # radius / sin(half_fov) is the distance at which the sphere exactly fills
    # the vertical FOV; dividing by fill_fraction backs the camera off so it
    # occupies that fraction of the view instead.
    distance = radius / math.sin(half_fov) / max(fill_fraction, 1e-3)

    d = np.asarray(view_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    camera_pos = center - d * distance

    ps.look_at(tuple(camera_pos), tuple(center))


def frame_camera_on_soft_body(sim: MFEMRefinementSim, *, fill_fraction: float = 0.7,
                              view_dir=(1.0, -1.0, -0.4)):
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
            The default is a 3/4 view looking down slightly toward the origin.
    """
    count = int(sim.solver._additional_state_0.active_particle_count.numpy()[0])
    pts = sim.state_0.particle_q[:count].numpy()

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))

    frame_camera_on_point(center, radius, fill_fraction=fill_fraction, view_dir=view_dir)


def run(sim: MFEMRefinementSim, args):
    # Edge width/color are reapplied every frame in render() itself, since
    # render() re-registers a fresh volume mesh object each frame.
    ps.set_up_dir('z_up')
    # Hide polyscope's built-in ground plane.
    ps.set_ground_plane_mode('none')
    if args.camera_zoom_contact:
        # Close-up, static for the whole run: centered on the poke's contact
        # point at the mesh's rest-pose surface (not the hover pose, which
        # sits --capsule-standoff above it). Base direction looks straight in
        # along the same surface normal the capsule presses along (face-on to
        # the contact patch); --camera-zoom-tilt-deg tips that off-axis so the
        # poker itself doesn't sit right in front of the dimple it's making.
        zoom_center = sim._poke_contact_point
        zoom_radius = args.camera_zoom_radius if args.camera_zoom_radius is not None \
            else 4.0 * sim.capsule_radius
        zoom_view_dir = -sim._poke_normal_world
        if args.camera_zoom_tilt_deg:
            zoom_view_dir = tilt_view_dir(zoom_view_dir, args.camera_zoom_tilt_deg)
        if args.camera_zoom_azimuth_deg:
            # Orbit the (already tilted) camera around the surface normal, i.e.
            # keep the same off-axis angle but look at the patch from a
            # different side -- e.g. one that shows more of the organ's outline.
            zoom_view_dir = Rotation.from_rotvec(
                math.radians(args.camera_zoom_azimuth_deg) * sim._poke_normal_world
            ).apply(zoom_view_dir)
        frame_camera_on_point(zoom_center, zoom_radius, fill_fraction=args.camera_zoom_fill,
                              view_dir=zoom_view_dir)
    else:
        # Frame the camera on the soft body automatically.
        frame_camera_on_soft_body(sim)

    recorder = _VideoRecorder(args.record, args.fps) if args.record else None

    recorded_frames = 0
    try:
        while not ps.window_requests_close():
            frame_start_time = time.perf_counter()

            with wp.ScopedTimer("Step and readback"):
                sim.step()

                sim.render()

            ps.frame_tick()

            if recorder is not None:
                recorder.add_frame()
                recorded_frames += 1
                if args.record_frames and recorded_frames >= args.record_frames:
                    break

            # _throttle_render_fps(frame_start_time, sim.fps)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Wrote {recorded_frames} frames to {args.record}")

if __name__ == "__main__":
    parser = MFEMRefinementSim.create_parser()
    args = init(parser)

    
    refinement_model = MFEMRefinementModel.load(args.model, 1.0, wp.vec3(0.0, args.translate_y, 0.0))

    sim = MFEMRefinementSim(refinement_model, args)

    run(sim, args)
    sim.solver.write_timings("results/data/timings.json")


