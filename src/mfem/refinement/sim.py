from mfem.refinement.solver import RefinementSolver
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


@wp.kernel
def _translate_bodies_kernel(body_q: wp.array(dtype=wp.transform), offset: wp.vec3):
    tid = wp.tid()
    tf = body_q[tid]
    pos = wp.transform_get_translation(tf) + offset
    rot = wp.transform_get_rotation(tf)
    body_q[tid] = wp.transform(pos, rot)


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



        builder = newton.ModelBuilder(gravity=-9.81)

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

        # Size the capsule relative to the octopus so it reads as a small
        # vertical "poker" hovering just above it.
        self.capsule_radius = 0.09 * float(max(octo_ext[0], octo_ext[1]))
        self.capsule_half_height = 0.183 * float(octo_ext[0])
        self.capsule_axis = Axis.Z

        # Default pose: centred over the octopus, standing vertically (the
        # Z-aligned capsule needs no rotation) with its lower cap a small
        # gap above the octopus top.
        gap = 0.5 * self.capsule_radius
        self._capsule_pos_default = np.array(
            [octo_center[0], octo_center[1],
             octo_hi[2] + gap + self.capsule_half_height + self.capsule_radius],
            dtype=np.float64,
        )
        self._capsule_rot_default = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # XYZ euler, degrees

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
        # width/length = 0 makes it infinite for collision.
        self.ground_height = 0.0
        self.ground_extent = 10.0
        builder.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_height),
            width=0.0,
            length=0.0,
        )

        max_particles = refinement_model.resolve_max_particles(args.max_particles_mult)
        max_tets = refinement_model.resolve_max_tets(args.max_tets_mult)

        self.model = refinement_model.load_sim_model(builder, args.mu, max_particles=max_particles)
        max_tris = refinement_model.resolve_max_tris(args.max_tris_mult, self.model)


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
            refine_geometric_threshold=args.refine_geometric_threshold,
            cg_max_iterations=args.cg_max_iterations,
            cg_tolerance=args.cg_tolerance,
            cg_check_every=args.cg_check_every,
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
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        particle_ct = self.solver._additional_state_0.active_particle_count.numpy()[0]
        self.ps_volume = ps.register_volume_mesh("Soft body", self.model.particle_q[:particle_ct].numpy(), self.model.tet_indices.numpy())
        self.ps_volume.add_scalar_quantity("elastic energy", self.solver._elastic_energy[:self.solver._additional_state_0.active_tet_count.numpy()[0]].numpy(), defined_on='cells', vminmax=(0.0, 10.0), cmap='blues', enabled=True)
        self._old_vertex_count = particle_ct

        capsule_mesh = newton.Mesh.create_capsule(
            self.capsule_radius, self.capsule_half_height, up_axis=self.capsule_axis
        )
        self.ps_capsule = ps.register_surface_mesh(
            "Capsule", capsule_mesh.vertices, capsule_mesh.indices.reshape(-1, 3)
        )
        self.ps_capsule.set_color([0.85, 0.55, 0.15])
        self._sync_capsule()

        # Drive the capsule from the polyscope window (ImGui panel + hotkeys).
        ps.set_user_callback(self._capsule_gui_callback)

        e = self.ground_extent
        h = self.ground_height
        ground_verts = np.array(
            [[-e, -e, h], [e, -e, h], [e, e, h], [-e, e, h]], dtype=np.float32
        )
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.ps_ground = ps.register_surface_mesh("Ground", ground_verts, ground_faces)
        # self.ps_ground.set_enabled(False)

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
            "--mu",
            help="First Lame's parameter for elasticity",
            type=float,
            default=1.0e2,
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
    # render() re-registers a fresh volume mesh object each frame.
    ps.set_up_dir('z_up')
    # Hide polyscope's built-in ground plane.
    ps.set_ground_plane_mode('none')
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

    
    refinement_model = MFEMRefinementModel.load("models/octopus_frame0_coarse.npz", 1.0, wp.vec3(0.0, 0.0, 0.0))

    sim = MFEMRefinementSim(refinement_model, args)

    run(sim, args)
    sim.solver.write_timings("timings.json")


