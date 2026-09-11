from typing import Any, override

import numpy as np
from mfem.refinement.additional_state import AdditionalState
import warp as wp
import warp.sparse as ws
import warp.optim.linear
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from mfem.boundary_condition import DirichletBoundaryCondition
from mfem.accumulate import TiledAccumulator
from mfem.line_search import backtracking_line_search
from mfem.utils import linear_accumulate


from mfem.refinement.contact import Contact
from mfem.refinement.elastic_energy import (
    arap_energy,
    arap_energy_only_kernel,
    neohookean_energy,
    neohookean_energy_only_kernel,
)
from mfem.refinement.shell_energy import Shell
from mfem.refinement.kernels import add_constraint_objective, compute_H_lmbda_G_x_blocks, compute_H_lmbda_blocks, compute_constraints, compute_g_lmbda, extract_attachment_indices, get_particle_velocity, integrate_particles, invert_diag_blocks, get_mass_diagonal_and_attachment, kinetic_gradient_kernel, kinetic_objective_kernel, local_solve_lambda, local_solve_stretch, stretch_lagrangian_gradient_kernel, update_constraint_gradient_topology
from mfem.refinement.refinement import refine, RefinementBuffers
from mfem.types import mat33, mat612, mat63, mat66, vec3, vec6

mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)


class _LineSearchState:
    """Minimal particle_q/body_q view so Contact's kernels can evaluate a line-search
    trial position without needing a full Newton State object."""

    def __init__(self, particle_q: wp.array, body_q: wp.array):
        self.particle_q = particle_q
        self.body_q = body_q

class RefinementSolver(SolverBase):
    def __init__(
        self,
        model: Model,
        iterations: int,
        max_tets: int,
        max_particles: int | None = None,
        max_tris: int | None = None,
        refine_every_n_steps: int = 10,
        min_refine_score: float = 10.1,
        **kwargs,
    ):
        super().__init__(model)
        self._step = -1
        self._timings = {}

        # ---- Tunables that were previously hard-coded constants -------------
        # Master switch for adaptive mesh refinement. When disabled, each step
        # forwards additional_state_0 -> additional_state_1 unchanged (no edge
        # splits) and the rest of the solve is identical.
        self._refine_enabled = kwargs.get("enable_refinement", True)

        # Refinement candidate scoring / selection (piped into refine()).
        self._refine_density = kwargs.get("refine_density", 1.0)
        self._refine_tet_score_weight = kwargs.get("refine_tet_score_weight", 0.0)
        self._refine_vertex_score_weight = kwargs.get("refine_vertex_score_weight", 0.1)
        self._penetrating_edge_score = kwargs.get("penetrating_edge_score", 1.0e6)
        self._refine_conflict_iterations = kwargs.get("refine_conflict_iterations", 5)
        self._refine_split_position = kwargs.get("refine_split_position", 0.5)
        self._refine_hashmap_load_factor = kwargs.get("refine_hashmap_load_factor", 0.5)
        self._refine_hashmap_edges_per_tet = kwargs.get("refine_hashmap_edges_per_tet", 6)
        # Scoring mode: "legacy" (the weighted energy sum above) or "geometric"
        # (dimensionless per-edge score from edge length, longest-edge ratio,
        # elastic energy density and tool-contact proximity -- see
        # refinement.populate_candidates_geometric). Geometric scoring reads
        # fresh contact distances at the start of the step and never creates
        # an edge shorter than refine_min_edge_length (default: contact_d1).
        self._refine_scoring = kwargs.get("refine_scoring", "legacy")
        if self._refine_scoring not in ("legacy", "geometric"):
            raise ValueError(f"Unknown refine_scoring {self._refine_scoring!r} (expected 'legacy' or 'geometric')")
        min_edge = kwargs.get("refine_min_edge_length", None)
        self._refine_min_edge_length = float(min_edge) if min_edge is not None else float(kwargs.get("contact_d1", 1.0e-1))
        self._refine_elastic_weight = kwargs.get("refine_elastic_weight", 1.0)
        self._refine_contact_weight = kwargs.get("refine_contact_weight", 1.0)
        self._refine_tri_contact_weight = kwargs.get("refine_tri_contact_weight", 1.0)
        # The geometric score is dimensionless, so it gets its own threshold
        # instead of min_refine_score (which is in the legacy score's units).
        self._refine_geometric_threshold = kwargs.get("refine_geometric_threshold", 1.0)

        # Global CG solve + block-Jacobi preconditioner.
        self._cg_max_iterations = kwargs.get("cg_max_iterations", 1000)
        self._cg_tolerance = kwargs.get("cg_tolerance", 1.0e-6)
        self._cg_check_every = kwargs.get("cg_check_every", 0)
        self._precond_singular_threshold = kwargs.get("preconditioner_singular_threshold", 1.0e-20)

        # Backtracking line search. The threshold default (1e-8) is tuned for
        # refinement's objective convention (kinetic term divided by dt*dt,
        # elastic/constraint/contact unscaled), which sits at a much smaller
        # characteristic magnitude than solver.py's -- see _line_search.
        self._line_search_max_iterations = kwargs.get("line_search_max_iterations", 20)
        self._line_search_alpha0 = kwargs.get("line_search_alpha0", 1.0)
        self._line_search_tau = kwargs.get("line_search_tau", 0.5)
        self._line_search_c = kwargs.get("line_search_c", 0.01)
        self._line_search_threshold = kwargs.get("line_search_threshold", 1.0e-8)

        # Stiffness of the Dirichlet constraint pinning zero-inv-mass particles.
        self._attachment_stiffness = kwargs.get("attachment_stiffness", 1.0e1)

        energy_arg = kwargs.get("energy", "arap")
        if energy_arg == "arap":
            self._elastic_energy_model = arap_energy
            self._elastic_energy_only_kernel = arap_energy_only_kernel
            # The ARAP Hessian in s is constant per tet (mu * vol * SYM), so the
            # blocks only need refreshing when refinement changes the topology.
            self._hessian_update_policy = "on_topology_change"
        elif energy_arg == "neohookean":
            self._elastic_energy_model = neohookean_energy
            self._elastic_energy_only_kernel = neohookean_energy_only_kernel
            # The Neo-Hookean Hessian depends on the current stretch, so the
            # blocks must be rebuilt every Newton iteration.
            self._hessian_update_policy = "every_iteration"
        else:
            raise ValueError(f"Unknown energy type: {energy_arg}")

        self._update_hessian_blocks = True

        # Block-Jacobi preconditioner for the global CG solve. Disable with
        # preconditioner=False to run plain (unpreconditioned) CG.
        self._use_preconditioner = kwargs.get("preconditioner", True)

        if max_particles is not None and max_particles != model.particle_count:
            raise ValueError(
                f"max_particles={max_particles} does not match model.particle_count="
                f"{model.particle_count}; the model's particle buffer must already be "
                "padded to max_particles by the caller (e.g. load_sim_model(..., "
                "max_particles=...)) before the solver is constructed"
            )
        self.max_particles = model.particle_count
        self.max_tets = max_tets
        self.refine_every_n_steps = refine_every_n_steps
        self.min_refine_score = min_refine_score
        if self._refine_scoring == "geometric":
            self.min_refine_score = self._refine_geometric_threshold

        # Round *up* so a tightened max_tets (base*mult + slack, which can land
        # below an integer multiple of tet_count) never collapses this to 1 and
        # starves the incident-tet / surface-tri capacities below.
        headroom_factor = max(1, -(-self.max_tets // max(1, model.tet_count)))
        original_incident_counts = np.bincount(model.tet_indices.numpy().flatten(), minlength=self.max_particles)
        self._max_incident_tet_counts = int(original_incident_counts.max()) * headroom_factor

        # Surface-triangle capacity: prefer the ceiling baked into the mesh .npz
        # (passed through as max_tris=, tuned to measured refinement growth); a
        # 0 / missing value falls back to the same headroom factor as max_tets
        # applied to the model's initial surface-tri count (model.tri_indices, as
        # computed by newton's add_soft_mesh).
        initial_tri_count = model.tri_indices.shape[0]
        self.max_tris = max_tris or (initial_tri_count * headroom_factor)

        self._additional_state_0 = AdditionalState.from_model(model, max_tets, self.max_tris)
        self._additional_state_1 = self._additional_state_0.clone()
        self._refine_buffers = RefinementBuffers(
            max_tets,
            self.max_particles,
            self.max_tris,
            threshold=self.min_refine_score,
            hashmap_load_factor=self._refine_hashmap_load_factor,
            hashmap_edges_per_tet=self._refine_hashmap_edges_per_tet,
        )

        # Scratch memory
        self._bsr_axpy_work_arrays = ws.bsr_axpy_work_arrays()
        self._bsr_mm_work_arrays = ws.bsr_mm_work_arrays()

        # For building BSR matrices
        self._seq = wp.array(list(range(max(self.max_tets, self.max_particles))), dtype=wp.int32)
        self._grad_constraints_dx_rows = wp.zeros(self.max_tets * 4, dtype=wp.int32)
        self._grad_constraints_dx_cols = wp.zeros(self.max_tets * 4, dtype=wp.int32)
        self._grad_constraints_dx_blocks = wp.zeros(self.max_tets * 4, dtype=mat63)
        self._H_lmbda_grad_constraints_dx_blocks = wp.zeros(self.max_tets * 4, dtype=mat63)

        self._particle_q_tilde = wp.zeros(self.max_particles, dtype=wp.vec3)

        self._elastic_energy = wp.zeros(self.max_tets, dtype=wp.float32)

        self._do_line_search = kwargs.get("line_search", True)
        self._particle_kinetic_energy = wp.zeros(self.max_particles, dtype=wp.float32)
        self._tet_objective_energy = wp.zeros(self.max_tets, dtype=wp.float32)
        self._stretch_lagrangian_gradient = wp.zeros(self.max_tets, dtype=vec6)
        self._accumulator = TiledAccumulator(
            max_length=max(3 * self.max_particles, 6 * self.max_tets),
            device=self.model.device,
            scalar_type=wp.float32,
        )

        self._precompute_mass_matrix_and_attachments()
        self._update_constraint_gradient_topology(self._additional_state_0)

        self._hessian_ds_blocks = wp.empty(self.max_tets, dtype=mat66)
        self._inv_hessian_ds_blocks = wp.empty(self.max_tets, dtype=mat66)

        self._grad_constraints_dx = ws.bsr_zeros(self.max_tets, self.max_particles, mat63, topology="padded", row_capacity=4, nnz_capacity=self.max_tets * 4)
        self._transpose_grad_constraints_dx = ws.bsr_zeros(self.max_particles, self.max_tets, mat36, topology="padded", row_capacity=self._max_incident_tet_counts) #, nnz_capacity=self.max_particles * self.max_incident_tets)
        self._global_lhs = ws.bsr_zeros(self.max_particles, self.max_particles, wp.mat33, topology="padded", row_capacity=(1 + self._max_incident_tet_counts * 3)) #, nnz_capacity=self.max_particles * (1 + self.max_incident_tets * 3))
        self._global_rhs = wp.zeros(self.max_particles, dtype=wp.vec3)

        self._gradient_dx = wp.zeros(self.max_particles, dtype=wp.vec3)
        self._gradient_ds = wp.zeros(self.max_tets, dtype=vec6)

        self._constraints = wp.empty(self.max_tets, dtype=vec6)

        self._dx = wp.zeros(self.max_particles, dtype=wp.vec3)
        self._ds = wp.zeros(self.max_tets, dtype=vec6)

        # Block-Jacobi preconditioner for the global CG solve: M ~= blockdiag(H)^-1.
        # The inv-diag scratch is refreshed in place from the current H at the top
        # of every _global_solve (see invert_diag_blocks); aslinearoperator wraps
        # the 1-D mat33 array as a block-diagonal operator (r[i] <- M[i] @ r[i]).
        self._precond_diag_blocks = wp.zeros(self.max_particles, dtype=wp.mat33)
        self._precond_inv_diag_blocks = wp.zeros(self.max_particles, dtype=wp.mat33)
        self._precond = (
            warp.optim.linear.aslinearoperator(self._precond_inv_diag_blocks)
            if self._use_preconditioner
            else None
        )
        self._cg = warp.optim.linear.cg(
            self._global_lhs,
            self._global_rhs,
            self._dx,
            maxiter=self._cg_max_iterations,
            check_every=self._cg_check_every,
            tol=self._cg_tolerance,
            use_cuda_graph=True,
            M=self._precond,
            run=False,
        )

        self._H_lmbda_blocks = wp.empty(self.max_tets, dtype=mat66)
        self._g_lmbda_blocks = wp.empty(self.max_tets, dtype=vec6)
        self._grad_constraints_dx_block_count = wp.empty(1, dtype=wp.int32)

        contact_d0 = kwargs.get("contact_d0", 1.0e-2)
        contact_d1 = kwargs.get("contact_d1", 1.0e-1)
        contact_stiffness = kwargs.get("contact_stiffness", 1.0e7)
        friction_mu = kwargs.get("friction_mu", 0.0)
        friction_eps_v = kwargs.get("friction_eps_v", 1.0e-3)
        # Surface-tri incidence bound (with the same refinement headroom as
        # max_tets) sizes the BSR row capacity of the per-tri contact and
        # shell Hessians.
        tri_incident_counts = np.bincount(
            model.tri_indices.numpy().flatten(), minlength=self.max_particles
        )
        self._max_incident_tri_counts = int(tri_incident_counts.max()) * headroom_factor
        # tri_contact=False falls back to vertex-only contact (the tool can then
        # slip between surface vertices wider apart than its diameter).
        self._contact = Contact(
            model,
            self.max_particles,
            contact_d0,
            contact_d1,
            contact_stiffness,
            friction_mu,
            friction_eps_v,
            max_tris=self.max_tris,
            max_incident_tris=self._max_incident_tri_counts,
            tri_contact=kwargs.get("tri_contact", True),
        )

        self._contact_hessian_dx = ws.bsr_zeros(
            self.max_particles,
            self.max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=1,
            nnz_capacity=self.max_particles,
        )
        self._contact_hessian_axpy_work_arrays = ws.bsr_axpy_work_arrays()
        self._tri_contact_hessian_axpy_work_arrays = ws.bsr_axpy_work_arrays()

        # Lagged-friction Hessian: same per-particle diagonal-block structure as
        # the barrier Hessian, assembled and added into H separately.
        self._friction_hessian_dx = ws.bsr_zeros(
            self.max_particles,
            self.max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=1,
            nnz_capacity=self.max_particles,
        )
        self._friction_hessian_axpy_work_arrays = ws.bsr_axpy_work_arrays()

        # Membrane shell elements over the surface tris (additional_state.
        # tri_indices). shell_mu is a *surface* shear modulus in Pa*m (bulk
        # Lame mu of the shell material times its thickness); 0 disables.
        # Every surface tri is a face of an active tet, so the shell
        # Hessian's (i, j) blocks are a subset of the global lhs's existing
        # sparsity and the bsr_axpy below never grows its topology.
        shell_mu = kwargs.get("shell_mu", 0.0)
        self._shell = None
        if shell_mu > 0.0:
            self._shell = Shell(self.max_particles, self.max_tris, shell_mu, self._max_incident_tri_counts)
            self._shell_hessian_axpy_work_arrays = ws.bsr_axpy_work_arrays()
            self._shell.refresh(self._additional_state_0)

        self.iterations = iterations

    def _update_constraint_gradient_topology(self, additional_state: AdditionalState):
        wp.launch(
            update_constraint_gradient_topology,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                additional_state.tet_indices,
            ],
            outputs=[
                self._grad_constraints_dx_rows,
                self._grad_constraints_dx_cols,
            ],
        )

    def _precompute_mass_matrix_and_attachments(self):
        attachment_predicate = wp.zeros(self.max_particles + 1, dtype=wp.int32)
        self._mass_diagonal_blocks = wp.empty(self.max_particles, dtype=wp.mat33)
        self._diag_row_indices = wp.empty(self.max_particles, dtype=wp.int32)
        self._diag_col_indices = wp.empty_like(self._diag_row_indices)
        # Scratch output for _refresh_mass_matrix's reuse of
        # get_mass_diagonal_and_attachment; attachments are static (never
        # re-derived after a refine pass) so this is never read there.
        self._refine_attachment_predicate_scratch = wp.empty(self.max_particles, dtype=wp.int32)

        wp.launch(
            get_mass_diagonal_and_attachment,
            dim=self.model.particle_count,
            inputs=[
                self._additional_state_0.active_particle_count,
                self.model.particle_inv_mass
            ],
            outputs=[
                attachment_predicate,
                self._diag_row_indices,
                self._diag_col_indices,
                self._mass_diagonal_blocks,
            ]
        )

        # D2H Copy here
        wp.utils.array_scan(attachment_predicate, attachment_predicate, inclusive=False)
        self._attachment_count = int(attachment_predicate[-1:].numpy()[0])
        self._attachment_indices = wp.empty(self._attachment_count, dtype=wp.int32)

        wp.launch(
            extract_attachment_indices,
            dim=self.max_particles,
            inputs=[attachment_predicate],
            outputs=[self._attachment_indices],
        )

        self._mass_matrix = ws.bsr_zeros(
            self.max_particles,
            self.max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=1,
            nnz_capacity=self.max_particles,
        )

        ws.bsr_set_from_triplets(
            self._mass_matrix,
            self._diag_row_indices,
            self._diag_row_indices,
            self._mass_diagonal_blocks,
            count=self._additional_state_0.active_particle_count,
            topology="padded",
        )
        self._attachments = None

        # Preallocated + rebuilt via bsr_axpy (rather than the `+`
        # operator, whose allocation behavior we don't want to depend on)
        # so this can be refreshed from inside a captured refine pass.
        self._hessian_dx = ws.bsr_zeros(
            self.max_particles,
            self.max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=1,
            nnz_capacity=self.max_particles,
        )
        self._hessian_refresh_work_arrays = ws.bsr_axpy_work_arrays()
        ws.bsr_axpy(self._mass_matrix, self._hessian_dx, alpha=1.0, beta=0.0, work_arrays=self._hessian_refresh_work_arrays, topology="padded")

        if self._attachment_count != 0:
            self._attachments = DirichletBoundaryCondition(wp.clone(self.model.particle_q[self._attachment_indices]), self._attachment_indices, wp.full(self._attachment_count, self._attachment_stiffness, dtype=wp.float32), self.max_particles)
         
            ws.bsr_axpy(self._attachments.hessian(), self._hessian_dx, alpha=1.0, beta=1.0, work_arrays=self._hessian_refresh_work_arrays, topology="padded")

    def _refresh_mass_matrix(self, additional_state: AdditionalState):
        """Re-derives self._mass_matrix (and, if present, self._hessian_dx)
        for the just-refined additional_state. Attachment points themselves
        are not re-derived: new vertices are never automatically pinned."""
        wp.launch(
            get_mass_diagonal_and_attachment,
            dim=self.max_particles,
            inputs=[
                additional_state.active_particle_count,
                self.model.particle_inv_mass,
            ],
            outputs=[
                self._refine_attachment_predicate_scratch,
                self._diag_row_indices,
                self._diag_col_indices,
                self._mass_diagonal_blocks,
            ],
        )

        ws.bsr_set_from_triplets(
            self._mass_matrix,
            self._diag_row_indices,
            self._diag_row_indices,
            self._mass_diagonal_blocks,
            count=additional_state.active_particle_count,
            topology="padded",
        )

        if self._attachments is not None:
            ws.bsr_axpy(self._mass_matrix, self._hessian_dx, alpha=1.0, beta=0.0, work_arrays=self._hessian_refresh_work_arrays, topology="padded")
            ws.bsr_axpy(self._attachments.hessian(), self._hessian_dx, alpha=1.0, beta=1.0, work_arrays=self._hessian_refresh_work_arrays, topology="padded")



    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        additional_state_0 = self._additional_state_0
        additional_state_1 = self._additional_state_1
        self._step += 1

        # state_out becomes the live working buffer immediately, so
        # refinement can write new-vertex data into it; state_in is never
        # mutated by this solver from this point on.
        state_out.assign(state_in)

        if self._refine_enabled:
            scoring_kwargs = {}
            if self._refine_scoring == "geometric":
                # Signed distances for the *current* tool pose; the barrier
                # energies left over from the previous step's Newton loop are
                # one step stale, and the geometric score wants distances, not
                # energies, anyway.
                self._contact.compute_distance(self.model, state_in, additional_state_0)
                scoring_kwargs = dict(
                    scoring="geometric",
                    particle_distance=self._contact.distance,
                    particle_shape_id=self._contact.shape_id,
                    tri_distance=self._contact.tri_distance if self._contact.tri_contact else None,
                    tri_bary=self._contact.tri_bary if self._contact.tri_contact else None,
                    contact_d1=self._contact.d1,
                    min_edge_length=self._refine_min_edge_length,
                    elastic_weight=self._refine_elastic_weight,
                    vertex_contact_weight=self._refine_contact_weight,
                    tri_contact_weight=self._refine_tri_contact_weight,
                )
            refine(
                self.model,
                self._refine_density,
                state_in,
                state_out,
                additional_state_0,
                additional_state_1,
                self._refine_buffers,
                self._elastic_energy,
                self._contact.barrier_energy,
                tet_score_weight=self._refine_tet_score_weight,
                vertex_score_weight=self._refine_vertex_score_weight,
                penetrating_edge_score=self._penetrating_edge_score,
                conflict_iterations=self._refine_conflict_iterations,
                split_position=self._refine_split_position,
                refine_every=max(int(self.refine_every_n_steps), 1),
                **scoring_kwargs,
            )
        else:
            # No refinement: forward the topology + per-tet fields unchanged so
            # additional_state_1 is still fully valid for the solve below.
            additional_state_0.asign(additional_state_1)

        self._update_constraint_gradient_topology(additional_state_1)
        self._refresh_mass_matrix(additional_state_1)
        if self._shell is not None:
            self._shell.refresh(additional_state_1)
        self._update_hessian_blocks = True
        # additional_state_1 is now fully valid (topology + all per-tet
        # fields + active_tet_count/active_particle_count) regardless of
        # whether an actual refine pass fired this step.

        wp.copy(state_in.particle_q, state_out.particle_q)

        wp.launch(
            integrate_particles,
            dim=self.max_particles,
            inputs=[
                additional_state_1.active_particle_count,
                state_out.particle_q,
                state_out.particle_qd,
                self.model.gravity,
                dt,
            ],
            outputs=[
                self._particle_q_tilde,
            ],
        )

        # Freeze the lagged-friction anchor (start-of-step positions + normal
        # force) before the Newton iterations. state_out.particle_q still holds
        # the start-of-step positions here -- integrate_particles only wrote
        # self._particle_q_tilde.
        self._contact.begin_step(self.model, state_out, additional_state_1)

        for i in range(self.iterations):
            with wp.ScopedTimer("Update system", dict=self._timings):
                self._update_system(state_out, additional_state_1, dt)

            with wp.ScopedTimer("Assemble system", dict=self._timings):
                H, g = self._assemble_system(state_out, additional_state_1, dt)

            with wp.ScopedTimer("Global solve", dict=self._timings):
                guess = self._dx
                wp.copy(guess, self._particle_q_tilde - state_out.particle_q)
                dx = self._global_solve(H, g, guess, additional_state_1)

            with wp.ScopedTimer("Local solve", dict=self._timings):
                ds, new_lmbda = self._local_solve(state_out, additional_state_1, dx)

            wp.copy(additional_state_1.tet_lambda, new_lmbda)

            if self._do_line_search:
                with wp.ScopedTimer("Line search", dict=self._timings):
                    self._line_search(
                        state_out.particle_q,
                        additional_state_1.tet_stretch,
                        additional_state_1.tet_lambda,
                        additional_state_1,
                        state_out.body_q,
                        self._transpose_grad_constraints_dx,
                        self._gradient_dx,
                        self._gradient_ds,
                        self._particle_q_tilde,
                        dx,
                        ds,
                        dt,
                    )
            else:
                state_out.particle_q += dx
                additional_state_1.tet_stretch += ds

        # Deliberately guarded on the OLD (pre-refinement) count, not
        # additional_state_1's: this finite-differences against
        # state_in.particle_q, which has no meaningful "before" for a vertex
        # that didn't exist before this step. New vertices keep the
        # interpolated velocity claim_new_vertices() already wrote into
        # state_out.particle_qd instead.
        wp.launch(
            get_particle_velocity,
            dim=self.max_particles,
            inputs=[
                additional_state_1.active_particle_count,
                state_out.particle_q,
                state_in.particle_q,
                dt,
            ],
            outputs=[
                state_out.particle_qd,
            ],
        )

        self._additional_state_1.asign(self._additional_state_0)

    def _global_solve(
        self, H: ws.BsrMatrix, g: wp.array, guess: wp.array, additional_state: AdditionalState
    ) -> wp.array:
        dx = guess

        # Refresh the block-Jacobi preconditioner M ~= blockdiag(H)^-1 from the
        # current H (H changes every Newton iteration: 1/dt^2 mass term, contact
        # Hessian, ...). Done here, outside the CG iteration that use_cuda_graph
        # captures, and allocation-free thanks to the preallocated scratch and
        # bsr_get_diag's out= argument.
        if self._use_preconditioner:
            ws.bsr_get_diag(H, out=self._precond_diag_blocks)
            wp.launch(
                invert_diag_blocks,
                dim=self.max_particles,
                inputs=[
                    additional_state.active_particle_count,
                    self._precond_diag_blocks,
                    self._precond_singular_threshold,
                ],
                outputs=[self._precond_inv_diag_blocks],
            )

        # self._cg was constructed against (self._global_lhs, self._global_rhs,
        # self._dx); H/g/guess are those same objects (guess is self._dx), so the
        # bare call reuses the preallocated CG buffers and captured graph.
        self._cg()

        return dx

    def _local_solve(
        self, state: State, additional_state: AdditionalState, dx: wp.array
    ) -> tuple[wp.array, wp.array]:
        new_lmbda = additional_state.tet_lambda
        ds = self._ds

        wp.launch(
            local_solve_lambda,
            dim=self.max_tets,
            inputs=[additional_state.active_tet_count, self._H_lmbda_blocks, self._g_lmbda_blocks, self._grad_constraints_dx_blocks, self._grad_constraints_dx_cols, dx],
            outputs=[new_lmbda],
        )

        wp.launch(
            local_solve_stretch,
            dim=self.max_tets,
            inputs=[additional_state.active_tet_count, self._inv_hessian_ds_blocks, self._gradient_ds, new_lmbda],
            outputs=[ds],
        )

        return ds, new_lmbda

    def _update_system(self, state: State, additional_state: AdditionalState, dt: float):
        self._compute_kinetic_derivatives(state, additional_state, dt)
        self._compute_elastic_derivatives(state, additional_state)
        self._compute_constraints(state, additional_state)
        self._contact.evaluate(self.model, state, additional_state, dt)
        if self._shell is not None:
            self._shell.evaluate(state, additional_state)

    def _assemble_system(self, state: State, additional_state: AdditionalState, dt: float) -> tuple[ws.BsrMatrix, wp.array]:

        g_s = self._gradient_ds
        g_x = self._gradient_dx

        constraints = self._constraints
        H_lmbda_blocks = self._H_lmbda_blocks
        g_lmbda = self._g_lmbda_blocks
        G_xt = self._transpose_grad_constraints_dx
        H_lmbda_G_x = self._grad_constraints_dx
        H = self._global_lhs
        g = self._global_rhs


        wp.launch(
            compute_H_lmbda_blocks,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                self._hessian_ds_blocks,
            ],
            outputs=[
                H_lmbda_blocks,
            ],
        )

        wp.launch(
            compute_g_lmbda,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                constraints,
                g_s,
                self._inv_hessian_ds_blocks,
            ],
            outputs=[
                g_lmbda,
            ],
        )

        # Avoiding allocation so that we can capture this in a conditional Cuda Graph loop if we want to
        wp.copy(self._grad_constraints_dx_block_count, additional_state.active_tet_count)
        self._grad_constraints_dx_block_count *= 4

        ws.bsr_set_from_triplets(
            G_xt,
            rows=self._grad_constraints_dx_cols,
            columns=self._grad_constraints_dx_rows,
            values=wp.map(wp.transpose, self._grad_constraints_dx_blocks),
            count=self._grad_constraints_dx_block_count,
            topology="padded"
        )


        wp.launch(
            compute_H_lmbda_G_x_blocks,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                H_lmbda_blocks,
                self._grad_constraints_dx_blocks,
            ],
            outputs=[
                self._H_lmbda_grad_constraints_dx_blocks,
            ],
        )

        ws.bsr_set_from_triplets(
            H_lmbda_G_x,
            rows=self._grad_constraints_dx_rows,
            columns=self._grad_constraints_dx_cols,
            values=self._H_lmbda_grad_constraints_dx_blocks,
            count=self._grad_constraints_dx_block_count,
            topology="padded"
        )


        ws.bsr_mm(
            G_xt,
            H_lmbda_G_x,
            H,
            work_arrays=self._bsr_mm_work_arrays,
            topology="padded",
        )

        wp.copy(g, g_x)
        ws.bsr_mv(
            H_lmbda_G_x,
            g_lmbda,
            g,
            transpose=True,
            beta=-1.0,
        )
        g -= self._contact.barrier_gradient
        g -= self._contact.friction_gradient

        ws.bsr_axpy(
            self._hessian_dx,
            H,
            alpha=1.0 / (dt * dt),
            work_arrays=self._bsr_axpy_work_arrays,
            topology="padded",
        )

        ws.bsr_set_from_triplets(
            self._contact_hessian_dx,
            self._diag_row_indices,
            self._diag_col_indices,
            self._contact.barrier_hessian_blocks,
            count=additional_state.active_particle_count,
            topology="padded",
        )
        ws.bsr_axpy(
            self._contact_hessian_dx,
            H,
            alpha=1.0,
            work_arrays=self._contact_hessian_axpy_work_arrays,
            topology="padded",
        )

        if self._contact.tri_contact:
            # Per-tri contact Hessian (blocks on surface-tri vertex pairs, all
            # already present in H's sparsity). Its gradient/energy are folded
            # into barrier_gradient / barrier_energy by Contact.evaluate.
            ws.bsr_axpy(
                self._contact.tri_barrier_hessian,
                H,
                alpha=1.0,
                work_arrays=self._tri_contact_hessian_axpy_work_arrays,
                topology="padded",
            )

        ws.bsr_set_from_triplets(
            self._friction_hessian_dx,
            self._diag_row_indices,
            self._diag_col_indices,
            self._contact.friction_hessian_blocks,
            count=additional_state.active_particle_count,
            topology="padded",
        )
        ws.bsr_axpy(
            self._friction_hessian_dx,
            H,
            alpha=1.0,
            work_arrays=self._friction_hessian_axpy_work_arrays,
            topology="padded",
        )

        if self._shell is not None:
            g -= self._shell.gradient
            ws.bsr_axpy(
                self._shell.hessian,
                H,
                alpha=1.0,
                work_arrays=self._shell_hessian_axpy_work_arrays,
                topology="padded",
            )

        return H, g



    def _objective(
        self,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        additional_state: AdditionalState,
        body_q: wp.array[wp.transform],
        x_tilde: wp.array[wp.vec3],
        dt: float,
        objective: wp.array[wp.float32],
    ) -> wp.array:
        """Evaluates the Lagrangian merit function (lambda held fixed) at a trial (x, s).
        Used by the line search, so this only ever needs the scalar energy -- the *_only
        kernels/methods skip the gradient/Hessian work that the main Newton iteration needs."""

        wp.launch(
            kinetic_objective_kernel,
            dim=self.max_particles,
            inputs=[additional_state.active_particle_count, x, x_tilde, self.model.particle_mass],
            outputs=[self._particle_kinetic_energy],
        )
        self._accumulator.compute_sum(self._particle_kinetic_energy)
        linear_accumulate(objective, self._accumulator.result(), alpha=0.0, beta=1.0 / (dt * dt))

        if self._attachments is not None:
            self._attachments.energy(x, self._accumulator)
            linear_accumulate(objective, self._accumulator.result(), beta=1.0 / (dt * dt))

        wp.launch(
            self._elastic_energy_only_kernel,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                s,
                additional_state.tet_materials,
                additional_state.tet_poses,
            ],
            outputs=[self._tet_objective_energy],
        )
        wp.launch(
            add_constraint_objective,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                additional_state.tet_indices,
                additional_state.tet_poses,
                x,
                s,
                lmbda,
            ],
            outputs=[self._tet_objective_energy],
        )
        self._accumulator.compute_sum(self._tet_objective_energy)
        linear_accumulate(objective, self._accumulator.result(), beta=1.0)

        self._contact.evaluate_energy_only(self.model, _LineSearchState(x, body_q), additional_state, dt)
        self._accumulator.compute_sum(self._contact.barrier_energy)
        linear_accumulate(objective, self._accumulator.result(), beta=1.0)
        self._accumulator.compute_sum(self._contact.friction_energy)
        linear_accumulate(objective, self._accumulator.result(), beta=1.0)

        if self._shell is not None:
            self._shell.evaluate_energy_only(x, additional_state)
            self._accumulator.compute_sum(self._shell.energy)
            linear_accumulate(objective, self._accumulator.result(), beta=1.0)

        return objective

    def _line_search(
        self,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        additional_state: AdditionalState,
        body_q: wp.array[wp.transform],
        G_xt: ws.BsrMatrix,
        g_x: wp.array[wp.vec3],
        g_s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dx: wp.array[wp.vec3],
        ds: wp.array[vec6],
        dt: float,
    ):
        # Line-search parameters (max_iterations/alpha0/tau/c/threshold) are
        # configured on the solver -- see the constructor. The threshold default
        # (1e-8) is deliberately much tighter than solver.py's 1e-4: that value was
        # tuned for its objective convention (kinetic term unscaled, elastic/
        # constraint terms scaled by dt*dt), whereas refinement's convention
        # divides the kinetic term by dt*dt and leaves elastic/constraint/contact
        # unscaled, so its objective sits at a much smaller characteristic
        # magnitude (empirically O(1e-4)-O(1e-3) even for a modest mesh).
        def objective_fn(dof_arrays: tuple[wp.array], objective: wp.array[wp.float32]):
            self._objective(dof_arrays[0], dof_arrays[1], lmbda, additional_state, body_q, x_tilde, dt, objective)

        # Newton solves against the stretch's own Lagrangian gradient directly (see
        # local_solve_stretch); mirror that here rather than the sign convention
        # solver.py uses for its x/s gradients.
        wp.launch(
            stretch_lagrangian_gradient_kernel,
            dim=self.max_tets,
            inputs=[additional_state.active_tet_count, g_s, lmbda],
            outputs=[self._stretch_lagrangian_gradient],
        )

        dof_arrays = (x, s)
        search_direction = (dx, ds)
        gradient_x = g_x + self._contact.barrier_gradient + self._contact.friction_gradient - G_xt @ lmbda
        if self._shell is not None:
            gradient_x = gradient_x + self._shell.gradient
        gradient = (
            gradient_x,
            self._stretch_lagrangian_gradient,
        )

        return backtracking_line_search(
            objective_fn,
            dof_arrays,
            search_direction,
            gradient,
            self._accumulator,
            self._line_search_max_iterations,
            self._line_search_alpha0,
            self._line_search_tau,
            self._line_search_c,
            self._line_search_threshold,
        )

    def _compute_kinetic_derivatives(self, state: State, additional_state: AdditionalState, dt: float):
        wp.launch(
            kinetic_gradient_kernel,
            dim=self.max_particles,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                self._particle_q_tilde,
                self.model.particle_mass,
                dt,
            ],
            outputs=[
                self._gradient_dx,
            ],
        )

        if self._attachments is not None:
            self._gradient_dx += self._attachments.gradient(state.particle_q) / (dt * dt)

    def _compute_elastic_derivatives(self, state: State, additional_state: AdditionalState):

        update_hessian_blocks = (
            self._update_hessian_blocks
            or self._hessian_update_policy == "every_iteration"
        )
        self._elastic_energy_model(
            state,
            additional_state,
            self._elastic_energy,
            self._gradient_ds,
            self._hessian_ds_blocks,
            self._inv_hessian_ds_blocks,
            update_hessian_blocks=update_hessian_blocks,
        )
        self._update_hessian_blocks = False

    def _compute_constraints(self, state: State, additional_state: AdditionalState):

        wp.launch(
            compute_constraints,
            dim=self.max_tets,
            inputs=[
                additional_state.active_tet_count,
                additional_state.tet_indices,
                additional_state.tet_poses,
                state.particle_q,
                additional_state.tet_stretch,
                self._constraints,
                self._grad_constraints_dx_blocks,
            ],
        )

    def write_timings(self, filename: str):
        import json
        with open(filename, "w") as f:
            json.dump(self._timings, f)
