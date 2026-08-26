from typing import Any, override

import numpy as np
from mfem.refinement.additional_state import AdditionalState
import warp as wp
import warp.sparse as ws
import warp.optim.linear
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from mfem.boundary_condition import DirichletBoundaryCondition


from mfem.refinement.contact import Contact
from mfem.refinement.elastic_energy import arap_energy
from mfem.refinement.kernels import compute_H_lmbda_G_x_blocks, compute_H_lmbda_blocks, compute_constraints, compute_g_lmbda, extract_attachment_indices, get_particle_velocity, integrate_particles, get_mass_diagonal_and_attachment, kinetic_gradient_kernel, local_solve_lambda, local_solve_stretch, update_constraint_gradient_topology
from mfem.refinement.refinement_human_made import refine, RefinementBuffers
from mfem.types import mat33, mat612, mat63, mat66, vec3, vec6

mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)

class RefinementSolver(SolverBase):
    def __init__(
        self,
        model: Model,
        iterations: int,
        max_tets: int,
        refine_every_n_steps: int = 10,
        min_refine_score: float = 10.1,
        **kwargs,
    ):
        super().__init__(model)
        self._step = -1
        self._timings = {}

        energy_arg = kwargs.get("energy", "arap")
        if energy_arg == "arap":
            self._elastic_energy_model = arap_energy
            self._hessian_update_policy = "on_topology_change"
        else:
            raise ValueError(f"Unknown energy type: {energy_arg}")

        self._update_hessian_blocks = True

        self.max_particles = self.model.particle_count
        self.max_tets = max_tets
        self.refine_every_n_steps = refine_every_n_steps
        self.min_refine_score = min_refine_score

        headroom_factor = max(1, self.max_tets // max(1, model.tet_count))
        original_incident_counts = np.bincount(model.tet_indices.numpy().flatten(), minlength=self.max_particles)
        self._max_incident_tet_counts = int(original_incident_counts.max()) * headroom_factor
        self._additional_state_0 = AdditionalState.from_model(model, max_tets)
        self._additional_state_1 = self._additional_state_0.clone()
        self._refine_buffers = RefinementBuffers(max_tets, self.max_particles, threshold=min_refine_score)

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

        self._H_lmbda_blocks = wp.empty(self.max_tets, dtype=mat66)
        self._g_lmbda_blocks = wp.empty(self.max_tets, dtype=vec6)
        self._grad_constraints_dx_block_count = wp.empty(1, dtype=wp.int32)

        contact_d0 = kwargs.get("contact_d0", 1.0e-2)
        contact_d1 = kwargs.get("contact_d1", 5.0e-2)
        contact_stiffness = kwargs.get("contact_stiffness", 1.0e4)
        self._contact = Contact(model, self.max_particles, contact_d0, contact_d1, contact_stiffness)
        self._contact_hessian_dx = ws.bsr_zeros(
            self.max_particles,
            self.max_particles,
            wp.mat33,
            topology="padded",
            row_capacity=1,
            nnz_capacity=self.max_particles,
        )
        self._contact_hessian_axpy_work_arrays = ws.bsr_axpy_work_arrays()

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
        if self._attachment_count != 0:
            self._attachments = DirichletBoundaryCondition(wp.clone(self.model.particle_q[self._attachment_indices]), self._attachment_indices, wp.full(self._attachment_count, 1.0e1, dtype=wp.float32), self.max_particles)
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

        refine(self.model, 1.0, state_in, state_out, additional_state_0, additional_state_1, self._refine_buffers, self._elastic_energy, self._contact.barrier_energy)

        self._update_constraint_gradient_topology(additional_state_1)
        self._refresh_mass_matrix(additional_state_1)
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

        for i in range(self.iterations):
            with wp.ScopedTimer("Update system", dict=self._timings):
                self._update_system(state_out, additional_state_1, dt)

            with wp.ScopedTimer("Assemble system", dict=self._timings):
                H, g = self._assemble_system(state_out, additional_state_1, dt)

            with wp.ScopedTimer("Global solve", dict=self._timings):
                guess = self._dx
                wp.copy(guess, self._particle_q_tilde - state_out.particle_q)
                dx = self._global_solve(H, g, guess)

            with wp.ScopedTimer("Local solve", dict=self._timings):
                ds, new_lmbda = self._local_solve(state_out, additional_state_1, dx)

            state_out.particle_q += dx
            wp.copy(additional_state_1.tet_lambda, new_lmbda)
            additional_state_1.tet_stretch += ds

            # Here is where we need to add line search

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

    def _global_solve(self, H: ws.BsrMatrix, g: wp.array, guess: wp.array) -> wp.array:
        dx = guess
        warp.optim.linear.cg(H, g, dx, maxiter=1000, check_every=0, tol=1e-6, use_cuda_graph=True)


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
        self._contact.evaluate(self.model, state, additional_state)

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

        return H, g



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

        self._elastic_energy_model(
            state,
            additional_state,
            self._elastic_energy,
            self._gradient_ds,
            self._hessian_ds_blocks,
            self._inv_hessian_ds_blocks,
            update_hessian_blocks=self._update_hessian_blocks
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
