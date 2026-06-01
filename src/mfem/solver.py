# import newton


import numpy as np
import warp as wp
import warp.sparse
import warp.sparse as ws
from matplotlib.pyplot import plot
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase
from warp.optim.linear import bicgstab, cg

from .kernels import (
    evaluate_constraint_gradient_dx,
    evaluate_constraints,
    precompute_bsr_topology,
    precompute_mass_matrix,
    precompute_rest,
    precompute_tet_stretch,
    test_deform_kernel,
)
from .material_model import StretchMaterialModel
from .types import mat63, mat66, vec6
from .utils import invert_diagonal_bsr, plot_bsr


class MFEMSolver(SolverBase):
    def __init__(
        self,
        model: Model,
        material_model: StretchMaterialModel,
        iterations: int,
    ):
        super().__init__(model)

        self.solver_iterations = iterations
        self._material_model = material_model
        self._work_arrays = {
            "HsGsi": ws.bsr_mm_work_arrays(),
            "Hlmbda": ws.bsr_mm_work_arrays(),
            "GxtHlmbda": ws.bsr_mm_work_arrays(),
            "GxtHlmbdaGx": ws.bsr_mm_work_arrays(),
        }

        print("Particles:", self.model.particle_count, "Tets:", self.model.tet_count)
        # self._B = subspace
        #

        self._initial_precompute()

    def _test_deform(self):

        wp.launch(
            test_deform_kernel,
            dim=self.model.particle_count,
            inputs=[self.model.particle_q],
            device=self.model.device,
        )

        wp.synchronize()
        print("test deform done")

    def _initial_precompute(self):

        self.tet_rest_volumes = wp.empty(self.model.tet_count)
        wp.launch(
            precompute_rest,
            dim=self.model.tet_count,
            inputs=[self.model.particle_q, self.model.tet_indices],
            outputs=[self.model.tet_poses, self.tet_rest_volumes],
            device=self.model.device,
        )

        wp.synchronize()
        print("precompute rest done")

        mass_diag = None
        if self.model.particle_mass is not None:
            mass_diag = (
                wp.full(
                    shape=self.model.particle_count,
                    value=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=wp.mat33,
                )
                * self.model.particle_mass
            )
        else:
            mass_diag = wp.zeros(self.model.particle_count, dtype=wp.mat33)

            particle_density = wp.ones(self.model.particle_count, dtype=wp.float32)
            wp.launch(
                precompute_mass_matrix,
                dim=self.model.tet_count,
                inputs=[
                    self.model.particle_q,
                    self.model.tet_indices,
                    particle_density,
                ],
                outputs=[mass_diag],
            )

        wp.synchronize()
        print("precompute mass done")

        self.tet_stretch = wp.empty(self.model.tet_count, dtype=vec6)
        wp.launch(
            precompute_tet_stretch,
            dim=self.model.tet_count,
            inputs=[
                self.model.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
            ],
            outputs=[self.tet_stretch],
            device=self.model.device,
        )

        wp.synchronize()
        print("precompute tet stretch done")

        self.model.mass_matrix = ws.bsr_diag(
            diag=mass_diag,
            rows_of_blocks=self.model.particle_count,
            cols_of_blocks=self.model.particle_count,
            block_type=wp.mat33,
            device=self.model.device,
        )
        self.coo_row_idx = wp.zeros(self.model.tet_count * 4, dtype=wp.int32)
        self.coo_col_idx = wp.zeros(self.model.tet_count * 4, dtype=wp.int32)

        wp.launch(
            precompute_bsr_topology,
            dim=self.model.tet_count,
            inputs=[self.model.tet_indices],
            outputs=[self.coo_row_idx, self.coo_col_idx],
        )

        # Run a solver iteration to fill in work arrays
        dummy_state0 = self.model.state()
        dummy_state1 = self.model.state()
        x = wp.array(dummy_state0.particle_q, dtype=wp.vec3)
        x_tilde = wp.array(dummy_state1.particle_q, dtype=wp.vec3)
        s = wp.array(self.tet_stretch, dtype=vec6)
        lmbda = wp.zeros_like(s)
        with wp.ScopedTimer("Warm up iteration", active=True):
            self._solver_iteration(
                dummy_state0,
                dummy_state1,
                x,
                s,
                lmbda,
                x_tilde,
                0.01,
                reuse_topology=False,
            )
        # self._precompute_bsr_topologies()

    def _kinetic_gradient_dx(
        self, state: State, x: wp.array[wp.vec3], x_tilde: wp.array[wp.vec3], dt: float
    ) -> wp.array:
        return self.model.mass_matrix @ (x - x_tilde)

    def _elastic_gradient_ds(
        self, state: State, s: wp.array[vec6], dt: float
    ) -> wp.array:
        volume = self.tet_rest_volumes
        materials = self.model.tet_materials

        return dt * dt * self._material_model.gradient_ds(s, materials, volume)

    def _constraints(
        self, state: State, x: wp.array[wp.vec3], s: wp.array[vec6]
    ) -> wp.array:
        tets = self.model.tet_indices
        rest = self.model.tet_poses

        constraints = wp.empty(shape=(self.model.tet_count,), dtype=vec6)

        wp.launch(
            evaluate_constraints,
            dim=tets.shape[0],
            inputs=[x, s, tets, rest],
            outputs=[constraints],
        )

        return constraints

    def _kinetic_hessian_dx(self, _state: State, dt: float) -> wp.array:
        return self.model.mass_matrix

    def _constraint_gradient_ds_inverse(self, state: State, dt: float) -> ws.BsrMatrix:
        # G_s = dc/ds = -D where D = diag(1,1,1,2,2,2), so G_si = D^{-1} = diag(1,1,1,0.5,0.5,0.5)
        return ws.bsr_diag(
            wp.diag(vec6(1.0, 1.0, 1.0, 0.5, 0.5, 0.5)),
            self.model.tet_count,
            block_type=mat66,
        )

    def _constraint_gradient_dx(self, x: wp.array, dt: float) -> ws.BsrMatrix:

        values = wp.zeros(self.model.tet_count * 4, dtype=mat63)

        wp.launch(
            evaluate_constraint_gradient_dx,
            dim=self.model.tet_count,
            inputs=[x, self.model.tet_indices, self.model.tet_poses],
            outputs=[values],
        )

        G_x = ws.bsr_from_triplets(
            self.model.tet_count,
            self.model.particle_count,
            self.coo_row_idx,
            self.coo_col_idx,
            values,
            prune_numerical_zeros=False,
        )

        return G_x

    def _elastic_hessian_ds(
        self, state: State, s: wp.array[vec6], dt: float
    ) -> wp.array:

        return (
            dt
            * dt
            * self._material_model.hessian_ds(
                stretch=s,
                material=self.model.tet_materials,
                volume=self.tet_rest_volumes,
            )
        )

    def _gradient_blocks(
        self,
        state: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
    ) -> tuple[wp.array, wp.array, wp.array]:
        kinetic_gradient = self._kinetic_gradient_dx(state, x, x_tilde, dt)
        elastic_gradient = self._elastic_gradient_ds(state, s, dt)
        constraints = self._constraints(state, x, s)

        return kinetic_gradient, elastic_gradient, constraints

    def _hessian_blocks(
        self,
        state: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix]:
        kinetic_hessian = self._kinetic_hessian_dx(state, dt)
        elastic_hessian = self._elastic_hessian_ds(state, s, dt)
        constraint_gradient_ds_inverse = self._constraint_gradient_ds_inverse(state, dt)
        constraint_gradient_dx = self._constraint_gradient_dx(x, dt)

        return (
            kinetic_hessian,
            elastic_hessian,
            constraint_gradient_ds_inverse,
            constraint_gradient_dx,
        )

    def _energy(
        self,
        state: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
    ) -> float:
        """For testing purposes won't work in a Cuda Graph"""

        # 0.5 * ||x- x' ||_M^2 = 0.5 * || = 0.5 * ||v * dt||_M^2
        kinetic_energy = 0.5 * np.dot(
            (x - x_tilde).numpy().flatten(),
            ws.bsr_mv(self.model.mass_matrix, x - x_tilde).numpy().flatten(),
        )
        print(f"kinetic_energy: {kinetic_energy}")
        elastic_energy = (
            dt
            * dt
            * (
                self._material_model.energy(
                    s, self.model.tet_materials, self.tet_rest_volumes
                )
                .numpy()
                .sum()
            )
        )
        print(f"elastic_energy: {elastic_energy}")
        constraints = self._constraints(state, x, s).numpy().flatten()
        constraint_norm = np.linalg.norm(constraints)
        print(f"constraint_norm: {constraint_norm:.6e}")

        return kinetic_energy + elastic_energy

    def _line_search(self, dx, ds, lmbda) -> float:
        return 1.0

    def _solver_iteration(
        self,
        state_in: State,
        state_out: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
        reuse_topology: bool = True,
        timer: bool = False,
    ) -> tuple[wp.array[wp.vec3], wp.array[vec6], wp.array[vec6]]:
        (
            H_x,  # Kinetic Hessian
            H_s,  # Elastic Hessian
            G_si,  # Inverse constraint gradient with respect to stretch
            G_x,  # Constraint gradient with respect to position
        ) = self._hessian_blocks(state_in, x, s, x_tilde, dt)
        (
            g_x,  # Kinetic gradient
            g_s,  # Constraint gradient with respect to stretch
            constraint,  # Constraint values
        ) = self._gradient_blocks(state_in, x, s, x_tilde, dt)

        # K = G_x.transpose() @ G_si @ H_s @ G_si @ G_x
        # [K0, K1, K2, K3] = self.intermediate_bsr_matrices
        # K0 = ws.bsr_mm(G_x.transpose(), G_si, K0, masked=True)
        # K1 = ws.bsr_mm(K0, H_s, K1, masked=True)
        # K2 = ws.bsr_mm(K1, G_si, K2, masked=True)
        # K3 = ws.bsr_mm(K2, G_x, K3, masked=True)
        # with wp.ScopedTimer("Assemble system"):
        #     G_xT = G_x.transpose()
        #     K = ws.bsr_mm(
        #         G_xT,
        #         G_si,
        #         reuse_topology=reuse_topology,
        #         work_arrays=self._work_arrays,
        #     )
        #     K = ws.bsr_mm(
        #         K, H_s, reuse_topology=reuse_topology, work_arrays=self._work_arrays
        #     )
        #     K = ws.bsr_mm(
        #         K, G_si, reuse_topology=reuse_topology, work_arrays=self._work_arrays
        #     )
        #     self._lhs = ws.bsr_mm(
        #         K, G_x, reuse_topology=reuse_topology, work_arrays=self._work_arrays
        #     )

        #     self._lhs = ws.bsr_axpy(H_x, self._lhs)

        #     A = ws.bsr_mm(
        #         G_xT,
        #         G_si,
        #         reuse_topology=reuse_topology,
        #         work_arrays=self._work_arrays,
        #     )
        #     B = ws.bsr_mm(
        #         H_s,
        #         G_si,
        #         reuse_topology=reuse_topology,
        #         work_arrays=self._work_arrays,
        #     )
        #     self._rhs = A @ (f_s - B @ f_lmbda) - f_x
        with wp.ScopedTimer("Assemble system", active=timer):
            # wp.launch(
            #     invert_diagonal_values,
            #     dim=min(H_s.nrow, H_s.ncol),
            #     inputs=[H_s.scalar_values(), H_s.block_shape()],
            # plot_bsr(H_s)
            #
            # At some point I can make these just arrays since they are all diagonal
            G_xt = G_x.transpose()
            HsGsi = ws.bsr_mm(
                H_s,
                G_si,
                reuse_topology=reuse_topology,
                work_arrays=self._work_arrays["HsGsi"],
            )  # )
            H_lmbda = ws.bsr_mm(
                G_si,
                HsGsi,
                reuse_topology=reuse_topology,
                work_arrays=self._work_arrays["Hlmbda"],
            )
            # plot_bsr(H_lmbda)

            invert_diagonal_bsr(HsGsi)

            # g_lmbda = c + D Hs^{-1} g_s  (= ĝ_λ in Schur complement derivation)
            g_lmbda = constraint + HsGsi @ g_s

            GxtHlmbda = ws.bsr_mm(
                G_xt,
                H_lmbda,
                work_arrays=self._work_arrays["GxtHlmbda"],
                reuse_topology=reuse_topology,
            )

            GxtHlmbdaGx = ws.bsr_mm(
                GxtHlmbda,
                G_x,
                reuse_topology=reuse_topology,
                work_arrays=self._work_arrays["GxtHlmbdaGx"],
            )
            H = H_x + GxtHlmbdaGx
            # RHS: -(g_x + Gx^T Hλ ĝλ)
            g = -GxtHlmbda @ g_lmbda - g_x
            self._lhs = H
            self._rhs = g

        dx = x_tilde - x  # This should be a better initial guess
        with wp.ScopedTimer("Global solve", active=timer):
            #     cg(self._lhs, self._rhs, dx, check_every=0, use_cuda_graph=True, maxiter=10)
            final_iteration, residual, absolute_tolerance = cg(
                self._lhs,
                self._rhs,
                dx,
                check_every=10,
                use_cuda_graph=False,
                maxiter=100,
            )
            print(
                f"final_iteration: {final_iteration}, residual: {residual}, absolute_tolerance: {absolute_tolerance}"
            )

            print(f"dx: {dx}")
            # print(self._lhs)

        with wp.ScopedTimer("Local solve", active=timer):
            # Δλ = Hλ(ĝλ + Gx Δx)
            lmbda = H_lmbda @ (g_lmbda + G_x @ dx)
            # Δs = D^{-1}(c + Gx Δx)  — from KKT row 3: Gx Δx - D Δs = -c
            ds = G_si @ (constraint + G_x @ dx)

        alpha = self._line_search(dx, ds, lmbda)
        return x + alpha * dx, s + alpha * ds, lmbda

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        x = wp.array(state_in.particle_q, dtype=wp.vec3)
        s = wp.array(self.tet_stretch, dtype=vec6)
        lmbda = wp.zeros_like(s)
        x_tilde = x + state_in.particle_qd * dt

        for i in range(self.solver_iterations):
            energy = self._energy(state_in, x, s, x_tilde, dt)
            print(f"Iteration {i}, energy: {energy:.6e}")

            with wp.ScopedTimer("Iteration " + str(i)):
                x, s, lmbda = self._solver_iteration(
                    state_in, state_out, x, s, lmbda, x_tilde, dt, timer=True
                )

        state_out.particle_qd = (x - state_in.particle_q) / dt
        state_out.particle_q = x
        self.tet_stretch = s

        # raise RuntimeError("Completed one solver iteration this is not an error")
