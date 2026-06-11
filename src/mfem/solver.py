# import newton


from typing import override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warp as wp
import warp.sparse as ws
from newton import Contacts, Control, Model, State
from newton._src.solvers.kamino.config import dataclass
from newton.solvers import SolverBase
from warp.optim.linear import cg

from mfem.accumulate import TiledAccumulator
from mfem.boundary_condition import DirichletBoundaryCondition

from .energies.material_model import StretchMaterialModel
from .kernels import (
    constraint_objective_kernel,
    evaluate_constraint_gradient_dx,
    evaluate_constraints,
    kinetic_objective_kernel,
    precompute_bsr_topology,
    precompute_mass_matrix,
    precompute_rest_volumes,
    precompute_tet_stretch,
    project_position,
    test_deform_kernel,
)
from .line_search import backtracking_line_search
from .types import mat63, mat66, vec6
from .utils import linear_accumulate


class MFEMSolver(SolverBase):
    def __init__(
        self,
        model: Model,
        material_model: StretchMaterialModel,
        iterations: int,
        record_energy: bool = False,
        debug_logs: bool = False,
        fix_indices: np.ndarray | None = None,
        line_search: bool = True,
    ):
        super().__init__(model)

        self.solver_iterations = iterations
        self._material_model = material_model
        self._bsr_work_arrays = {
            "HsGsi": ws.bsr_mm_work_arrays(),
            "GsHsi": ws.bsr_mm_work_arrays(),
            "Hlmbda": ws.bsr_mm_work_arrays(),
            "GxtHlmbda": ws.bsr_mm_work_arrays(),
            "GxtHlmbdaGx": ws.bsr_mm_work_arrays(),
        }

        self._particle_energies = wp.empty(self.model.particle_count, dtype=wp.float32)
        self._tet_enegies = wp.empty(self.model.tet_count, dtype=wp.float32)
        self._s = wp.empty(self.model.tet_count, dtype=vec6)
        log_columns = ["Sim Time", "Iteration"]
        self._log = None

        if debug_logs:
            log_columns.extend(
                [
                    "g_s",
                    "g_x",
                    "constraint",
                    "g_lmbda",
                    "dx",
                    "ds",
                    "lmbda_new",
                ]
            )

        if record_energy:
            log_columns.extend(
                [
                    "Total Energy",
                    "Kinetic Energy",
                    "Elastic Energy",
                    "Constraint Energy",
                ]
            )

        self._record_energy = record_energy
        self._debug_logs = debug_logs
        self._log = pd.DataFrame(columns=log_columns)
        self._sim_time = 0.0
        self._do_line_search = line_search
        # self._B = subspace
        #

        if fix_indices is None:
            fix_indices = np.array([], dtype=np.int32)

        self._boundary_condition = DirichletBoundaryCondition(
            pin_positions=wp.array(
                self.model.particle_q.numpy()[fix_indices], dtype=wp.vec3
            ),
            constraint_indices=wp.array(fix_indices, dtype=wp.int32),
            constraint_stiffness=wp.full(
                shape=(fix_indices.shape[0],), value=1.0, dtype=wp.float32
            ),
            n_particles=self.model.particle_count,
        )

        self._accumulator = TiledAccumulator(
            max_length=max(3 * self.model.particle_count, 6 * self.model.tet_count),
            device=self.model.device,
            scalar_type=wp.float32,
        )

        self._initial_precompute()

    def _test_deform(self):

        wp.launch(
            test_deform_kernel,
            dim=self.model.particle_count,
            inputs=[self.model.particle_q],
            device=self.model.device,
        )

    def _initial_precompute(self):

        self._tet_rest_volumes = wp.empty(self.model.tet_count)
        wp.launch(
            precompute_rest_volumes,
            dim=self.model.tet_count,
            inputs=[self.model.tet_poses],
            outputs=[self._tet_rest_volumes],
            device=self.model.device,
        )

        wp.launch(
            precompute_tet_stretch,
            dim=self.model.tet_count,
            inputs=[
                self.model.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
            ],
            outputs=[self._s],
            device=self.model.device,
        )
        # self._test_deform()

        mass_diag = None
        if self.model.particle_mass is not None:
            print("Got particle mass, using it to compute mass matrix diagonal")
            mass_diag = (
                wp.full(
                    shape=self.model.particle_count,
                    value=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=wp.mat33,
                )
                * self.model.particle_mass
            )

        elif self.model.particle_inv_mass is not None:
            print("Got inverse mass, computing mass diagonal")
            mass_diag = wp.full(
                shape=self.model.particle_count,
                value=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                dtype=wp.mat33,
            ) * (1.0 / self.model.particle_inv_mass)
        else:
            mass_diag = wp.zeros(self.model.particle_count, dtype=wp.mat33)

            particle_density = wp.ones(self.model.particle_count, dtype=wp.float32)
            print(
                "Assuming uniform density of 1.0 since no mass or inverse mass provided"
            )
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
        x = wp.array(dummy_state0.particle_q, dtype=wp.vec3)
        x_tilde = wp.array(dummy_state0.particle_q, dtype=wp.vec3)
        s = wp.array(self._s, dtype=vec6)
        lmbda = wp.zeros_like(s)
        with wp.ScopedTimer("Warm up iteration", active=True):
            self._solver_iteration(
                dummy_state0,
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
        volume = self._tet_rest_volumes
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

    # TODO: fix block dimensions
    def _constraint_gradient_ds_inverse(
        self, state: State, dt: float
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix]:
        return (
            ws.bsr_diag(
                wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)),
                self.model.tet_count,
                block_type=mat66,
            ),
            ws.bsr_diag(
                wp.diag(vec6(1.0, 1.0, 1.0, 0.5, 0.5, 0.5)),
                self.model.tet_count,
                block_type=mat66,
            ),
        )

    def _constraint_gradient_dx(self, state: State, dt: float) -> ws.BsrMatrix:

        values = wp.zeros(self.model.tet_count * 4, dtype=mat63)

        wp.launch(
            evaluate_constraint_gradient_dx,
            dim=self.model.tet_count,
            inputs=[state.particle_q, self.model.tet_indices, self.model.tet_poses],
            outputs=[values],
        )

        G_x = -ws.bsr_from_triplets(
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
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix]:

        H_s, H_si = self._material_model.hessian_ds(
            stretch=s,
            material=self.model.tet_materials,
            volume=self._tet_rest_volumes,
        )

        return (dt * dt * H_s, H_si / (dt * dt))

    def _gradient_blocks(
        self,
        state: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
    ) -> tuple[wp.array, wp.array, wp.array]:
        kinetic_gradient = self._kinetic_gradient_dx(state, x, x_tilde, dt)
        boundary_condition_gradient = self._boundary_condition.gradient(x)
        elastic_gradient = self._elastic_gradient_ds(state, s, dt)
        constraints = self._constraints(state, x, s)

        return (
            kinetic_gradient + boundary_condition_gradient,
            elastic_gradient,
            constraints,
        )

    def _hessian_blocks(
        self,
        state: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix]:
        kinetic_hessian = self._kinetic_hessian_dx(state, dt)
        boundary_condition_hessian = self._boundary_condition.hessian()
        elastic_hessian, inv_elastic_hessian = self._elastic_hessian_ds(state, s, dt)
        constraint_gradient_ds, constraint_gradient_ds_inverse = (
            self._constraint_gradient_ds_inverse(state, dt)
        )
        constraint_gradient_dx = self._constraint_gradient_dx(state, dt)

        return (
            kinetic_hessian + boundary_condition_hessian,
            elastic_hessian,
            inv_elastic_hessian,
            constraint_gradient_ds,
            constraint_gradient_ds_inverse,
            constraint_gradient_dx,
        )

    def _objective(
        self,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
        objective: wp.array[wp.float32],
    ) -> wp.array:
        """For testing purposes won't work in a Cuda Graph"""
        nodal_kinetic_objective = self._particle_energies
        wp.launch(
            kinetic_objective_kernel,
            dim=self.model.particle_count,
            inputs=[x, x_tilde, self.model.particle_mass],
            outputs=[nodal_kinetic_objective],
        )

        self._accumulator.compute_sum(nodal_kinetic_objective)
        wp.copy(objective, self._accumulator.result())

        # 0.5 * ||x- x' ||_M^2 = 0.5 * || = 0.5 * ||v * dt||_M^2
        tet_energy = self._tet_enegies
        self._material_model.energy(
            s,
            self.model.tet_materials,
            self._tet_rest_volumes,
            tet_energy,
        )

        wp.launch(
            constraint_objective_kernel,
            dim=self.model.tet_count,
            inputs=[x, s, self.model.tet_indices, self.model.tet_poses, lmbda],
            outputs=[tet_energy],
        )

        self._accumulator.compute_sum(tet_energy)
        linear_accumulate(objective, self._accumulator.result(), beta=(dt * dt))

        self._boundary_condition.energy(x, self._accumulator)
        linear_accumulate(objective, self._accumulator.result(), beta=(dt * dt))

        return objective

    # I should probably refactor all of this
    def _line_search(
        self,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        G_xt: ws.BsrMatrix,
        G_st: ws.BsrMatrix,
        g_x: wp.array[wp.vec3],
        g_s: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dx: wp.array[wp.vec3],
        ds: wp.array[vec6],
        dt: wp.float32,
        max_iterations=10,
        alpha0=3.0,
        tau=0.333,
        c=0.7,
        threshold=1e-4,
    ):

        def objective_fn(dof_arrays: tuple[wp.array], objective: wp.array[wp.float32]):
            self._objective(dof_arrays[0], dof_arrays[1], lmbda, x_tilde, dt, objective)

        dof_arrays = (x, s)
        search_direction = (dx, ds)
        gradient = (
            g_x + G_xt @ lmbda,
            g_s + G_st @ lmbda,
        )
        return backtracking_line_search(
            objective_fn,
            dof_arrays,
            search_direction,
            gradient,
            self._accumulator,
            max_iterations,
            alpha0,
            tau,
            c,
            threshold,
        )

    def _solver_iteration(
        self,
        state_in: State,
        x: wp.array[wp.vec3],
        s: wp.array[vec6],
        lmbda: wp.array[vec6],
        x_tilde: wp.array[wp.vec3],
        dt: float,
        reuse_topology: bool = True,
        timer: bool = False,
    ) -> tuple[wp.array[wp.float32]] | None:
        # log = {}
        (
            H_x,  # Kinetic Hessian
            H_s,  # Elastic Hessian
            H_si,
            G_s,
            G_si,  # Inverse constraint gradient with respect to stretch
            G_x,  # Constraint gradient with respect to position
        ) = self._hessian_blocks(state_in, x, s, x_tilde, dt)
        (
            g_x,  # Kinetic gradient
            g_s,  # Elastic gradient with respect to stretch
            g_lmbda,  # Constraint values (i.e. gradient with respecto to lagrange multipliers)
        ) = self._gradient_blocks(state_in, x, s, x_tilde, dt)

        # if self._debug_logs:
        #     log.update({"g_s": g_s.numpy().flatten(), "g_x": g_x.numpy().flatten(), "constraint": constraint.numpy().flatten()})
        #     self._rhs = A @ (f_s - B @ f_lmbda) - f_x
        with wp.ScopedTimer("Assemble system", active=timer):
            G_xt = G_x.transpose()
            HsGsi = ws.bsr_mm(
                H_s,
                G_si,
                reuse_topology=reuse_topology,
                work_arrays=self._bsr_work_arrays["HsGsi"],
            )  # )
            H_lmbda = ws.bsr_mm(
                G_si,
                HsGsi,
                reuse_topology=reuse_topology,
                work_arrays=self._bsr_work_arrays["Hlmbda"],
            )
            # plot_bsr(H_lmbda)

            GsHsi = ws.bsr_mm(
                G_s,
                H_si,
                reuse_topology=reuse_topology,
                work_arrays=self._bsr_work_arrays["GsHsi"],
            )

            g_lmbda = g_lmbda + GsHsi @ g_s
            # if self._debug_logs:
            #     log.update({"g_lmbda": g_lmbda.numpy().flatten()})

            GxtHlmbda = ws.bsr_mm(
                G_xt,
                H_lmbda,
                work_arrays=self._bsr_work_arrays["GxtHlmbda"],
                reuse_topology=reuse_topology,
            )
            # plot_bsr(GxtHlmbda)

            GxtHlmbdaGx = ws.bsr_mm(
                GxtHlmbda,
                G_x,
                reuse_topology=reuse_topology,
                work_arrays=self._bsr_work_arrays["GxtHlmbdaGx"],
            )
            H = H_x + GxtHlmbdaGx
            g = GxtHlmbda @ g_lmbda - g_x

            self._lhs = H
            self._rhs = g

        dx = x_tilde - x  # This should be a better initial guess
        with wp.ScopedTimer("Global solve", active=timer):
            #     cg(self._lhs, self._rhs, dx, check_every=0, use_cuda_graph=True, maxiter=10)
            # final_iteration, residual, absolute_tolerance =
            cg(
                self._lhs,
                self._rhs,
                dx,
                check_every=0,
                use_cuda_graph=True,
                maxiter=100,
            )

            # print(final_iteration, residual, absolute_tolerance)
            # print(self._lhs)

        with wp.ScopedTimer("Local solve", active=timer):
            lmbda = -H_lmbda @ (g_lmbda - G_x @ dx)
            ds = -H_si @ (G_s @ lmbda + g_s)

        # if self._debug_logs:
        #     log.update(
        #         {
        #             "dx": dx.numpy().flatten(),
        #             "ds": ds.numpy().flatten(),
        #             "lmbda_new": lmbda.numpy().flatten(),
        #         }
        #     )

        # self.apply_updates(x, s, lmbda, dx, ds)
        if self._do_line_search:
            return self._line_search(
                x,
                s,
                lmbda,
                G_xt,
                G_s,
                g_x,
                g_s,
                x_tilde,
                dx,
                ds,
                dt,
                max_iterations=1000,
            )
        else:
            x += dx
            s += ds

    # def apply_updates(self, x, s, lmbda, dx, ds):
    #     alpha = self._line_search(dx, ds, lmbda)

    #     wp.launch(
    #         update_position,
    #         dim=self.model.particle_count,
    #         inputs=[x, dx, alpha],
    #     )

    #     wp.launch(
    #         update_stretch,
    #         dim=self.model.tet_count,
    #         inputs=[s, ds, alpha],
    #     )

    def get_log(self) -> pd.DataFrame | None:
        return self._log

    def write_log(self, path: str) -> None:
        self._log.to_parquet(path)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        x = state_out.particle_q
        wp.copy(x, state_in.particle_q)
        s = self._s
        lmbda = wp.zeros_like(s)

        x_tilde = wp.empty_like(x)
        wp.launch(
            project_position,
            dim=self.model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.model.gravity,
                state_in.particle_f,
                x_tilde,
                dt,
            ],
        )

        # wp.copy(x, x_tilde)
        # x_tilde = (
        #     x
        #     + state_in.particle_qd * dt
        #     + 0.5 * state_in.particle_f * self.model.particle_inv_mass * dt * dt
        # )
        #
        energy_log = []
        for i in range(self.solver_iterations):
            # log_row = {
            #     "Sim Time": self._sim_time,
            #     "Iteration": i,
            # }

            # if self._record_energy:
            #     kinetic_energy, elastic_energy, constraint_energy = self._energy(
            #         state_in, x, s, lmbda, x_tilde, dt
            #     )

            #     log_row.update(
            #         {
            #             "Kinetic Energy": kinetic_energy,
            #             "Elastic Energy": elastic_energy,
            #             "Constraint Energy": constraint_energy,
            #             "Total Energy": kinetic_energy
            #             + elastic_energy
            #             + constraint_energy,
            #         }
            #     )

            with wp.ScopedTimer("Iteration " + str(i), active=True):
                # x, s, lmbda =
                objective = self._solver_iteration(
                    state_in, x, s, lmbda, x_tilde, dt, timer=False
                )
                # energy_log.append(objective.numpy()[1])

            # log_row.update(iteration_log)
            # self._log.loc[len(self._log)] = log_row

        # plt.plot(energy_log)
        # plt.xlabel("Iteration")
        # plt.ylabel("Energy")
        # plt.show()

        wp.copy(
            state_out.particle_qd,
            (x - state_in.particle_q) / dt
            + state_in.particle_f * self.model.particle_inv_mass * dt,
        )
        # print("start position:", state_in.particle_q.numpy()[7])
        # print("end position:", state_out.particle_q.numpy()[7])
        # wp.copy(state_out.particle_q, x)

        # self._sim_time += dt
        # raise RuntimeError("Completed one solver iteration this is not an error")
