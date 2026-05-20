# import newton
import warp as wp
import warp.sparse
import warp.sparse as ws
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase
from warp.optim.linear import cg

from .kernels import (
    evaluate_constraints,
    evalutate_constraint_gradient_dx,
    precompute_mass_matrix,
    precompute_rest,
)
from .material_model import StretchMaterialModel
from .utils import mat63, vec6


class MFEMSolver(SolverBase):
    def __init__(self, model: Model, material_model: StretchMaterialModel):
        super().__init__(model)

        self._material_model = material_model
        # self._B = subspace
        self._initial_precompute()

    def _initial_precompute(self):

        self.model.volume = wp.empty(self.model.tet_count)

        wp.launch(
            precompute_rest,
            dim=self.model.tet_count,
            inputs=[self.model.particle_q, self.model.tet_indices],
            outputs=[self.model.volume, self.model.tet_rest_volumes],
            device=self.model.device,
        )

        mass_diag = wp.empty(self.model.particle_count, dtype=wp.mat33)
        wp.launch(
            precompute_mass_matrix,
            dim=self.model.tet_count,
            inputs=[
                self.model.particle_q,
                self.model.tet_indices,
                self.model.particle_density,
            ],
            outputs=[mass_diag],
        )

        self.model.mass_matrix = warp.sparse.bsr_diag(
            diag=mass_diag,
            rows_of_blocks=self.model.particle_count,
            cols_of_blocks=self.model.particle_count,
            block_type=wp.mat33,
            device=self.model.device,
        )

    def _kinetic_gradient_dx(self, state: State, dt: float) -> wp.array:
        particle_qd = state.particle_qd
        return self.model.mass_matrix @ (particle_qd.reshape(-1, 1) / dt)

    def _elastic_gradient_ds(self, state: State) -> wp.array:
        mu = self.model.particle_mu
        lmbda = self.model.particle_lambda
        stretch = state.tet_stretch
        volume = self.model.tet_rest_volumes

        return self._material_model.gradient_ds(stretch, mu, lmbda, volume)

    def _constraints(self, state: State) -> wp.array:
        particle_q = state.particle_q
        stretch = state.tet_stretch
        tets = self.model.tet_indices
        rest = self.model.tet_poses

        constraints = wp.empty(shape=self.model.tet_count, dtype=vec6)

        wp.launch(
            evaluate_constraints,
            dim=tets.shape[0],
            inputs=[particle_q, stretch, tets, rest],
            outputs=[constraints],
        )

        return constraints.flatten()

    def _kinetic_hessian_dx(self, state: State, dt: float) -> wp.array:
        return (1 / (dt * dt)) * self.model.mass_matrix

    def _constraint_gradient_ds_inverse(self, state: State) -> ws.BsrMatrix:
        return -ws.bsr_identity(self.model.tet_count, block_type=wp.float32)

    def _constraint_gradient_dx(self, state: State) -> ws.BsrMatrix:
        row_idx = wp.zeros(self.model.tet_count * 4, dtype=wp.uint32)
        col_idx = wp.zeros(self.model.tet_count * 4, dtype=wp.uint32)
        values = wp.zeros(self.model.tet_count * 4, dtype=mat63)

        wp.launch(
            evalutate_constraint_gradient_dx,
            dim=self.model.tet_count,
            inputs=[state.particle_q, self.model.tet_indices, self.model.tet_poses],
            outputs=[row_idx, col_idx, values],
        )

        G_x = ws.bsr_from_triplets(
            self.model.tet_count,
            self.model.particle_count,
            row_idx,
            col_idx,
            values,
            prune_numerical_zeros=False,
        )

        return G_x

    def _elastic_hessian_ds(self, state: State) -> wp.array:
        return self._material_model.hessian_ds(
            state.stretch, state.mu, state.lmbda, state.volume
        )

    def _gradient_blocks(
        self, state: State, dt: float
    ) -> tuple[wp.array, wp.array, wp.array]:
        kinetic_gradient = self._kinetic_gradient_dx(state, dt)
        elastic_gradient = self._elastic_gradient_ds(state)
        constraints = self._constraints(state)

        return kinetic_gradient, elastic_gradient, constraints

    def _hessian_blocks(
        self, state: State, dt: float
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix, ws.BsrMatrix]:
        kinetic_hessian = self._kinetic_hessian_dx(state, dt)
        elastic_hessian = self._elastic_hessian_ds(state)
        constraint_gradient_ds_inverse = self._constraint_gradient_ds_inverse(state)
        constraint_gradient_dx = self._constraint_gradient_dx(state)

        return (
            kinetic_hessian,
            elastic_hessian,
            constraint_gradient_ds_inverse,
            constraint_gradient_dx,
        )

    def _line_search(self, dx, ds, lmbda) -> float:
        return 1.0

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        (
            H_x,  # Kinetic Hessian
            H_s,  # Elastic Hessian
            G_si,  # Inverse constraint gradient with respect to stretch
            G_x,  # Constraint gradient with respect to position
        ) = self._hessian_blocks(state_in, dt)
        (
            f_x,  # Kinetic gradient
            f_s,  # Constraint gradient with respect to stretch
            f_lmbda,  # Constraint values
        ) = self._gradient_blocks(state_in, dt)

        K = G_x @ G_si @ H_x @ G_si @ G_x.transpose()
        lhs = H_x + K
        rhs = (
            G_x.transpose()
            @ G_si
            @ (f_s - H_s @ G_si @ (f_lmbda - H_x @ G_si @ f_lmbda))
            - f_x
        )
        dx = wp.empty_like(state_in.particle_q)
        cg.solve(lhs, rhs, dx)
        ds = -G_si @ (f_lmbda + G_x @ dx)
        lmbda = -G_si @ (f_s + H_s @ ds)

        alpha = self._line_search(dx, ds, lmbda)
        state_out.particle_q = state_in.particle_q + alpha * dx
        state_out.stretch = state_in.stretch + alpha * ds
