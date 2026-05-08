# import newton
import warp as wp
import warp.sparse
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase
from warp.optim.linear import cg
from warp.sparse import BsrMatrix

from .kernels import evaluate_constraints, precompute_rest
from .material_model import MaterialModel
from .utils import vec6


class MFEMSolver(SolverBase):
    def __init__(
        self, model: Model, material_model: MaterialModel, subspace: BsrMatrix
    ):
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

        self.model.mass_matrix = warp.sparse.bsr_diag(
            diag=self.model.particle_masses,
            rows_of_blocks=1,
            cols_of_blocks=1,
            block_type=wp.float32,
            device=self.model.device,
        )

    def _kinetic_gradient_dx(self, state: State, dt: float) -> wp.array:
        particle_qd = state.particle_qd
        return self.model.mass_matrix @ (particle_qd / dt)

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

        return constraints

    def _kinetic_hessian_dx(self, state: State, dt: float) -> wp.array:
        return (1 / (dt * dt)) * self.model.mass_matrix

    def _constraint_gradient_ds_inverse(self, state: State) -> wp.array:
        raise -warp.sparse.bsr_identity(self.model.tet_count, block_type=wp.float32)

    def _constraint_gradient_dx(self, state: State) -> wp.array:
        raise NotImplementedError()

    def _elastic_hessian_ds(self, state: State) -> wp.array:
        return self._material_model.elastic_hessian_ds(
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
    ) -> tuple[BsrMatrix, BsrMatrix, BsrMatrix, BsrMatrix]:
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
        raise NotImplementedError()

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
        ds = -G_si @ (f_lmbda + G_x & dx)
        lmbda = -G_si @ (f_s + H_s @ ds)

        alpha = self._line_search(dx, ds, lmbda)
        state_out.particle_q = state_in.particle_q + alpha * dx
        state_out.stretch = state_in.stretch + alpha * ds
