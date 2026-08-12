from typing import Optional

import numpy as np
import warp as wp
import warp.sparse as ws

from mfem.accumulate import TiledAccumulator


class DirichletBoundaryCondition:
    def __init__(
        self,
        pin_positions: wp.array[wp.vec3],
        constraint_indices: wp.array[wp.int32],
        constraint_stiffness: wp.array[wp.float32],
        n_particles: int,
    ):
        S: ws.BsrMatrix = ws.bsr_from_triplets(
            rows_of_blocks=n_particles,
            cols_of_blocks=constraint_indices.shape[0],
            rows=constraint_indices,
            columns=wp.array(np.arange(constraint_indices.shape[0]), dtype=wp.int32),
            values=wp.full(
                constraint_indices.shape[0], np.eye(3, dtype=np.float32), dtype=wp.mat33
            ),
        )

        Gamma = ws.bsr_diag(
            constraint_stiffness
            * wp.full(
                constraint_indices.shape[0], np.eye(3, dtype=np.float32), dtype=wp.mat33
            ),
            rows_of_blocks=constraint_indices.shape[0],
            cols_of_blocks=constraint_indices.shape[0],
        )

        self._work_arrays = {"Qx_b": wp.empty(n_particles, dtype=wp.vec3)}

        SGamma = S @ Gamma

        b = -SGamma @ pin_positions
        Q = SGamma @ S.transpose()

        self._SGamma = SGamma
        self._pin_positions = pin_positions
        self._Q = Q
        self._b = b
        self._S = S
        self._Gamma = Gamma

        self._c = wp.array(
            [
                0.5
                * np.dot(
                    pin_positions.numpy().flatten(),
                    (Gamma @ pin_positions).numpy().flatten(),
                )
            ],
            dtype=wp.float32,
        )

    def energy(self, x: wp.array[wp.vec3], accumulator: TiledAccumulator):
        wp.copy(self._work_arrays["Qx_b"], self._b)
        y = ws.bsr_mv(self._Q, x, self._work_arrays["Qx_b"], alpha=0.5, beta=1.0)
        # print(self._work_arrays["Qx_b"])
        # y = 0.5 * self._Q @ x + self._b

        accumulator.compute_dot(x, y)

        res = accumulator.result()
        res += self._c

    def gradient(
        self, x: wp.array[wp.vec3], grad: Optional[wp.array[wp.vec3]] = None
    ) -> Optional[wp.array[wp.vec3]]:
        if grad is not None:
            ws.bsr_mv(self._Q, x, grad)
            grad += self._b
        else:
            return self._Q @ x + self._b

    def hessian(self):
        return self._Q

    def update_b(self, accumulator: TiledAccumulator | None = None):
        self._b = ws.bsr_mv(-self._SGamma, self._pin_positions, self._b)

        if accumulator is not None:
            accumulator.compute_dot(
                self._pin_positions, self._Gamma @ self._pin_positions
            )
            wp.copy(self._c, 0.5 * accumulator.result())
