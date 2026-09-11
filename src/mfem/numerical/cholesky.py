from typing import Any

import warp as wp

from ..types import mat66

# Float = typing.TypeVar("Float", wp.float16, wp.float32, wp.float64)
# Dim = typing.TypeVar("Dim", bound=int)


# class SquareMatrix[Dim, Float](
#     wp.types.matrix(shape=(Generic[Dim], Generic[Dim]), dtype=Generic[Float])
# ):
#     """A square matrix type with generic dimension and data type."""


@wp.func
def cholesky6(A: mat66) -> mat66:
    return cholesky(A, 6)


@wp.func
def cholesky3(A: wp.mat33) -> wp.mat33:
    return cholesky(A, 3)


@wp.func
def cholesky(A: Any, dim: int) -> Any:
    # Cholesky-Banachiewicz: A = L @ L.T, computed in-place (lower triangle).
    # Assumes A is symmetric positive definite.
    for j in range(dim):
        # Diagonal: L[j,j] = sqrt(A[j,j] - sum_k L[j,k]^2)
        s = A[j, j]
        for k in range(j):
            s = s - A[j, k] * A[j, k]
        A[j, j] = wp.sqrt(s)

        inv_ljj = 1.0 / A[j, j]

        # Sub-diagonal column j: L[i,j] = (A[i,j] - sum_k L[i,k]*L[j,k]) / L[j,j]
        for i in range(j + 1, dim):
            s = A[i, j]
            for k in range(j):
                s = s - A[i, k] * A[j, k]
            A[i, j] = s * inv_ljj

        # Zero upper triangle so the returned matrix is strictly lower triangular
        for i in range(j):
            A[i, j] = 0.0

    return A
