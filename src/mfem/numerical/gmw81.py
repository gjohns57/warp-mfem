from typing import Any

import warp as wp

from ..types import mat66


@wp.func
def gmw81_6(A: mat66) -> mat66:
    return gmw81(A, 6)


@wp.func
def gmw81_3(A: wp.mat33) -> wp.mat33:
    return gmw81(A, 3)


@wp.func
def gmw81(A: Any, dim: int) -> Any:
    # Modified Cholesky (Gill-Murray-Wright 1981, Nocedal & Wright Algorithm 3.4).
    # Returns L such that A + E = L @ L.T, where E >= 0 is an implicit diagonal
    # correction that makes the system positive definite. A must be symmetric.
    # Upper triangle is read as the original values; lower triangle and diagonal
    # are overwritten with the factorization.

    eps = float(1.19e-7)  # float32 machine epsilon
    delta = wp.sqrt(eps)  # pivot floor (~3.4e-4 for float32)

    # γ = max|a_ii|, ξ = max_{i≠j}|a_ij|
    gamma = float(0.0)
    xi = float(0.0)
    for i in range(dim):
        v = wp.abs(A[i, i])
        if v > gamma:
            gamma = v
        for j in range(i + 1, dim):
            v = wp.abs(A[i, j])
            if v > xi:
                xi = v

    # β² = max(γ, ξ/√(n²−1), ε)
    beta2 = wp.max(gamma, wp.max(xi / wp.sqrt(float(dim * dim - 1)), eps))

    # Column-by-column factorization.
    # Invariant entering iteration j:
    #   A[k,k]  = sqrt(d_k)                  for k < j
    #   A[i,k]  = c_ik / d_k  (l_ik)         for i > k, k < j
    #   A[j,j]  = a_jj − Σ_{k<j} c_jk²/d_k  (c_jj, ready to use)
    #   A[j,i]  = original a_ij              for i > j  (upper triangle, never written)
    for j in range(dim):
        # Compute c_ij = a_ij − Σ_{k<j} d_k · l_ik · l_jk  for i > j
        for i in range(j + 1, dim):
            c_ij = A[j, i]  # read original a_ij from untouched upper triangle
            for k in range(j):
                c_ij -= A[k, k] * A[k, k] * A[i, k] * A[j, k]
            A[i, j] = c_ij

        # θ_j = max_{i>j} |c_ij|
        theta_j = float(0.0)
        for i in range(j + 1, dim):
            v = wp.abs(A[i, j])
            if v > theta_j:
                theta_j = v

        # d_j = max(δ, |c_jj|, θ_j²/β²)  — the corrected pivot
        c_jj = A[j, j]
        d_j = wp.max(delta, wp.max(wp.abs(c_jj), theta_j * theta_j / beta2))

        A[j, j] = wp.sqrt(d_j)

        # l_ij = c_ij / d_j; update c_ii for future iterations
        for i in range(j + 1, dim):
            c_ij = A[i, j]
            A[i, j] = c_ij / d_j
            A[i, i] -= c_ij * c_ij / d_j

    # Absorb sqrt(d_j) into each column so L_tilde[i,j] = l_ij·√d_j,
    # giving A + E = L_tilde @ L_tilde.T
    for j in range(dim):
        sqrt_dj = A[j, j]
        for i in range(j + 1, dim):
            A[i, j] = A[i, j] * sqrt_dj
        for i in range(j):
            A[i, j] = float(0.0)

    return A
