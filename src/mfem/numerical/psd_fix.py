import warp as wp

from ..types import mat66
from .gmw81 import gmw81_3, gmw81_6


@wp.struct
class PSDFix3:
    mat: wp.mat33  # PSD-fixed matrix  A + E
    inv: wp.mat33  # its inverse (A + E)^{-1}


@wp.struct
class PSDFix6:
    mat: mat66
    inv: mat66


@wp.func
def psd_fix_3(A: wp.mat33) -> PSDFix3:
    return _psd_fix_3(gmw81_3(A))


@wp.func
def psd_fix_6(A: mat66) -> PSDFix6:
    return _psd_fix_6(gmw81_6(A))


@wp.func
def _psd_fix_3(L: wp.mat33) -> PSDFix3:
    # --- A_fixed = L @ L^T ---
    # A_fixed[i,j] = sum_{k=0}^{j} L[i,k]*L[j,k]  for j <= i (then mirror)
    A_fixed = wp.mat33(0.0)
    for i in range(3):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j + 1):
                s += L[i, k] * L[j, k]
            A_fixed[i, j] = s
            A_fixed[j, i] = s

    # --- Z = L^{-1} in-place via forward substitution ---
    # Column k of Z: Z[k,k] = 1/L[k,k]; Z[i,k] = -Σ_{j=k}^{i-1} L[i,j]*Z[j,k] / L[i,i]
    # Reads from L[i,j] (j > k, original) and L[j,k] (Z[j,k], already computed).
    # L[i,i] is original at the time of division since diagonal k < i hasn't been processed.
    for k in range(3):
        L[k, k] = float(1.0) / L[k, k]
        for i in range(k + 1, 3):
            s = float(0.0)
            for j in range(k, i):
                s += L[i, j] * L[j, k]
            L[i, k] = -s / L[i, i]

    # --- A_inv = Z^T @ Z ---
    # A_inv[i,j] = sum_{k=i}^{n-1} Z[k,i]*Z[k,j]  for j <= i (then mirror)
    A_inv = wp.mat33(0.0)
    for i in range(3):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(i, 3):
                s += L[k, i] * L[k, j]
            A_inv[i, j] = s
            A_inv[j, i] = s

    result = PSDFix3()
    result.mat = A_fixed
    result.inv = A_inv
    return result


@wp.func
def _psd_fix_6(L: mat66) -> PSDFix6:
    # --- A_fixed = L @ L^T ---
    A_fixed = mat66(0.0)
    for i in range(6):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j + 1):
                s += L[i, k] * L[j, k]
            A_fixed[i, j] = s
            A_fixed[j, i] = s

    # --- Z = L^{-1} in-place via forward substitution ---
    for k in range(6):
        L[k, k] = float(1.0) / L[k, k]
        for i in range(k + 1, 6):
            s = float(0.0)
            for j in range(k, i):
                s += L[i, j] * L[j, k]
            L[i, k] = -s / L[i, i]

    # --- A_inv = Z^T @ Z ---
    A_inv = mat66(0.0)
    for i in range(6):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(i, 6):
                s += L[k, i] * L[k, j]
            A_inv[i, j] = s
            A_inv[j, i] = s

    result = PSDFix6()
    result.mat = A_fixed
    result.inv = A_inv
    return result
