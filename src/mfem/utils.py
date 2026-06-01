from tkinter.constants import N

import numpy as np
import warp as wp
import warp.sparse as ws

from .types import mat69, mat99, mat912, vec6, vec9

"""
Utilities for MFEM elastic energy functions:
I found this awesome guide: https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf
"""


# class vec6(vector(length=6, dtype=wp.float32)):
#     """Symmetric 3x3 matrix stored as a 6-element vector."""

# vec6 = wp.vector(length=6, dtype=wp.float32)


@wp.func
def flatten(mat: wp.mat33) -> vec9:
    c0 = mat[:, 0]
    c1 = mat[:, 1]
    c2 = mat[:, 2]

    return vec9(c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2])


@wp.func
def unflatten(vec: vec9) -> wp.mat33:
    return wp.matrix_from_cols(vec[0:3], vec[3:6], vec[6:9])


@wp.func
def unflatten_t(vec: vec9) -> wp.mat33:
    return wp.matrix_from_rows(vec[0:3], vec[3:6], vec[6:9])


@wp.func
def row_flatten(mat: wp.mat33) -> vec9:
    r0 = mat.get_row(0)
    c1 = mat.get_row(1)
    c2 = mat.get_row(2)

    return vec9(r0[0], r0[1], r0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2])


@wp.func
def sym_mat33_to_vec6(mat: wp.mat33) -> vec6:
    return vec6(mat[0, 0], mat[1, 1], mat[2, 2], mat[0, 1], mat[0, 2], mat[1, 2])


@wp.func
def frob2_sym_vec6(vec: vec6) -> wp.float32:
    return (
        vec[0] * vec[0]
        + vec[1] * vec[1]
        + vec[2] * vec[2]
        + 2.0 * (vec[3] * vec[3] + vec[4] * vec[4] + vec[5] * vec[5])
    )


@wp.func
def frob_sym_vec6(vec: vec6) -> wp.float32:
    return wp.sqrt(frob2_sym_vec6(vec))


@wp.func
def tr_sym_vec6(vec: vec6) -> wp.float32:
    return vec[0] + vec[1] + vec[2]


@wp.func
def deformation_gradient(
    position: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    inv_rest_matrix: wp.array[wp.mat33],
    tid: int,
) -> wp.mat33:
    x0 = position[indices[tid, 0]]
    x1 = position[indices[tid, 1]]
    x2 = position[indices[tid, 2]]
    x3 = position[indices[tid, 3]]

    deformed_edge_matrix = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    rest_edge_matrix = inv_rest_matrix[tid]
    return deformed_edge_matrix * rest_edge_matrix


@wp.func
def deformation_gradient_dF_dx(DmInv: wp.mat33) -> mat912:
    m = DmInv[0, 0]
    n = DmInv[0, 1]
    o = DmInv[0, 2]
    p = DmInv[1, 0]
    q = DmInv[1, 1]
    r = DmInv[1, 2]
    s = DmInv[2, 0]
    t = DmInv[2, 1]
    u = DmInv[2, 2]

    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u

    # Row-major: row i*3+j corresponds to F[i,j]; nonzero cols are 3*k+i for vertices k=0..3
    dFdx = mat912()
    dFdx[0, 0] = t1
    dFdx[0, 3] = m
    dFdx[0, 6] = p
    dFdx[0, 9] = s
    dFdx[1, 0] = t2
    dFdx[1, 3] = n
    dFdx[1, 6] = q
    dFdx[1, 9] = t
    dFdx[2, 0] = t3
    dFdx[2, 3] = o
    dFdx[2, 6] = r
    dFdx[2, 9] = u
    dFdx[3, 1] = t1
    dFdx[3, 4] = m
    dFdx[3, 7] = p
    dFdx[3, 10] = s
    dFdx[4, 1] = t2
    dFdx[4, 4] = n
    dFdx[4, 7] = q
    dFdx[4, 10] = t
    dFdx[5, 1] = t3
    dFdx[5, 4] = o
    dFdx[5, 7] = r
    dFdx[5, 10] = u
    dFdx[6, 2] = t1
    dFdx[6, 5] = m
    dFdx[6, 8] = p
    dFdx[6, 11] = s
    dFdx[7, 2] = t2
    dFdx[7, 5] = n
    dFdx[7, 8] = q
    dFdx[7, 11] = t
    dFdx[8, 2] = t3
    dFdx[8, 5] = o
    dFdx[8, 8] = r
    dFdx[8, 11] = u
    return dFdx


@wp.func
def rotational_svd(A: wp.mat33):
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, sigma, V)

    det_U = wp.determinant(U)
    det_V = wp.determinant(V)

    flipped = det_U * det_V
    sigma[2] *= flipped
    if det_U < 0 and det_V >= 0:
        U[0, 2] *= flipped
        U[1, 2] *= flipped
        U[2, 2] *= flipped
    if det_V < 0 and det_U >= 0:
        V[0, 2] *= flipped
        V[1, 2] *= flipped
        V[2, 2] *= flipped

    return U, sigma, V


@wp.func
def stretch_component(A: wp.mat33):
    U, sigma, V = rotational_svd(A)
    return V * wp.diag(sigma) * wp.transpose(V)


# @wp.func
# def stretch_gradient(F: wp.mat33) -> mat99:
#     dR_dF, U, sigma, V = rotation_gradient(F)
#     R = U * wp.transpose(V)
#     grad = mat99()

#     for i in range(3):
#         for j in range(3):
#             grad[i * 3 + j, :] = flatten(
#                 wp.transpose(F) * unflatten(dR_dF[:, j * 3 + i])
#             )
#             for s in range(3):
#                 grad[i * 3 + j, s * 3 + j] += R[i, s]

#     return grad


# Returns the gradient of s(F) the first index is the dependent variable s and the second index is the flattened independent variable F
@wp.func
def stretch_gradient(F: wp.mat33) -> mat69:
    dR_dF, U, sigma, V = rotation_gradient(F)
    R = U * wp.transpose(V)
    grad = mat69()
    FT = wp.transpose(F)

    for i in range(3):
        for j in range(3):
            dS_dF_ij = FT * unflatten(dR_dF[:, j * 3 + i])
            dS_dF_ij[j, 0] += R[i, 0]
            dS_dF_ij[j, 1] += R[i, 1]
            dS_dF_ij[j, 2] += R[i, 2]
            grad[:, i * 3 + j] = sym_mat33_to_vec6(dS_dF_ij)

    return grad


@wp.func
def rotation_gradient(F: wp.mat33) -> tuple[mat99, wp.mat33, wp.vec3, wp.mat33]:
    U, sigma, V = rotational_svd(F)

    T0 = wp.mat33(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T1 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    T2 = wp.mat33(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)

    T0 = (1.0 / wp.sqrt(2.0)) * U * T0 * wp.transpose(V)
    T1 = (1.0 / wp.sqrt(2.0)) * U * T1 * wp.transpose(V)
    T2 = (1.0 / wp.sqrt(2.0)) * U * T2 * wp.transpose(V)

    t0 = flatten(T0)
    t1 = flatten(T1)
    t2 = flatten(T2)

    s0 = sigma[0]
    s1 = sigma[1]
    s2 = sigma[2]

    s01 = wp.max(s0 + s1, 1.0e-8)
    s12 = wp.max(s1 + s2, 1.0e-8)
    s02 = wp.max(s0 + s2, 1.0e-8)

    dR_dF = (
        (2.0 / s01) * wp.outer(t0, t0)
        + (2.0 / s12) * wp.outer(t1, t1)
        + (2.0 / s02) * wp.outer(t2, t2)
    )
    return dR_dF, U, sigma, V


@wp.kernel
def invert_diagonal_values(
    values: wp.array3d[wp.float32],
    dim: int,
):
    tid = wp.tid()

    for i in range(dim):
        values[tid, i, i] = 1.0 / values[tid, i, i]


def invert_diagonal_bsr(mat: ws.BsrMatrix) -> ws.BsrMatrix:
    if hasattr(mat, "eval"):
        mat = mat.eval()

    wp.launch(
        kernel=invert_diagonal_values,
        dim=mat.nnz,
        inputs=[mat.scalar_values, min(mat.block_shape[0], mat.block_shape[1])],
    )
    return mat


def plot_bsr(mat: ws.BsrMatrix, ax=None, title: str = "") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    offsets = mat.offsets.numpy()
    columns = mat.columns.numpy()
    values = mat.scalar_values.numpy()
    br, bc = mat.block_shape
    dense = np.zeros((mat.nrow * br, mat.ncol * bc), dtype=np.float32)
    for row in range(mat.nrow):
        for k in range(offsets[row], offsets[row + 1]):
            col = columns[k]
            dense[row * br : (row + 1) * br, col * bc : (col + 1) * bc] = values[k]

    cmap = LinearSegmentedColormap.from_list("rbg", ["red", "black", "green"])
    abs_max = max(float(np.abs(dense).max()), 1e-8)
    norm = TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)

    show = ax is None
    if show:
        _, ax = plt.subplots()

    im = ax.imshow(dense, aspect="auto", cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)

    if show:
        plt.show()


def plot_array(arr: wp.array):
    import matplotlib.pyplot as plt

    plt.plot(arr.numpy())
    plt.show()
