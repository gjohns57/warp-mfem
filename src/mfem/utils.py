import warp as wp
from warp.types import matrix, vector

"""
Utilities for MFEM elastic energy functions:
I found this awesome guide: https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf
"""


class vec6(vector(length=6, dtype=wp.float32)):
    """Symmetric 3x3 matrix stored as a 6-element vector."""


class vec9(vector(length=9, dtype=wp.float32)):
    """"""


class mat99(matrix(shape=(9, 9), dtype=wp.float32)):
    f"""9x9 Matrix for partial^2 I_1 / partial f^2"""


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
def deformation_gradient(
    position: wp.array[wp.vec3],
    indices: wp.array2d[wp.uint32],
    inv_rest_matrix: wp.array[wp.mat33],
    tid: int,
):
    x0 = position[indices[tid, 0]]
    x1 = position[indices[tid, 1]]
    x2 = position[indices[tid, 2]]
    x3 = position[indices[tid, 3]]

    deformed_edge_matrix = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    rest_edge_matrix = inv_rest_matrix[tid]
    return deformed_edge_matrix * rest_edge_matrix


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


@wp.func
def stretch_gradient(F: wp.mat33) -> mat99:
    dR_dF, U, sigma, V = rotation_gradient(F)
    R = U * wp.transpose(V)
    grad = mat99()

    for i in range(3):
        for j in range(3):
            grad[i * 3 + j, :] = flatten(wp.transpose(F) * unflatten(dR_dF[:, j * 3 + i]))
            for s in range(3):
                grad[i * 3 + j, s * 3 + j] += R[i, s]

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
