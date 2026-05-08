import warp as wp
from warp.types import vector


class vec6(vector(length=6, dtype=wp.float32)):
    """Symmetric 3x3 matrix stored as a 6-element vector."""


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
def polar_decomosition(A: wp.mat33):

    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, sigma, V)

    return V * wp.diag(sigma) * wp.transpose(V)
