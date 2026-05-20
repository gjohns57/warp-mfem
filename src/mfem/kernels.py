import warp as wp

from mfem.utils import (
    deformation_gradient,
    deformation_gradient_dF_dx,
    mat63,
    stretch_component,
    stretch_gradient,
    sym_mat33_to_vec6,
    vec6,
)


@wp.kernel
def precompute_rest(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.uint32],
    rest: wp.array[wp.mat33],
    volume: wp.array[wp.float32],
):
    tid = wp.tid()

    x0 = particle_q[tets[tid, 0]]
    x1 = particle_q[tets[tid, 1]]
    x2 = particle_q[tets[tid, 2]]
    x3 = particle_q[tets[tid, 3]]

    # I am not sure whether it is worth storing the volume since it can be computed
    # easily from the rest pose
    edge_matrix = wp.matrix_from_rows(x1 - x0, x2 - x0, x3 - x0)
    volume[tid] = wp.abs(wp.determinant(edge_matrix)) / 6.0
    rest[tid] = wp.inverse(edge_matrix)


@wp.kernel
def precompute_mass_matrix(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.uint32],
    density: wp.array[wp.float32],
    col_indices: wp.array[wp.uint32],
    row_indices: wp.array[wp.uint32],
    values: wp.array[wp.mat33],
):
    tid = wp.tid()

    x0 = particle_q[tets[tid, 0]]
    x1 = particle_q[tets[tid, 1]]
    x2 = particle_q[tets[tid, 2]]
    x3 = particle_q[tets[tid, 3]]

    row_indices[tid * 4 + 0] = tid
    row_indices[tid * 4 + 1] = tid
    row_indices[tid * 4 + 2] = tid
    row_indices[tid * 4 + 3] = tid

    col_indices[tid * 4 + 0] = tets[tid, 0]
    col_indices[tid * 4 + 1] = tets[tid, 1]
    col_indices[tid * 4 + 2] = tets[tid, 2]
    col_indices[tid * 4 + 3] = tets[tid, 3]

    edge_matrix = wp.matrix_from_rows(x1 - x0, x2 - x0, x3 - x0)
    volume = wp.abs(wp.determinant(edge_matrix)) / 6.0

    values[4 * tid + 0] = wp.identity(3) * density[tid] * volume / 4
    values[4 * tid + 1] = wp.identity(3) * density[tid] * volume / 4
    values[4 * tid + 2] = wp.identity(3) * density[tid] * volume / 4
    values[4 * tid + 3] = wp.identity(3) * density[tid] * volume / 4


@wp.kernel
def evaluate_constraints(
    particle_q: wp.array[wp.vec3],
    stretch: wp.array[vec6],
    tets: wp.array2d[wp.uint32],
    rest: wp.array[wp.mat33],
    constraint: wp.array2d[wp.float32],
):

    tid = wp.tid()
    s = stretch[tid]

    F = deformation_gradient(particle_q, tets, rest, tid)
    stretch_grad = sym_mat33_to_vec6(stretch_component(F))

    c = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)) * (stretch_grad - s)

    constraint[tid, 0] = c[0]
    constraint[tid, 1] = c[1]
    constraint[tid, 2] = c[2]
    constraint[tid, 3] = c[3]
    constraint[tid, 4] = c[4]
    constraint[tid, 5] = c[5]


# @wp.kernel
# def evaluate_kinetic_gradient_dx(
#     particle_qd: wp.array[wp.vec3],
#     kinetic_gradient_dx: wp.array[wp.vec3],
#     dt: float,
# ):
#     tid = wp.tid()
#     kinetic_gradient_dx[tid] = particle_qd[tid] / dt


@wp.kernel
def evalutate_constraint_gradient_dx(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.uint32],
    rest: wp.array[wp.mat33],
    row_idx: wp.array[wp.uint32],
    col_idx: wp.array[wp.uint32],
    values: wp.array[mat63],
):
    tid = wp.tid()

    row_idx[tid * 4 + 0] = wp.uint32(tid)
    row_idx[tid * 4 + 1] = wp.uint32(tid)
    row_idx[tid * 4 + 2] = wp.uint32(tid)
    row_idx[tid * 4 + 3] = wp.uint32(tid)

    col_idx[tid * 4 + 0] = tets[tid, 0]
    col_idx[tid * 4 + 1] = tets[tid, 1]
    col_idx[tid * 4 + 2] = tets[tid, 2]
    col_idx[tid * 4 + 3] = tets[tid, 3]

    F = deformation_gradient(particle_q, tets, rest, tid)
    dS_dF = stretch_gradient(F)  # 6x9
    dF_dX = deformation_gradient_dF_dx(rest[tid])  # 9x12

    dS_dX = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)) * dS_dF * dF_dX  # 6x12

    values[tid * 4 + 0] = dS_dX[:, 0:3]
    values[tid * 4 + 1] = dS_dX[:, 3:6]
    values[tid * 4 + 2] = dS_dX[:, 6:9]
    values[tid * 4 + 3] = dS_dX[:, 9:12]
