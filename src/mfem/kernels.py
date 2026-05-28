import warp as wp

from .types import mat63, vec6
from .utils import (
    deformation_gradient,
    deformation_gradient_dF_dx,
    stretch_component,
    stretch_gradient,
    sym_mat33_to_vec6,
)


@wp.kernel
def precompute_tet_stretch(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.int32],
    rest: wp.array[wp.mat33],
    tet_stretch: wp.array[vec6],
):
    tid = wp.tid()

    F = deformation_gradient(particle_q, tets, rest, tid)
    tet_stretch[tid] = sym_mat33_to_vec6(stretch_component(F))


@wp.kernel
def precompute_rest(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.int32],
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
    rest[tid] = wp.inverse(edge_matrix)
    volume[tid] = wp.abs(wp.determinant(edge_matrix)) / 6.0


@wp.kernel
def precompute_mass_matrix(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.int32],
    density: wp.array[wp.float32],
    mass: wp.array[wp.mat33],
):
    tid = wp.tid()

    x0 = particle_q[tets[tid, 0]]
    x1 = particle_q[tets[tid, 1]]
    x2 = particle_q[tets[tid, 2]]
    x3 = particle_q[tets[tid, 3]]

    edge_matrix = wp.matrix_from_rows(x1 - x0, x2 - x0, x3 - x0)
    volume = wp.abs(wp.determinant(edge_matrix)) / 6.0

    wp.atomic_add(
        mass,
        tets[tid, 0],
        wp.identity(3, dtype=wp.float32) * density[tets[tid, 0]] * volume / 4.0,
    )
    wp.atomic_add(
        mass,
        tets[tid, 1],
        wp.identity(3, dtype=wp.float32) * density[tets[tid, 1]] * volume / 4.0,
    )
    wp.atomic_add(
        mass,
        tets[tid, 2],
        wp.identity(3, dtype=wp.float32) * density[tets[tid, 2]] * volume / 4.0,
    )
    wp.atomic_add(
        mass,
        tets[tid, 3],
        wp.identity(3, dtype=wp.float32) * density[tets[tid, 3]] * volume / 4.0,
    )


@wp.kernel
def precompute_bsr_topology(
    tets: wp.array2d[wp.int32],
    row_idx: wp.array[wp.int32],
    col_idx: wp.array[wp.int32],
):
    tid = wp.tid()

    row_idx[tid * 4 + 0] = wp.int32(tid)
    row_idx[tid * 4 + 1] = wp.int32(tid)
    row_idx[tid * 4 + 2] = wp.int32(tid)
    row_idx[tid * 4 + 3] = wp.int32(tid)

    col_idx[tid * 4 + 0] = tets[tid, 0]
    col_idx[tid * 4 + 1] = tets[tid, 1]
    col_idx[tid * 4 + 2] = tets[tid, 2]
    col_idx[tid * 4 + 3] = tets[tid, 3]


@wp.kernel
def evaluate_constraints(
    particle_q: wp.array[wp.vec3],
    stretch: wp.array[vec6],
    tets: wp.array2d[wp.int32],
    rest: wp.array[wp.mat33],
    constraint: wp.array[vec6],
):

    tid = wp.tid()
    s = stretch[tid]

    F = deformation_gradient(particle_q, tets, rest, tid)
    stretch_grad = sym_mat33_to_vec6(stretch_component(F))

    constraint[tid] = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)) * (stretch_grad - s)


# @wp.kernel
# def evaluate_kinetic_gradient_dx(
#     particle_qd: wp.array[wp.vec3],
#     kinetic_gradient_dx: wp.array[wp.vec3],
#     dt: float,
# ):
#     tid = wp.tid()
#     kinetic_gradient_dx[tid] = particle_qd[tid] / dt


@wp.kernel
def evaluate_constraint_gradient_dx(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.int32],
    rest: wp.array[wp.mat33],
    values: wp.array[mat63],
):
    tid = wp.tid()

    F = deformation_gradient(particle_q, tets, rest, tid)
    dS_dF = stretch_gradient(F)  # 6x9
    dF_dX = deformation_gradient_dF_dx(rest[tid])  # 9x12

    dS_dX = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)) * dS_dF * dF_dX  # 6x12

    values[tid * 4 + 0] = dS_dX[:, 0:3]
    values[tid * 4 + 1] = dS_dX[:, 3:6]
    values[tid * 4 + 2] = dS_dX[:, 6:9]
    values[tid * 4 + 3] = dS_dX[:, 9:12]


@wp.kernel
def test_deform_kernel(x: wp.array[wp.vec3]):
    tid = wp.tid()
    shear = wp.mat33(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    x[tid] = shear * x[tid]
