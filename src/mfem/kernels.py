import warp as wp

from mfem.utils import deformation_gradient, stretch_component, sym_mat33_to_vec6, vec6


@wp.kernel
def precompute_rest(
    particle_q: wp.array[wp.vec3],
    tets: wp.array2d[wp.uint32],
    rest: wp.array[wp.mat33],
    volume: wp.array[float],
):
    tid = wp.tid()

    x0 = particle_q[tets[tid, 0]]
    x1 = particle_q[tets[tid, 1]]
    x2 = particle_q[tets[tid, 2]]
    x3 = particle_q[tets[tid, 3]]

    # I am not sure whether it is worth storing the volume since it can be computed
    # easily from the rest pose
    edge_matrix = wp.matrix_from_rows(x3 - x0, x1 - x0, x2 - x0)
    volume[tid] = wp.abs(wp.determinant(edge_matrix)) / 6.0
    rest[tid] = wp.inverse(edge_matrix)


@wp.kernel
def evaluate_constraints(
    particle_q: wp.array[wp.vec3],
    stretch: wp.array[vec6],
    tets: wp.array2d[wp.uint32],
    rest: wp.array[wp.mat33],
    constraint: wp.array[vec6],
):

    tid = wp.tid()
    s = stretch[tid]

    F = deformation_gradient(particle_q, tets, rest, tid)
    stretch_grad = sym_mat33_to_vec6(stretch_component(F))

    sym = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))

    constraint[tid] = sym * (stretch_grad - s)


@wp.kernel
def evaluate_kinetic_gradient_dx(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    dt: float,
):
    tid = wp.tid()
    x = particle_q[tid]
    y = x + particle_qd[tid] * dt

    pass
