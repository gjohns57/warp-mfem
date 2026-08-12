from mfem.refinement.utils import SYM, INV_SYM, deformation_gradient_dF_dx, mat33_to_sym6, rotational_svd, stretch_component, stretch_gradient, sym6_to_mat33, unflatten
from mfem.types import mat63, vec6
from mfem.utils import mat66
from numpy.linalg import inv
import warp as wp

@wp.kernel
def get_mass_diagonal_and_attachment(
    active_particle_count: wp.array[wp.int32],
    particle_inv_mass: wp.array[wp.float32],
    attachment_predicate: wp.array[wp.int32],
    row_indices: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    mass_diagonal: wp.array[wp.mat33],
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        attachment_predicate[tid] = 0
        return

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        attachment_predicate[tid] = 1
        mass_diagonal[tid] = wp.identity(3, dtype=wp.float32)
    else:
        attachment_predicate[tid] = 0
        mass_diagonal[tid] = wp.identity(3, dtype=wp.float32) / inv_mass

    row_indices[tid] = tid
    col_indices[tid] = tid

@wp.kernel
def precompute_tet_stretch(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_stretch: wp.array[vec6],
):
    tid = wp.tid()
    x0 = particle_q[tet_indices[tid, 0]]
    x1 = particle_q[tet_indices[tid, 1]]
    x2 = particle_q[tet_indices[tid, 2]]
    x3 = particle_q[tet_indices[tid, 3]]

    e01 = x1 - x0
    e02 = x2 - x0
    e03 = x3 - x0
    deformed_pos = wp.matrix_from_cols(e01, e02, e03)
    inv_rest_pose = tet_poses[tid]
    F = deformed_pos * inv_rest_pose
    U, sigma, V = rotational_svd(F)
    tet_stretch[tid] = mat33_to_sym6(stretch_component(U, sigma, V))

@wp.kernel
def integrate_particles(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    dt: float,
    new_particle_q: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return

    g = gravity[0]
    new_particle_q[tid] = particle_q[tid] + particle_qd[tid] * dt + g * dt * dt / 2.0


@wp.kernel
def extract_attachment_indices(
    attachment_scan: wp.array[wp.int32],
    attachment_indices: wp.array[wp.int32],
):
    tid = wp.tid()

    if attachment_scan[tid] != attachment_scan[tid + 1]:
        attachment_indices[attachment_scan[tid]] = tid


@wp.kernel
def update_constraint_gradient_topology(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    row_indices: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    i0 = tet_indices[tid, 0]
    i1 = tet_indices[tid, 1]
    i2 = tet_indices[tid, 2]
    i3 = tet_indices[tid, 3]

    row_indices[tid * 4 + 0] = wp.int32(tid)
    row_indices[tid * 4 + 1] = wp.int32(tid)
    row_indices[tid * 4 + 2] = wp.int32(tid)
    row_indices[tid * 4 + 3] = wp.int32(tid)
    col_indices[tid * 4 + 0] = i0
    col_indices[tid * 4 + 1] = i1
    col_indices[tid * 4 + 2] = i2
    col_indices[tid * 4 + 3] = i3


@wp.kernel
def compute_constraints(
    active_tet_count: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    particle_q: wp.array[wp.vec3],
    stretch: wp.array[vec6],
    constraint: wp.array[vec6],
    grad_constraint_blocks: wp.array[mat63],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    x0 = particle_q[tet_indices[tid, 0]]
    x1 = particle_q[tet_indices[tid, 1]]
    x2 = particle_q[tet_indices[tid, 2]]
    x3 = particle_q[tet_indices[tid, 3]]
    s = stretch[tid]

    e01 = x1 - x0
    e02 = x2 - x0
    e03 = x3 - x0
    deformed_pos = wp.matrix_from_cols(e01, e02, e03)
    inv_rest_pose = tet_poses[tid]
    F = deformed_pos * inv_rest_pose

    U, sigma, V = rotational_svd(F)

    sF = mat33_to_sym6(stretch_component(U, sigma, V))
    dS_dF = stretch_gradient(F, U, sigma, V)
    dF_dX = deformation_gradient_dF_dx(inv_rest_pose)
    dC_dX = -(SYM * dS_dF * dF_dX)

    constraint[tid] = SYM * (sF - s)
    grad_constraint_blocks[tid * 4 + 0] = dC_dX[:, 0:3]
    grad_constraint_blocks[tid * 4 + 1] = dC_dX[:, 3:6]
    grad_constraint_blocks[tid * 4 + 2] = dC_dX[:, 6:9]
    grad_constraint_blocks[tid * 4 + 3] = dC_dX[:, 9:12]


@wp.kernel
def kinetic_gradient_kernel(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_tilde: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    dt: wp.float32,
    kinetic_gradient: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return

    kinetic_gradient[tid] = particle_mass[tid] * (particle_q[tid] - particle_q_tilde[tid]) / (dt * dt)


# We might need to add dt * dt to some of this stuff
@wp.kernel
def compute_g_lmbda(
    active_tet_count: wp.array[wp.int32],
    constraints: wp.array[vec6],
    gradient_ds: wp.array[vec6],
    inv_hessian_ds_blocks: wp.array[mat66],
    g_lmbda: wp.array[vec6],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    c = constraints[tid]
    H_si = inv_hessian_ds_blocks[tid]
    g_s = gradient_ds[tid]

    g_lmbda[tid] = c + SYM * H_si * g_s

@wp.kernel
def compute_H_lmbda_blocks(
    active_tet_count: wp.array[wp.int32],
    hessian_ds_blocks: wp.array[mat66],
    H_lmbda_blocks: wp.array[mat66],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    H_lmbda_blocks[tid] = INV_SYM * hessian_ds_blocks[tid] * INV_SYM


@wp.kernel
def compute_H_lmbda_g_lmbda(
    active_tet_count: wp.array[wp.int32],
    H_lmbda_blocks: wp.array[mat66],
    g_lmbda: wp.array[vec6],
    H_lmbda_g_lmbda: wp.array[vec6],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    H_lmbda_g_lmbda[tid] = H_lmbda_blocks[tid] * g_lmbda[tid]


@wp.kernel
def compute_H_lmbda_G_x_blocks(
    active_tet_count: wp.array[wp.int32],
    H_lmbda_blocks: wp.array[mat66],
    G_x_blocks: wp.array[mat63],
    H_lmbda_G_x_blocks: wp.array[mat63],
):
    tid = wp.tid()
    if tid >= active_tet_count[0] * 4:
        return

    H_lmbda_G_x_blocks[tid] = H_lmbda_blocks[tid // 4] * G_x_blocks[tid]
    H_lmbda_G_x_blocks[4 * tid + 1] = H_lmbda_blocks[tid] * G_x_blocks[4 * tid + 1]
    H_lmbda_G_x_blocks[4 * tid + 2] = H_lmbda_blocks[tid] * G_x_blocks[4 * tid + 2]
    H_lmbda_G_x_blocks[4 * tid + 3] = H_lmbda_blocks[tid] * G_x_blocks[4 * tid + 3]


@wp.kernel
def local_solve_lambda(
    active_tet_count: wp.array[wp.int32],
    H_lmbda_blocks: wp.array[mat66],
    g_lmbda: wp.array[vec6],
    constraint_gradient_dx_blocks: wp.array[mat63],
    constraint_gradient_dx_cols: wp.array[wp.int32],
    dx: wp.array[wp.vec3],
    new_lmbda: wp.array[vec6],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    lmbda = g_lmbda[tid]
    for i in range(4):
        lmbda -= constraint_gradient_dx_blocks[tid * 4 + i] * dx[constraint_gradient_dx_cols[tid * 4 + i]]
    new_lmbda[tid] = -H_lmbda_blocks[tid] * lmbda

@wp.kernel
def local_solve_stretch(
    active_tet_count: wp.array[wp.int32],
    inv_hessian_ds_blocks: wp.array[mat66],
    g_s: wp.array[vec6],
    new_lmbda: wp.array[vec6],
    ds: wp.array[vec6],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    ds[tid] = -inv_hessian_ds_blocks[tid] * (SYM * new_lmbda[tid] + g_s[tid])

@wp.kernel
def get_particle_velocity(
    active_particle_count: wp.array[wp.int32],
    particle_q_new: wp.array[wp.vec3],
    particle_q_old: wp.array[wp.vec3],
    dt: wp.float32,
    particle_qd: wp.array[wp.vec3]
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return

    particle_qd[tid] = (particle_q_new[tid] - particle_q_old[tid]) / dt
