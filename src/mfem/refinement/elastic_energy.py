from mfem.refinement.additional_state import AdditionalState
from mfem.refinement.utils import INV_SYM, SYM
from mpmath import e1
import warp as wp
import warp.sparse as ws
from mfem.types import mat66, vec6
# from mfem.refinement.solver import AdditionalState
# from mfem.refinement.utils import sym6_to_mat33, grad_det_sym_vec6
from newton import State
import numpy as np

# # TODO: implement cholesky factorization to invert PSD mat
# @wp.func
# def mat66_inv(mat: mat66) -> mat66:
#     return wp.identity(6, dtype=wp.float32)

# @wp.func
# def psd_project(mat: mat66) -> mat66:
#     pass

# class ElasticEnergyBase:
#     @staticmethod
#     def energy(
#         state: State,
#         additional_state: AdditionalState,
#         energy: wp.array[wp.float32],
#     ):
#         raise NotImplementedError()

#     @staticmethod
#     def gradient_ds(
#         state: State,
#         additional_state: AdditionalState,
#         gradient_ds: wp.array[vec6],
#     ):
#         raise NotImplementedError()

#     @staticmethod
#     def hessian_ds(
#         state: State,
#         additional_state: AdditionalState,
#         hessian_ds_blocks: wp.array[mat66],
#         row_indices: wp.array[wp.int32],
#         col_indices: wp.array[wp.int32],
#     ):
#         raise NotImplementedError()

# @wp.kernel
# def neohookean_energy_kernel(
#     active_tets: wp.array[wp.int32],
#     stretch: wp.array[vec6],
#     rest_volume: wp.array[wp.float32],
#     material: wp.array2d[wp.float32],
#     energy: wp.array[wp.float32],
#     gradient_ds: wp.array[vec6],
#     hessian_ds: wp.array[mat66],
#     inv_hessian_ds: wp.array[mat66],
# ):
#     tid = wp.tid()

#     if tid >= active_tets.shape[0]:
#         return

#     Sym = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
#     S = sym6_to_mat33(stretch[tid])
#     Q = wp.mat33()
#     eig = wp.vec3()
#     wp.eig3(S, Q, eig)

#     mu = material[tid, 0]
#     lmbda = material[tid, 1]

#     I2 = eig[0] * eig[0] + eig[1] * eig[1] + eig[2] * eig[2]
#     I3 = eig[0] * eig[1] * eig[2]
#     gradI2ds = 2.0 * Sym * stretch[tid]
#     gradI3ds = grad_det_sym_vec6(stretch[tid])

#     energy[tid] = rest_volume[tid] * (
#         mu * (I2 - 3.0) / 2.0 - mu * (I3 - 1.0) + lmbda / 2.0 * (I3 - 1.0) * (I3 - 1.0)
#     )


#     pass

# # @wp.func
# def neohookean_energy(s: vec6) -> tuple[wp.float32, vec6, mat66]:
#     S = sym6_to_mat33(s)
#     Q = wp.mat33()
#     eig = wp.vec3()
#     wp.eig3(S, Q, eig)

#     energy =


#     # energy[tid] = volume[tid] * (
#     #     mu * (I2 - 3.0) / 2.0 - mu * (I3 - 1.0) + lmbda / 2.0 * (I3 - 1.0) * (I3 - 1.0)
#     # )

@wp.func
def elastic_invariant1(s: vec6) -> wp.float32:
    return s[0] + s[1] + s[2]

@wp.func
def elastic_invariant2(s: vec6) -> wp.float32:
    return s[0] * s[1] + s[1] * s[2] + s[2] * s[0] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])

@wp.func
def grad_elastic_invariant1(s: vec6) -> vec6:
    return vec6(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)

@wp.func
def hessian_elastic_invariant1(s: vec6) -> mat66:
    return mat66()

@wp.func
def grad_elastic_invariant2(s: vec6) -> vec6:
    return 2.0 * SYM * s


@wp.kernel
def arap_energy_kernel(
    active_tet_count: wp.array[wp.int32],
    tet_stretch: wp.array[vec6],
    tet_materials: wp.array2d[wp.float32],
    tet_poses: wp.array[wp.mat33],
    energy: wp.array[wp.float32],
    gradient: wp.array[vec6],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    mu = tet_materials[tid, 0]
    I1 = elastic_invariant1(tet_stretch[tid])
    I2 = elastic_invariant2(tet_stretch[tid])
    vol = 6.0 / (wp.determinant(tet_poses[tid]))

    energy[tid] = vol * (
        mu * (I2 + 3.0 - 2.0 * I1) / 2.0
    )
    gradient[tid] = vol * (
        mu * (grad_elastic_invariant2(tet_stretch[tid]) - 2.0 * grad_elastic_invariant1(tet_stretch[tid])) / 2.0
    )


@wp.kernel
def arap_energy_hessian_kernel(
    active_tet_count: wp.array[wp.int32],
    tet_materials: wp.array2d[wp.float32],
    tet_poses: wp.array[wp.mat33],
    hessian_blocks: wp.array[mat66],
    inv_hessian_blocks: wp.array[mat66],
):
    tid = wp.tid()
    if tid >= active_tet_count[0]:
        return

    mu = tet_materials[tid, 0]
    vol = 6.0 / (wp.determinant(tet_poses[tid]))

    hessian_blocks[tid] = mu * vol * SYM
    inv_hessian_blocks[tid] = 1.0 / (mu * vol) * INV_SYM

def arap_energy(
    state: State,
    additional_state: AdditionalState,
    energy: wp.array[wp.float32],
    gradient: wp.array[vec6],
    hessian_blocks: wp.array[mat66],
    inv_hessian_blocks: wp.array[mat66],
    update_hessian_blocks: bool,
) -> None:
    if update_hessian_blocks:
        wp.launch(
            arap_energy_hessian_kernel,
            dim=additional_state.tet_indices.shape[0],
            inputs=[
                additional_state.active_tet_count,
                additional_state.tet_materials,
                additional_state.tet_poses,
            ],
            outputs=[
                hessian_blocks,
                inv_hessian_blocks,
            ],
        )

    wp.launch(
        arap_energy_kernel,
        dim=additional_state.tet_indices.shape[0],
        inputs=[
            additional_state.active_tet_count,
            additional_state.tet_stretch,
            additional_state.tet_materials,
            additional_state.tet_poses,
        ],
        outputs=[
            energy,
            gradient,
        ],
    )

def neohookean(eig: np.ndarray, mu: float, lmbda: float) -> float:
    I2 = eig[0] * eig[0] + eig[1] * eig[1] + eig[2] * eig[2]
    I3 = eig[0] * eig[1] * eig[2]
    return mu * (I2 - 3.0) / 2.0 - mu * (I3 - 1.0) + lmbda / 2.0 * (I3 - 1.0) * (I3 - 1.0)

def gradient(eig: np.ndarray, mu: float, lmbda: float) -> np.ndarray:
    gradI2ds = 2.0 * np.array([eig[0], eig[1], eig[2]])
    gradI3ds = np.array([eig[1] * eig[2], eig[0] * eig[2], eig[0] * eig[1]])
    I3 = eig[0] * eig[1] * eig[2]
    return mu * (gradI2ds / 2.0) - mu * gradI3ds + lmbda * (I3 - 1.0) * gradI3ds

def hessian_deig(eig: np.ndarray, mu: float, lmbda: float) -> np.ndarray:
    hessI2ds = 2.0 * np.eye(3)
    hessI3ds = np.array([
        [0.0, eig[2], eig[1]],
        [eig[2], 0.0, eig[0]],
        [eig[1], eig[0], 0.0]
    ])
    return mu * hessI2ds / 2.0 + hessI3ds

def neohookean_test(s: np.ndarray, mu, lmbda):
    S = np.array([
        [s[0], s[3], s[4]],
        [s[3], s[1], s[5]],
        [s[4], s[5], s[2]]
    ], dtype=np.float32)

    eig = np.linalg.eig(S)
    n0 = eig.eigenvectors[0]
    n1 = eig.eigenvectors[1]
    n2 = eig.eigenvectors[2]

    print(n0, n1, n2)

if __name__ == "__main__":
    s = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    neohookean_test(s, mu=1.0, lmbda=1.0)
