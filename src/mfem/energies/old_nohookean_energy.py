import warp as wp
import warp.sparse as ws

from ..numerical import psd_fix_6
from ..types import mat66, vec6
from ..utils import (
    det_sym_vec6,
    frob2_sym_vec6,
    grad_det_sym_vec6,
    hess_det_sym_vec6,
    tr_sym_vec6,
)
from .material_model import StretchMaterialModel

MIN_J = 0.0


@wp.kernel
def energy_kernel(
    stretch: wp.array[vec6],
    tet_materials: wp.array2d[wp.float32],
    volume: wp.array[wp.float32],
    energy: wp.array[wp.float32],
):
    tid = wp.tid()
    mu = tet_materials[tid, 0]
    lmbda = tet_materials[tid, 1]

    frob = frob2_sym_vec6(stretch[tid])
    logJ = wp.log(det_sym_vec6(stretch[tid]))

    energy[tid] = volume[tid] * (
        (mu / 2.0 * (frob - 3.0)) - mu * logJ + lmbda / 2.0 * logJ * logJ
    )


@wp.kernel
def gradient_ds_kernel(
    stretch: wp.array[vec6],
    tet_materials: wp.array2d[wp.float32],
    volume: wp.array[wp.float32],
    gradient_ds: wp.array[vec6],
):
    tid = wp.tid()
    mu = tet_materials[tid, 0]
    lmbda = tet_materials[tid, 1]

    Sym = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
    J = wp.max(det_sym_vec6(stretch[tid]), 0.1)
    gradJds = grad_det_sym_vec6(stretch[tid])

    # dPhi/ds = mu * volume * (S - I)
    gradient_ds[tid] = volume[tid] * (
        mu * Sym * stretch[tid] + (lmbda * wp.log(J) - mu) / J * gradJds
    )


@wp.kernel
def hessian_ds_kernel(
    stretch: wp.array[vec6],
    tet_materials: wp.array2d[wp.float32],
    volume: wp.array[wp.float32],
    hessian_ds: wp.array[mat66],
    hessian_dsi: wp.array[mat66],
):
    tid = wp.tid()
    mu = tet_materials[tid, 0]
    lmbda = tet_materials[tid, 1]

    Sym = wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
    J = wp.max(det_sym_vec6(stretch[tid]), 0.1)
    logJ = wp.log(J)
    gradJds = grad_det_sym_vec6(stretch[tid])
    hessJds = hess_det_sym_vec6(stretch[tid])

    hess = volume[tid] * (
        mu * Sym
        + (lmbda * (1.0 - logJ) + mu) / (J * J) * wp.outer(gradJds, gradJds)
        + (lmbda * logJ - mu) / J * hessJds
    )

    fix = psd_fix_6(hess)
    hessian_ds[tid] = fix.mat
    hessian_dsi[tid] = fix.inv


class StableNeohookeanEnergy(StretchMaterialModel):
    @staticmethod
    def energy(
        stretch: wp.array[vec6],
        tet_materials: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
        energy: wp.array[wp.float32],
    ):
        wp.launch(
            energy_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, tet_materials, volume],
            outputs=[energy],
        )

    @staticmethod
    def gradient_ds(
        stretch: wp.array[vec6],
        tet_materials: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
    ):
        gradient_ds = wp.empty(shape=(stretch.shape[0],), dtype=vec6)
        wp.launch(
            gradient_ds_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, tet_materials, volume],
            outputs=[gradient_ds],
        )
        return gradient_ds

    @staticmethod
    def hessian_ds(
        stretch: wp.array[vec6],
        material: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
    ):
        # d^2Phi/ds^2 = mu * volume * (S - I)
        diagonal_blocks = wp.empty(shape=(stretch.shape[0],), dtype=mat66)
        inverse_diagonal_blocks = wp.empty(shape=(stretch.shape[0],), dtype=mat66)
        wp.launch(
            hessian_ds_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, material, volume],
            outputs=[diagonal_blocks, inverse_diagonal_blocks],
        )
        H_s = ws.bsr_diag(diagonal_blocks, block_type=mat66)
        H_si = ws.bsr_diag(inverse_diagonal_blocks, block_type=mat66)
        return H_s, H_si
