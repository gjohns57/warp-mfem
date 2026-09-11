import warp as wp
import warp.sparse as ws

from ..types import mat66, vec6
from ..utils import frob2_sym_vec6, tr_sym_vec6
from .material_model import StretchMaterialModel


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
    energy[tid] = (
        volume[tid]
        * mu
        / 2.0
        * (frob2_sym_vec6(stretch[tid]) + 3.0 - 2.0 * tr_sym_vec6(stretch[tid]))
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

    # dPhi/ds = mu * volume * (S - I)
    gradient_ds[tid] = (
        mu
        * volume[tid]
        * (
            wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)) * stretch[tid]
            - vec6(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        )
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

    hessian_ds[tid] = mu * volume[tid] * wp.diag(vec6(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
    hessian_dsi[tid] = wp.diag(vec6(1.0, 1.0, 1.0, 0.5, 0.5, 0.5)) / (mu * volume[tid])


class ARAPEnergy(StretchMaterialModel):
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
    ) -> tuple[ws.BsrMatrix, ws.BsrMatrix]:
        # d^2Phi/ds^2 = mu * volume * (S - I)
        diagonal_blocks = wp.empty(shape=(stretch.shape[0],), dtype=mat66)
        inv_diagonal_bocks = wp.empty_like(diagonal_blocks)
        wp.launch(
            hessian_ds_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, material, volume],
            outputs=[diagonal_blocks, inv_diagonal_bocks],
        )
        H_s = ws.bsr_diag(diagonal_blocks, block_type=mat66)
        H_si = ws.bsr_diag(inv_diagonal_bocks, block_type=mat66)
        return H_s, H_si
