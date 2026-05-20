import warp as wp
import warp.sparse as ws

from .material_model import StretchMaterialModel
from .utils import frob2_sym_vec6, mat66, tr_sym_vec6, vec6


@wp.kernel
def energy_kernel(
    stretch: wp.array[vec6],
    mu: wp.array[wp.float32],
    lmbda: wp.array[wp.float32],
    volume: wp.array[wp.float32],
    energy: wp.array[wp.float32],
):
    tid = wp.tid()
    energy[tid] = (
        volume[tid]
        * mu[tid]
        / 2
        * (frob2_sym_vec6(stretch[tid]) + 3 - 2 * tr_sym_vec6(stretch[tid]))
    )


@wp.kernel
def gradient_ds_kernel(
    stretch: wp.array[vec6],
    mu: wp.array[wp.float32],
    lmbda: wp.array[wp.float32],
    volume: wp.array[wp.float32],
    gradient_ds: wp.array2d[wp.float32],
):
    tid = wp.tid()

    # dPhi/ds = mu * volume * (S - I)
    grad = (
        mu[tid]
        * volume[tid]
        * (
            wp.diag(vec6(1, 1, 1, 2, 2, 2)) * stretch[tid]
            - wp.diag(vec6(1, 1, 1, 0, 0, 0))
        )
    )

    for i in range(6):
        gradient_ds[tid, i] = grad[i]


@wp.kernel
def hessian_ds_kernel(
    stretch: wp.array[vec6],
    mu: wp.array[wp.float32],
    lmbda: wp.array[wp.float32],
    volume: wp.array[wp.float32],
    hessian_ds: wp.array[mat66],
):
    tid = wp.tid()

    hessian_ds[tid] = mu[tid] * volume[tid] * wp.diag(vec6(1, 1, 1, 2, 2, 2))


class ARAPEnergy(StretchMaterialModel):
    def __init__(self):
        pass

    def energy(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        return super().energy(stretch, mu, lmbda, volume)

    def gradient_ds(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        gradient_ds = wp.empty(shape=(stretch.shape[0], 6), dtype=wp.float32)
        wp.launch(
            gradient_ds_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, mu, lmbda, volume],
            outputs=[gradient_ds],
        )
        return gradient_ds.reshape((-1, 1))

    def hessian_ds(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        # d^2Phi/ds^2 = mu * volume * (S - I)
        diagonal_blocks = wp.empty(shape=(stretch.shape[0],), dtype=mat66)
        wp.launch(
            hessian_ds_kernel,
            dim=stretch.shape[0],
            inputs=[stretch, mu, lmbda, volume],
            outputs=[diagonal_blocks],
        )
        H_s = ws.bsr_diag(diagonal_blocks, block_type=mat66)
        return H_s
