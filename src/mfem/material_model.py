import warp as wp

from src.mfem.utils import vec6


class StretchMaterialModel:
    def __init__(self):
        raise NotImplementedError()

    def energy(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()

    def gradient_ds(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()

    def hessian_ds(
        self,
        stretch: wp.array[vec6],
        mu: wp.array[wp.float32],
        lmbda: wp.array[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()
