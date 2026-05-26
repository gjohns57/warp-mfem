import warp as wp

from .utils import vec6


class StretchMaterialModel:
    @staticmethod
    def energy(
        stretch: wp.array[vec6],
        material: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()

    @staticmethod
    def gradient_ds(
        stretch: wp.array[vec6],
        material: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()

    @staticmethod
    def hessian_ds(
        stretch: wp.array[vec6],
        material: wp.array2d[wp.float32],
        volume: wp.array[wp.float32],
    ):
        raise NotImplementedError()
