import math
from typing import Any

import warp as wp

from mfem.types import vec6
from src.mfem.kernels import precompute_rest, precompute_tet_stretch


class Example:
    def __init__(self, position: wp.array[wp.vec3], tets: wp.array2d[wp.int32]):
        self.position = position
        self.tets = tets
        self.rest = wp.empty(tets.shape[0], dtype=wp.mat33)
        self.stretch = wp.empty(tets.shape[0], dtype=vec6)
        self.volume = wp.empty(tets.shape[0], dtype=wp.float32)
        self.n_particles = position.shape[0]
        self.n_tets = tets.shape[0]

        wp.launch(
            precompute_rest,
            dim=len(tets),
            inputs=[position, tets],
            outputs=[self.rest, self.volume],
        )

        wp.launch(
            precompute_tet_stretch,
            dim=len(tets),
            inputs=[position, tets, self.rest],
            outputs=[self.stretch],
        )


def create_single_tet_example() -> Example:
    position = wp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=wp.vec3,
    )
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.int32)
    return Example(position=position, tets=tets)


@wp.kernel
def flatten_mat_array(
    array: wp.array[Any], rows: int, cols: int, flattened: wp.array[Any]
):
    tid = wp.tid()
    stride = rows * cols

    for i in range(rows):
        for j in range(cols):
            flattened[tid * stride + i * cols + j] = array[tid][i, j]


@wp.kernel
def flatten_vec_array(array: wp.array[Any], len: int, flattened: wp.array[Any]):
    tid = wp.tid()
    stride = len

    for i in range(len):
        flattened[tid * stride + i] = array[tid][i]


def flatten_array(array: wp.array, requires_grad: bool = False) -> wp.array:
    array = array.flatten()

    if wp.types.type_is_matrix(array.dtype) or wp.types.type_is_vector(array.dtype):
        new_shape = math.prod(array.shape) * wp.types.type_size(array.dtype)
        return wp.array(
            ptr=array.ptr,
            dtype=array.dtype._wp_scalar_type_,
            shape=new_shape,
            requires_grad=requires_grad,
        )

    return array.flatten()


def reshape_array(
    array: wp.array, dtype: type[Any], requires_grad: bool = False
) -> wp.array:
    if wp.types.type_is_matrix(dtype) or wp.types.type_is_vector(dtype):
        if array.size % wp.types.type_size(dtype) != 0:
            raise ValueError(
                f"Array size {array.size} is not a multiple of dtype size {wp.types.type_size(dtype)}"
            )

        unflattened = wp.array(
            ptr=array.ptr,
            dtype=dtype,
            shape=(array.size // wp.types.type_size(dtype),),
            requires_grad=requires_grad,
        )

        return unflattened
    raise ValueError(f"Unsupported dtype: {dtype}")
