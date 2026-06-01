import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.sparse as ws
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CMAP = LinearSegmentedColormap.from_list("rbg", ["red", "black", "green"])


def zero_norm(vmin: float, vmax: float):
    """Norm that always maps 0 to the colormap midpoint (black) with a symmetric range."""
    abs_max = max(abs(vmin), abs(vmax), 1e-8)
    return TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)

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


def flatten_array(array: wp.array) -> wp.array:

    if wp.types.type_is_matrix(array.dtype) or wp.types.type_is_vector(array.dtype):
        new_shape = math.prod(array.shape) * wp.types.type_size(array.dtype)

        new_grad = None
        if isinstance(array.grad, wp.array):
            new_grad = wp.array(
                ptr=array.grad.ptr,
                dtype=array.dtype._wp_scalar_type_,
                shape=new_shape,
                requires_grad=False,
            )
            new_grad._ref = array.grad

        new_array = wp.array(
            ptr=array.ptr,
            grad=new_grad,
            dtype=array.dtype._wp_scalar_type_,
            shape=new_shape,
            requires_grad=array.requires_grad,
        )

        new_array._ref = array
        return new_array
    return array.flatten()


def reshape_array(array: wp.array, dtype: type[Any]) -> wp.array:
    if wp.types.type_is_matrix(dtype) or wp.types.type_is_vector(dtype):
        if array.size % wp.types.type_size(dtype) != 0:
            raise ValueError(
                f"Array size {array.size} is not a multiple of dtype size {wp.types.type_size(dtype)}"
            )

        new_shape = (array.size // wp.types.type_size(dtype),)
        new_grad = None
        if isinstance(array.grad, wp.array):
            new_grad = wp.array(
                ptr=array.grad.ptr,
                dtype=dtype,
                shape=new_shape,
                requires_grad=False,
            )
            new_grad._ref = array.grad

        new_array = wp.array(
            ptr=array.ptr,
            dtype=dtype,
            shape=new_shape,
            requires_grad=array.requires_grad,
            grad=new_grad,
        )

        new_array._ref = array
        return new_array
    raise ValueError(f"Unsupported dtype: {dtype}")


@wp.kernel
def _bsr_to_dense_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array3d[wp.float32],
    block_rows: int,
    block_cols: int,
    out: wp.array2d[wp.float32],
):
    row = wp.tid()
    for k in range(offsets[row], offsets[row + 1]):
        col = columns[k]
        for i in range(block_rows):
            for j in range(block_cols):
                out[row * block_rows + i, col * block_cols + j] = values[k, i, j]


def bsr_to_dense(mat: ws.BsrMatrix) -> wp.array:
    br, bc = mat.block_shape
    out = wp.zeros((mat.nrow * br, mat.ncol * bc), dtype=wp.float32)
    wp.launch(
        _bsr_to_dense_kernel,
        dim=mat.nrow,
        inputs=[mat.offsets, mat.columns, mat.scalar_values, br, bc, out],
    )
    return out
