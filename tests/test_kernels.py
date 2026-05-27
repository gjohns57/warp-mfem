import numpy as np
import pytest
import scipy.spatial
import warp as wp
import warp.autograd as wg
import warp.sparse as ws
from numpy._core.numeric import require
from torch.nn.modules import loss

from mfem.kernels import (
    evaluate_constraint_gradient_dx,
    evaluate_constraints,
    precompute_bsr_topology,
)
from mfem.utils import mat63, vec6

# from mfem.kernels import *


# def test_evaluate_constraints1():
#     position = wp.array(
#         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         dtype=wp.vec3,
#     )
#     stretch = wp.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=vec6)
#     tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.uint32)
#     # Edge matrix rows: (x0-x3), (x1-x3), (x2-x3) — kernel inverts this to get F
#     rest = wp.array(
#         [[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]], dtype=wp.mat33
#     )
#     constraint = wp.zeros(1, dtype=vec6)

#     wp.launch(
#         evaluate_constraints,
#         dim=1,
#         inputs=[position, stretch, tets, rest],
#         outputs=[constraint],
#     )

#     c = constraint.numpy()
#     assert c[0, 0] == pytest.approx(0.0, abs=1e-6)
#     assert c[0, 1] == pytest.approx(0.0, abs=1e-6)
#     assert c[0, 2] == pytest.approx(0.0, abs=1e-6)
#     assert c[0, 3] == pytest.approx(0.0, abs=1e-6)
#     assert c[0, 4] == pytest.approx(0.0, abs=1e-6)
#     assert c[0, 5] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_constraints_rotation():
    og_position = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    new_position = scipy.spatial.transform.Rotation.from_euler(
        "xyz", [0, 0, 12], degrees=True
    ).apply(og_position)

    position = wp.array(new_position, dtype=wp.vec3)
    stretch = wp.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=vec6)
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.int32)

    cpu_rest = np.hstack(
        [
            og_position[1, :] - og_position[0, :],
            og_position[2, :] - og_position[0, :],
            og_position[3, :] - og_position[0, :],
        ]
    ).reshape(3, 3)

    rest = wp.array(
        np.linalg.inv(cpu_rest),
        dtype=wp.mat33,
    )

    constraint = wp.zeros(1, dtype=vec6)

    wp.launch(
        evaluate_constraints,
        dim=1,
        inputs=[position, stretch, tets, rest],
        outputs=[constraint],
    )

    c = constraint.numpy()
    assert c[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 2] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 3] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 4] == pytest.approx(0.0, abs=1e-5)
    assert c[0, 5] == pytest.approx(0.0, abs=1e-5)


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


@wp.kernel
def vec6_array_to_array2d(vec6_array: wp.array[vec6], output: wp.array2d[wp.float32]):
    tid = wp.tid()

    for j in range(6):
        output[tid, j] = vec6_array[tid][j]


@wp.kernel
def array2d_to_vec3_array(array2d: wp.array2d[wp.float32], output: wp.array[wp.vec3]):
    tid = wp.tid()
    for i in range(3):
        output[tid][i] = array2d[tid, i]


def flattened_evaluate_constraint(
    position: wp.array[wp.float32],
    stretch: wp.array[vec6],
    tets: wp.array2d[wp.int32],
    rest: wp.array[wp.vec3],
    n_tets: int,
) -> wp.array2d[wp.float32]:

    constraint = wp.zeros((n_tets, 6), dtype=wp.float32, requires_grad=True)

    constraints_vec = wp.zeros(n_tets, dtype=vec6)
    position_vec = wp.zeros(n_tets, dtype=wp.vec3, requires_grad=True)

    wp.launch(
        array2d_to_vec3_array,
        dim=position.shape[0],
        inputs=[position, position_vec],
    )

    wp.launch(
        evaluate_constraints,
        dim=1,
        inputs=[position_vec, stretch, tets, rest],
        outputs=[constraints_vec],
    )
    wp.launch(
        vec6_array_to_array2d,
        dim=1,
        inputs=[constraints_vec],
        outputs=[constraint],
    )

    return constraint


def test_elastic_gradient_dx():
    # Unit tet: x0 at origin, edges along coordinate axes
    og_position = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    n_tets = 1
    n_particles = 4

    position = wp.array(og_position, dtype=wp.vec3, requires_grad=True)
    position_array2d = wp.array2d(og_position, dtype=wp.float32, requires_grad=True)
    stretch = wp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=vec6)
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.int32)

    cpu_rest = np.hstack(
        [
            og_position[1] - og_position[0],
            og_position[2] - og_position[0],
            og_position[3] - og_position[0],
        ]
    ).reshape(3, 3)
    rest = wp.array(np.linalg.inv(cpu_rest), dtype=wp.mat33)

    row_idx = wp.zeros(n_tets * 4, dtype=wp.int32)
    col_idx = wp.zeros(n_tets * 4, dtype=wp.int32)

    wp.launch(
        precompute_bsr_topology,
        dim=n_tets,
        inputs=[tets],
        outputs=[row_idx, col_idx],
    )

    values = wp.zeros(n_tets * 4, dtype=mat63)

    wp.launch(
        evaluate_constraint_gradient_dx,
        dim=n_tets,
        inputs=[position, tets, rest],
        outputs=[values],
    )

    # bsr_from_triplets expects int32, kernel writes uint32
    rows_int = wp.array(row_idx.numpy(), dtype=wp.int32)
    cols_int = wp.array(col_idx.numpy(), dtype=wp.int32)

    bsr = ws.bsr_from_triplets(
        n_tets, n_particles, rows_int, cols_int, values, prune_numerical_zeros=False
    )

    jacobians = wg.jacobian_fd(
        flattened_evaluate_constraint,
        inputs=[position_array2d, stretch, tets, rest, n_tets],
        dim=(n_tets,),
    )

    dense_bsr = bsr_to_dense(bsr)
    assert dense_bsr.numpy() == pytest.approx(jacobians[(0, 0)].numpy(), rel=1e-3)


@wp.kernel
def test_kernel(
    input: wp.array[vec6],
    output: wp.array[vec6],
):
    tid = wp.tid()
    output[tid] = input[tid] * 2.0


def test_autograd_jacobian():

    input = wp.zeros(4, dtype=wp.vec3, requires_grad=True)
    output = wp.zeros(4, dtype=wp.vec3, requires_grad=True)

    jacobians = wg.jacobian(
        test_kernel,
        inputs=[input],
        outputs=[output],
        dim=(4,),
    )
    print(jacobians[(0, 0)])
