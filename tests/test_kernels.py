import numpy as np
import pytest
import scipy.spatial
import warp.sparse as ws
import warp as wp

from mfem.kernels import elastic_gradient_dx, evaluate_constraints
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
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.uint32)

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


def test_elastic_gradient_dx():
    # Unit tet: x0 at origin, edges along coordinate axes
    og_position = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    n_tets = 1
    n_particles = 4

    position = wp.array(og_position, dtype=wp.vec3)
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.uint32)

    cpu_rest = np.hstack(
        [
            og_position[1] - og_position[0],
            og_position[2] - og_position[0],
            og_position[3] - og_position[0],
        ]
    ).reshape(3, 3)
    rest = wp.array(np.linalg.inv(cpu_rest), dtype=wp.mat33)

    row_idx = wp.zeros(n_tets * 4, dtype=wp.uint32)
    col_idx = wp.zeros(n_tets * 4, dtype=wp.uint32)
    values = wp.zeros(n_tets * 4, dtype=mat63)

    wp.launch(
        elastic_gradient_dx,
        dim=n_tets,
        inputs=[position, tets, rest],
        outputs=[row_idx, col_idx, values],
    )

    # bsr_from_triplets expects int32, kernel writes uint32
    rows_int = wp.array(row_idx.numpy(), dtype=wp.int32)
    cols_int = wp.array(col_idx.numpy(), dtype=wp.int32)

    bsr = ws.bsr_from_triplets(
        n_tets, n_particles, rows_int, cols_int, values, prune_numerical_zeros=False
    )

    # Assemble dense view from BSR internals for printing
    br, bc = bsr.block_shape
    offsets = bsr.offsets.numpy()
    columns = bsr.columns.numpy()
    blocks = bsr.values.numpy()  # (nnz, 6, 3)

    dense = np.zeros(bsr.shape)
    for row in range(bsr.nrow):
        for k in range(offsets[row], offsets[row + 1]):
            dense[row * br : (row + 1) * br, columns[k] * bc : (columns[k] + 1) * bc] = blocks[k]

    print("\nElastic gradient dS/dX (6 x 12):")
    print(dense)
