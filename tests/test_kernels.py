import numpy as np
import pytest
import scipy as sp
import scipy.spatial
import warp as wp

from mfem.kernels import evaluate_constraints
from mfem.utils import vec6

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
