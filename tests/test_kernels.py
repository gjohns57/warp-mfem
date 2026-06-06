import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.spatial
import warp as wp
import warp.autograd as wg
import warp.sparse as ws

from mfem.kernels import (
    evaluate_constraint_gradient_dx,
    evaluate_constraints,
    precompute_bsr_topology,
)
from mfem.types import mat63, vec6

from .utils import (
    CMAP,
    Example,
    bsr_to_dense,
    create_cube_example,
    create_single_tet_example,
    flatten_array,
    reshape_array,
    zero_norm,
)

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
    n_particles: int,
) -> wp.array2d[wp.float32]:

    constraint = wp.empty(n_tets, dtype=vec6, requires_grad=True)
    position_vec = reshape_array(position, wp.vec3)

    wp.launch(
        evaluate_constraints,
        dim=n_tets,
        inputs=[position_vec, stretch, tets, rest],
        outputs=[constraint],
    )

    return flatten_array(constraint)


def test_constraint_gradient_dx_visual():
    rng = np.random.default_rng(42)
    deformations = [np.eye(3)] + [
        rng.normal(size=(3, 3)) * 0.3 + np.eye(3) for _ in range(4)
    ]

    rows = []
    for deformation in deformations:
        example = create_single_tet_example(deformation)
        n_tets = example.n_tets
        n_particles = example.n_particles

        row_idx = wp.zeros(n_tets * 4, dtype=wp.int32)
        col_idx = wp.zeros(n_tets * 4, dtype=wp.int32)
        wp.launch(
            precompute_bsr_topology,
            dim=n_tets,
            inputs=[example.tets],
            outputs=[row_idx, col_idx],
        )

        values = wp.zeros(n_tets * 4, dtype=mat63)
        wp.launch(
            evaluate_constraint_gradient_dx,
            dim=n_tets,
            inputs=[example.position, example.tets, example.rest],
            outputs=[values],
        )

        bsr = ws.bsr_from_triplets(
            n_tets, n_particles, row_idx, col_idx, values, prune_numerical_zeros=False
        )
        analytical = bsr_to_dense(bsr).numpy()

        jacobians = wg.jacobian_fd(
            flattened_evaluate_constraint,
            inputs=[
                flatten_array(example.position),
                example.stretch,
                example.tets,
                example.rest,
                n_tets,
                n_particles,
            ],
            dim=(n_tets,),
        )
        fd = jacobians[(0, 0)].numpy()

        rows.append((analytical, fd))

    diffs = [a - f for a, f in rows]
    global_abs_max = max(max(np.abs(d).max(), 1e-8) for d in diffs)
    diff_norm = zero_norm(-global_abs_max, global_abs_max)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    for i, ((analytical, fd), diff) in enumerate(zip(rows, diffs)):
        vmin = min(analytical.min(), fd.min())
        vmax = max(analytical.max(), fd.max())

        row_axes = axes[i] if n > 1 else axes
        im0 = row_axes[0].imshow(
            analytical, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[0].set_title(f"Sample {i}: Analytical gradient")
        fig.colorbar(im0, ax=row_axes[0])
        im1 = row_axes[1].imshow(
            fd, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[1].set_title(f"Sample {i}: FD Jacobian")
        fig.colorbar(im1, ax=row_axes[1])
        im2 = row_axes[2].imshow(diff, aspect="auto", cmap=CMAP, norm=diff_norm)
        row_axes[2].set_title(f"Sample {i}: Difference (analytical - FD)")
        fig.colorbar(im2, ax=row_axes[2])

    plt.tight_layout()
    plt.savefig("constraint_gradient_dx.png")

    for analytical, fd in rows:
        assert analytical == pytest.approx(fd, abs=5e-2)


def test_cube_constraint_gradient_dx_visual():
    rng = np.random.default_rng(42)
    deformations = [np.eye(3)] + [
        rng.normal(size=(3, 3)) * 0.3 + np.eye(3) for _ in range(4)
    ]

    rows = []
    for deformation in deformations:
        example = create_cube_example(deformation)
        n_tets = example.n_tets
        n_particles = example.n_particles

        row_idx = wp.zeros(n_tets * 4, dtype=wp.int32)
        col_idx = wp.zeros(n_tets * 4, dtype=wp.int32)
        wp.launch(
            precompute_bsr_topology,
            dim=n_tets,
            inputs=[example.tets],
            outputs=[row_idx, col_idx],
        )

        values = wp.zeros(n_tets * 4, dtype=mat63)
        wp.launch(
            evaluate_constraint_gradient_dx,
            dim=n_tets,
            inputs=[example.position, example.tets, example.rest],
            outputs=[values],
        )

        bsr = ws.bsr_from_triplets(
            n_tets, n_particles, row_idx, col_idx, values, prune_numerical_zeros=False
        )
        analytical = bsr_to_dense(bsr).numpy()
        jacobians = wg.jacobian_fd(
            flattened_evaluate_constraint,
            inputs=[
                flatten_array(example.position),
                example.stretch,
                example.tets,
                example.rest,
                n_tets,
                n_particles,
            ],
        )
        fd = jacobians[(0, 0)].numpy()

        rows.append((analytical, fd))

    diffs = [a - f for a, f in rows]
    global_abs_max = max(max(np.abs(d).max(), 1e-8) for d in diffs)
    diff_norm = zero_norm(-global_abs_max, global_abs_max)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    for i, ((analytical, fd), diff) in enumerate(zip(rows, diffs)):
        vmin = min(analytical.min(), 0.0)
        vmax = max(analytical.max(), 0.0)

        row_axes = axes[i] if n > 1 else axes
        im0 = row_axes[0].imshow(
            analytical, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[0].set_title(f"Sample {i}: Analytical gradient")
        fig.colorbar(im0, ax=row_axes[0])
        im1 = row_axes[1].imshow(
            fd, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[1].set_title(f"Sample {i}: FD Jacobian")
        fig.colorbar(im1, ax=row_axes[1])
        im2 = row_axes[2].imshow(diff, aspect="auto", cmap=CMAP, norm=diff_norm)
        row_axes[2].set_title(f"Sample {i}: Difference (analytical - FD)")
        fig.colorbar(im2, ax=row_axes[2])

    plt.tight_layout()
    plt.savefig("cube_constraint_gradient_dx.png")

    for i, (analytical, fd) in enumerate(rows):
        assert analytical == pytest.approx(fd, abs=1e-2)
