import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
import warp as wp
import warp.autograd as wg

from src.mfem.types import mat69, mat99, vec6
from src.mfem.utils import (
    deformation_gradient,
    rotation_gradient,
    stretch_component,
    stretch_gradient,
    sym_mat33_to_vec6,
    unflatten,
)

from .utils import (
    CMAP,
    Example,
    bsr_to_dense,
    create_single_tet_example,
    flatten_array,
    reshape_array,
    zero_norm,
)


@wp.kernel
def helper_kernel_test_stretch_component(
    A: wp.array[wp.mat33], result: wp.array[wp.mat33]
):
    tid = wp.tid()
    result[tid] = stretch_component(A[tid])


@wp.kernel
def helper_kernel_test_rotation_gradient(
    A: wp.array[wp.mat33], result: wp.array[mat99]
):
    tid = wp.tid()
    dR_dF, U, sigma, V = rotation_gradient(A[tid])
    result[tid] = dR_dF


# @wp.kernel
# def helper_kernel_test_stretch_gradient(A: wp.array[wp.mat33], result: wp.array[mat99]):
#     tid = wp.tid()
#     result[tid] = stretch_gradient(A[tid])


@wp.kernel
def _stretch_gradient_fd_test_deformation_gradient_kernel(
    position: wp.array[wp.vec3],
    tets: wp.array2d[wp.int32],
    rest: wp.array[wp.mat33],
    result: wp.array[wp.mat33],
):
    tid = wp.tid()

    result[tid] = deformation_gradient(position, tets, rest, tid)


@wp.kernel
def _stretch_gradient_fd_test_stretch_gradient(
    def_grad: wp.array[wp.mat33],
    result: wp.array[mat69],
):
    tid = wp.tid()

    result[tid] = stretch_gradient(def_grad[tid])


@wp.kernel
def _stretch_gradient_fd_test_stretch(
    deformation_gradient: wp.array[wp.mat33],
    result: wp.array[vec6],
):
    tid = wp.tid()

    result[tid] = sym_mat33_to_vec6(stretch_component(deformation_gradient[tid]))


def _compute_stretch(
    deformation_gradient: wp.array[wp.float32],
) -> wp.array[wp.float32]:
    deformation_gradient = reshape_array(deformation_gradient, wp.mat33)

    stretch = wp.zeros(
        shape=(deformation_gradient.shape[0],), dtype=vec6, requires_grad=True
    )
    wp.launch(
        _stretch_gradient_fd_test_stretch,
        dim=deformation_gradient.shape[0],
        inputs=[deformation_gradient],
        outputs=[stretch],
    )

    return flatten_array(stretch)


def _make_deformed_example(deformation: np.ndarray) -> Example:
    base = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    position = wp.array(base @ deformation.T, dtype=wp.vec3)
    tets = wp.array2d([[0, 1, 2, 3]], dtype=wp.int32)
    return Example(position=position, tets=tets)


def test_stretch_gradient_fd():
    rng = np.random.default_rng(42)
    deformations = [np.eye(3)] + [
        rng.normal(size=(3, 3)) * 0.3 + np.eye(3) for _ in range(4)
    ]
    examples = [_make_deformed_example(d) for d in deformations]

    rows = []
    for example in examples:
        deformation_gradient = wp.zeros(shape=(1,), dtype=wp.mat33, requires_grad=True)
        wp.launch(
            _stretch_gradient_fd_test_deformation_gradient_kernel,
            dim=1,
            inputs=[example.position, example.tets, example.rest],
            outputs=[deformation_gradient],
        )

        stretch_gradient = wp.zeros(shape=example.n_tets, dtype=mat69)
        wp.launch(
            _stretch_gradient_fd_test_stretch_gradient,
            dim=1,
            inputs=[deformation_gradient],
            outputs=[stretch_gradient],
        )

        flat_def_gradient = flatten_array(deformation_gradient)
        jacobian = wg.jacobian_fd(
            _compute_stretch,
            inputs=[flat_def_gradient],
            input_output_mask=[(0, 0)],
        )

        fd = jacobian[(0, 0)].numpy()
        analytical = stretch_gradient.numpy()[0]
        rows.append((fd, analytical))

        assert analytical.shape == (6, 9)
        assert analytical == pytest.approx(fd, abs=2e-2)

    diffs = [fd - analytical for fd, analytical in rows]
    global_abs_max = max(max(np.abs(d).max(), 1e-8) for d in diffs)
    diff_norm = zero_norm(-global_abs_max, global_abs_max)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    for i, ((fd, analytical), diff) in enumerate(zip(rows, diffs)):
        vmin = min(fd.min(), analytical.min())
        vmax = max(fd.max(), analytical.max())

        row_axes = axes[i] if n > 1 else axes
        im0 = row_axes[0].imshow(
            fd, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[0].set_title(f"Sample {i}: FD Jacobian")
        fig.colorbar(im0, ax=row_axes[0])
        im1 = row_axes[1].imshow(
            analytical, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax)
        )
        row_axes[1].set_title(f"Sample {i}: Analytical stretch gradient")
        fig.colorbar(im1, ax=row_axes[1])
        im2 = row_axes[2].imshow(diff, aspect="auto", cmap=CMAP, norm=diff_norm)
        row_axes[2].set_title(f"Sample {i}: Difference (FD - analytical)")
        fig.colorbar(im2, ax=row_axes[2])

    plt.tight_layout()
    plt.savefig("stretch_gradient.png")
