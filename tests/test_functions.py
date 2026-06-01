import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
import warp as wp
import warp.autograd as wg

from simkit.simkit import polar_svd, stretch_gradient_dF
from simkit.simkit.rotation_gradient import rotation_gradient_F
from simkit.simkit.stretch import stretch
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


def test_stretch_component():
    A = wp.array(
        [wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.mat33
    )
    result = wp.zeros_like(A)

    wp.launch(helper_kernel_test_stretch_component, dim=1, inputs=[A], outputs=[result])
    assert result.numpy()[0] == pytest.approx(
        np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)).reshape((3, 3)),
        abs=1e-6,
    )

    for _ in range(20):
        A_cpu = np.random.normal(size=(3, 3))
        A = wp.array(A_cpu, dtype=wp.mat33)
        result = wp.zeros_like(A)
        wp.launch(
            helper_kernel_test_stretch_component, dim=3, inputs=[A], outputs=[result]
        )

        rot, check = polar_svd(A_cpu.reshape((1, 3, 3)))

        # This is not great tolerance, maybe the random matrices are not well-conditioned for this
        assert result.numpy()[0] == pytest.approx(check.reshape((3, 3)), abs=1e-4)


def test_rotation_gradient():
    rng = np.random.default_rng(0)
    tested = 0
    while tested < 20:
        F_cpu = rng.normal(size=(3, 3))
        _, s, _ = np.linalg.svd(F_cpu)
        # Skip ill-conditioned matrices: small s[-1] causes float32 SVD inaccuracy;
        # small s[-2]-s[-1] means s12 = s[1]-s[2] is near zero after the sign flip
        # in polar SVD, amplifying float32 errors by a factor of 2/s12.
        if s[-1] < 0.3 or s[-2] - s[-1] < 0.3:
            continue

        F = wp.array(F_cpu, dtype=wp.mat33)
        result = wp.zeros(shape=(1,), dtype=mat99)
        wp.launch(
            helper_kernel_test_rotation_gradient, dim=1, inputs=[F], outputs=[result]
        )

        simkit_result = rotation_gradient_F(F_cpu.reshape((1, 3, 3)))  # (1, 9, 9)

        # warp flatten is col-major: index col*3+row -> mat[row,col]
        # simkit reshape is row-major: index row*3+col -> mat[row,col]
        # so warp_4d.transpose(1,0,3,2) matches simkit_4d
        warp_4d = result.numpy()[0].reshape(3, 3, 3, 3)
        simkit_4d = simkit_result[0].reshape(3, 3, 3, 3)
        assert warp_4d.transpose(1, 0, 3, 2) == pytest.approx(simkit_4d, abs=1e-3)
        tested += 1


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
        im0 = row_axes[0].imshow(fd, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax))
        row_axes[0].set_title(f"Sample {i}: FD Jacobian")
        fig.colorbar(im0, ax=row_axes[0])
        im1 = row_axes[1].imshow(analytical, aspect="auto", cmap=CMAP, norm=zero_norm(vmin, vmax))
        row_axes[1].set_title(f"Sample {i}: Analytical stretch gradient")
        fig.colorbar(im1, ax=row_axes[1])
        im2 = row_axes[2].imshow(diff, aspect="auto", cmap=CMAP, norm=diff_norm)
        row_axes[2].set_title(f"Sample {i}: Difference (FD - analytical)")
        fig.colorbar(im2, ax=row_axes[2])

    plt.tight_layout()
    plt.savefig("stretch_gradient.png")
