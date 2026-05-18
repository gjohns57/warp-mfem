import re

import numpy as np
import pytest
import warp as wp

from mfem.utils import mat99, rotation_gradient, stretch_component, stretch_gradient
from simkit.simkit import polar_svd, stretch_gradient_dF
from simkit.simkit.rotation_gradient import rotation_gradient_F


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


@wp.kernel
def helper_kernel_test_stretch_gradient(A: wp.array[wp.mat33], result: wp.array[mat99]):
    tid = wp.tid()
    result[tid] = stretch_gradient(A[tid])


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


def test_stretch_gradient():
    rng = np.random.default_rng(0)
    tested = 0
    while tested < 20:
        F_cpu = rng.normal(size=(3, 3))
        _, s, _ = np.linalg.svd(F_cpu)
        if s[-1] < 0.3 or s[-2] - s[-1] < 0.3:
            continue

        F = wp.array(F_cpu, dtype=wp.mat33)
        result = wp.zeros(shape=(1,), dtype=mat99)
        wp.launch(helper_kernel_test_stretch_gradient, dim=1, inputs=[F], outputs=[result])

        simkit_result = stretch_gradient_dF(F_cpu.reshape((1, 3, 3))).reshape((3, 3, 3, 3))
        warp_4d = result.numpy()[0].reshape(3, 3, 3, 3)

        assert warp_4d == pytest.approx(simkit_result, abs=1e-3)
        tested += 1
