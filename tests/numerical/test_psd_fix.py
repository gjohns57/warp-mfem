import numpy as np
import warp as wp

from mfem.numerical import psd_fix_3, psd_fix_6
from mfem.types import mat66


@wp.kernel
def _psd_fix_3_kernel(
    a: wp.array(dtype=wp.mat33),
    fixed: wp.array(dtype=wp.mat33),
    inv: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    fix = psd_fix_3(a[i])
    fixed[i] = fix.mat
    inv[i] = fix.inv


@wp.kernel
def _psd_fix_6_kernel(
    a: wp.array(dtype=mat66),
    fixed: wp.array(dtype=mat66),
    inv: wp.array(dtype=mat66),
):
    i = wp.tid()
    fix = psd_fix_6(a[i])
    fixed[i] = fix.mat
    inv[i] = fix.inv


def _run_3(A_np: np.ndarray):
    A_np = A_np.astype(np.float32)
    a = wp.array([wp.mat33(*A_np.flatten().tolist())], dtype=wp.mat33)
    fixed = wp.zeros_like(a)
    inv = wp.zeros_like(a)
    wp.launch(_psd_fix_3_kernel, dim=1, inputs=[a], outputs=[fixed, inv])
    return fixed.numpy()[0], inv.numpy()[0]


def _run_6(A_np: np.ndarray):
    A_np = A_np.astype(np.float32)
    a = wp.array([mat66(*A_np.flatten().tolist())], dtype=mat66)
    fixed = wp.zeros_like(a)
    inv = wp.zeros_like(a)
    wp.launch(_psd_fix_6_kernel, dim=1, inputs=[a], outputs=[fixed, inv])
    return fixed.numpy()[0], inv.numpy()[0]


# Already positive-definite 3x3 (diagonal dominant)
A3_pd = np.array(
    [
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 5.0],
    ],
    dtype=np.float32,
)
# Symmetric with negative eigenvalues
A3_non_pd = np.array(
    [
        [1.0, 3.0, 2.0],
        [3.0, 1.0, -4.0],
        [2.0, -4.0, 1.0],
    ],
    dtype=np.float32,
)

# Already positive-definite 6x6 (diagonal)
A6_pd = np.diag([4.0, 3.0, 5.0, 2.0, 6.0, 4.0]).astype(np.float32)

# Symmetric 6x6 with mixed eigenvalues
_rng = np.random.default_rng(0)
_Q = np.linalg.qr(_rng.standard_normal((6, 6)))[0].astype(np.float32)
_D = np.diag([3.0, 2.0, 5.0, -1.0, 1.0, -2.0]).astype(np.float32)
A6_non_pd = (lambda M: (M + M.T) / 2)(_Q @ _D @ _Q.T)


class TestPSDFix3:
    def test_pd_input_mat_matches_original(self):
        mat, _ = _run_3(A3_pd)
        np.testing.assert_allclose(mat, A3_pd, rtol=1e-4, atol=1e-5)

    def test_pd_input_inverse_correct(self):
        mat, inv = _run_3(A3_pd)
        np.testing.assert_allclose(mat @ inv, np.eye(3), atol=2e-5)

    def test_non_pd_output_is_pd(self):
        mat, _ = _run_3(A3_non_pd)
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(eigvals > 0), f"Expected positive eigenvalues, got {eigvals}"

    def test_non_pd_output_is_symmetric(self):
        mat, inv = _run_3(A3_non_pd)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        np.testing.assert_allclose(inv, inv.T, atol=1e-6)

    def test_non_pd_inverse_correct(self):
        mat, inv = _run_3(A3_non_pd)
        np.testing.assert_allclose(mat @ inv, np.eye(3), atol=1e-4)


class TestPSDFix6:
    def test_pd_input_mat_matches_original(self):
        mat, _ = _run_6(A6_pd)
        np.testing.assert_allclose(mat, A6_pd, rtol=1e-4, atol=1e-5)

    def test_pd_input_inverse_correct(self):
        mat, inv = _run_6(A6_pd)
        np.testing.assert_allclose(mat @ inv, np.eye(6), atol=2e-5)

    def test_non_pd_output_is_pd(self):
        mat, _ = _run_6(A6_non_pd)
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(eigvals > 0), f"Expected positive eigenvalues, got {eigvals}"

    def test_non_pd_output_is_symmetric(self):
        mat, inv = _run_6(A6_non_pd)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        np.testing.assert_allclose(inv, inv.T, atol=1e-6)

    def test_non_pd_inverse_correct(self):
        mat, inv = _run_6(A6_non_pd)
        np.testing.assert_allclose(mat @ inv, np.eye(6), atol=1e-4)
