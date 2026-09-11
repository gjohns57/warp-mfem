import pytest
import warp as wp

from mfem.numerical import gmw81_3


@wp.kernel
def _test_cholesky_kernel(
    A: wp.array(dtype=wp.mat33),
    L: wp.array(dtype=wp.mat33),
):
    L[0] = gmw81_3(A[0])


def _test_cholesky():

    A = wp.array(
        [wp.mat33(2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0)], dtype=wp.mat33
    )
    L = wp.zeros_like(A)
    wp.launch(
        _test_cholesky_kernel,
        dim=1,
        inputs=[A, L],
    )
    Lcpu = L.numpy()[0]

    print(Lcpu @ Lcpu.T)
    assert False
