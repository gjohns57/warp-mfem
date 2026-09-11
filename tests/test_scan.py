import numpy as np
import pytest
import warp as wp

from mfem.scan import TiledScan

MAX_LEN = 1 << 14


def test_exclusive_scan_matches_array_scan():
    scan = TiledScan(max_length=MAX_LEN, scalar_type=wp.int32, tile_size=64)

    for _ in range(20):
        length = np.random.randint(low=1, high=MAX_LEN)
        a_cpu = np.random.randint(low=0, high=10, size=length).astype(np.int32)
        a = wp.array(a_cpu, dtype=wp.int32)
        out = wp.zeros(length, dtype=wp.int32)

        scan.compute_exclusive_scan(a, out)

        expected = np.cumsum(a_cpu) - a_cpu
        np.testing.assert_array_equal(out.numpy(), expected)


def test_inclusive_scan_matches_array_scan():
    scan = TiledScan(max_length=MAX_LEN, scalar_type=wp.float32, tile_size=64)

    for _ in range(20):
        length = np.random.randint(low=1, high=MAX_LEN)
        a_cpu = np.random.uniform(size=length).astype(np.float32)
        a = wp.array(a_cpu, dtype=wp.float32)
        out = wp.zeros(length, dtype=wp.float32)

        scan.compute_inclusive_scan(a, out)

        expected = np.cumsum(a_cpu)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="requires a CUDA device")
def test_exclusive_scan_inside_capture_while():
    device = "cuda:0"
    length = MAX_LEN - 7
    scan = TiledScan(max_length=MAX_LEN, scalar_type=wp.int32, tile_size=64, device=device)

    a_cpu = np.random.randint(low=0, high=10, size=length).astype(np.int32)
    a = wp.array(a_cpu, dtype=wp.int32, device=device)
    out = wp.zeros(length, dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    cond = wp.zeros(1, dtype=wp.int32, device=device)

    @wp.kernel
    def bump_and_check(counter: wp.array(dtype=wp.int32), cond: wp.array(dtype=wp.int32)):
        counter[0] = counter[0] + 1
        cond[0] = wp.where(counter[0] < 3, 1, 0)

    def body():
        scan.compute_exclusive_scan(a, out)
        wp.launch(bump_and_check, dim=1, inputs=[counter], outputs=[cond], device=device)

    with wp.ScopedDevice(device):
        wp.launch(bump_and_check, dim=1, inputs=[counter], outputs=[cond])
        with wp.ScopedCapture() as capture:
            wp.capture_while(cond, while_body=body)

        wp.capture_launch(capture.graph)

    expected = np.cumsum(a_cpu) - a_cpu
    np.testing.assert_array_equal(out.numpy(), expected)
