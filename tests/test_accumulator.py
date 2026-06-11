import numpy as np
import pytest
import warp as wp

from mfem.accumulate import TiledAccumulator


def test_sum():
    accumulator = TiledAccumulator(max_length=(1 << 21), scalar_type=wp.float32)

    for _ in range(20):
        a_cpu = np.random.uniform(size=np.random.randint(low=0, high=((1 << 21) - 1)))
        a = wp.array(a_cpu, dtype=wp.float32)

        accumulator.compute_sum(a)
        suma = accumulator.col()
        # assert False
        assert suma.numpy() == pytest.approx(a.numpy().sum())


def test_dot():

    accumulator = TiledAccumulator(max_length=(1 << 21), scalar_type=wp.float32)
    for _ in range(20):
        len = np.random.randint(low=0, high=((1 << 21) - 1))
        a_cpu = np.random.uniform(size=len)
        b_cpu = np.random.uniform(size=len)
        a = wp.array(a_cpu, dtype=wp.float32)
        b = wp.array(b_cpu, dtype=wp.float32)

        accumulator.compute_dot(a, b)
        suma = accumulator.col()
        # assert False
        assert suma.numpy() == pytest.approx(np.dot(a.numpy(), b.numpy()))
