# Tiled dot product code stolen from warp cg implementation I would just import it but they didn't think it neccessary to make it visible
#

import functools
from typing import Any

import warp as wp


def _as_scalar_array(x: wp.array):
    return x.view(wp.float32).flatten()


class TiledAccumulator:
    """Compute the dot product of two arrays in a way that is compatible with CUDA sub-graphs."""

    def __init__(
        self,
        max_length: int,
        scalar_type: type,
        tile_size=512,
        device=None,
        max_column_count: int = 1,
    ):
        self.tile_size = tile_size
        self.device = device
        self.max_column_count = max_column_count

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        scratch = wp.empty(
            shape=(2, max_column_count, num_blocks),
            dtype=scalar_type,
            device=self.device,
        )
        self.partial_sums_a = scratch[0]
        self.partial_sums_b = scratch[1]

        self.dot_kernel, self.sum_kernel = _create_tiled_dot_kernels(self.tile_size)

        rounds = 0
        length = num_blocks
        while length > 1:
            length = (length + self.tile_size - 1) // self.tile_size
            rounds += 1

        self.rounds = rounds
        self._output = self.partial_sums_a if rounds % 2 == 0 else self.partial_sums_b

        self.dot_launch: wp.Launch = wp.launch(
            self.dot_kernel,
            dim=(max_column_count, num_blocks, self.tile_size),
            inputs=(self.partial_sums_a, self.partial_sums_b),
            outputs=(self.partial_sums_a,),
            block_dim=self.tile_size,
            record_cmd=True,
        )
        self.sum_launch: wp.Launch = wp.launch(
            self.sum_kernel,
            dim=(max_column_count, num_blocks, self.tile_size),
            inputs=(self.partial_sums_a,),
            outputs=(self.partial_sums_b,),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute_dot(self, a: wp.array, b: wp.array, col_offset: int = 0):
        a = _as_scalar_array(a)
        b = _as_scalar_array(b)
        if a.ndim == 1:
            a = a.reshape((1, -1))
        if b.ndim == 1:
            b = b.reshape((1, -1))

        column_count = a.shape[0]
        num_blocks = (a.shape[1] + self.tile_size - 1) // self.tile_size

        data_out = self.partial_sums_a[col_offset : col_offset + column_count]
        data_in = self.partial_sums_b[col_offset : col_offset + column_count]

        self.dot_launch.set_param_at_index(0, a)
        self.dot_launch.set_param_at_index(1, b)
        self.dot_launch.set_param_at_index(2, data_out)
        self.dot_launch.set_dim((column_count, num_blocks, self.tile_size))
        self.dot_launch.launch()

        for _r in range(self.rounds):
            array_length = num_blocks
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            data_in, data_out = data_out, data_in

            self.sum_launch.set_param_at_index(0, data_in[:, :array_length])
            self.sum_launch.set_param_at_index(1, data_out)
            self.sum_launch.set_dim((column_count, num_blocks, self.tile_size))
            self.sum_launch.launch()

        return data_out

    def compute_sum(self, a: wp.array, col_offset: int = 0):
        a = _as_scalar_array(a)
        if a.ndim == 1:
            a = a.reshape((1, -1))

        column_count = a.shape[0]
        num_blocks = (a.shape[1] + self.tile_size - 1) // self.tile_size

        data_out = self.partial_sums_a[col_offset : col_offset + column_count]
        data_in = self.partial_sums_b[col_offset : col_offset + column_count]
        self.sum_launch.set_param_at_index(0, a)
        self.sum_launch.set_param_at_index(1, data_out)
        self.sum_launch.set_dim((column_count, num_blocks, self.tile_size))
        self.sum_launch.launch()

        for _r in range(self.rounds):
            array_length = num_blocks
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            data_in, data_out = data_out, data_in

            self.sum_launch.set_param_at_index(0, data_in[:, :array_length])
            self.sum_launch.set_param_at_index(1, data_out)
            self.sum_launch.set_dim((column_count, num_blocks, self.tile_size))
            self.sum_launch.launch()

        return data_out

    def col(self, col: int = 0):
        return self._output[col][:1]

    def result(self, col: int = 0):
        return self.col()

    def cols(self, count, start: int = 0):
        return self._output[start : start + count, :1]


@functools.cache
def _create_tiled_dot_kernels(tile_size):
    @wp.kernel
    def block_dot_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, _tid_block = wp.tid()

        start = block_id * tile_size

        a_block = wp.tile_load(a[column], shape=tile_size, offset=start)
        b_block = wp.tile_load(b[column], shape=tile_size, offset=start)
        t = wp.tile_map(wp.mul, a_block, b_block)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    @wp.kernel
    def block_sum_kernel(
        data: wp.array2d(dtype=Any),
        partial_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, _tid_block = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[column], shape=tile_size, offset=start)

        tile_sum = wp.tile_sum(t)
        wp.tile_store(partial_sums[column], tile_sum, offset=block_id)

    return block_dot_kernel, block_sum_kernel
