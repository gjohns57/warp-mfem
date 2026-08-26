# Tiled prefix-sum (scan) built the same way as TiledAccumulator in accumulate.py:
# every kernel launch is pre-recorded with record_cmd=True and replayed by
# mutating its params/dims, so no host-side allocation happens on the hot path.
#
# wp.utils.array_scan() allocates CUB temporary storage internally on every call,
# which is illegal inside a conditional CUDA (sub-)graph (wp.capture_if /
# wp.capture_while bodies must not allocate memory). This module trades that
# convenience for a fixed set of preallocated scratch buffers so the scan can be
# recorded inside a conditional capture.
#
# Only scalar float32/int32/uint32 dtypes are supported (the same restriction
# wp.tile_scan_inclusive/exclusive impose) -- unlike wp.utils.array_scan, vector
# dtypes are not handled here.

import functools
from typing import Any

import warp as wp


class TiledScan:
    """Compute an inclusive or exclusive prefix sum (scan) of an array in a way
    that is compatible with CUDA sub-graphs, including conditional ones
    (wp.capture_if / wp.capture_while), unlike wp.utils.array_scan.
    """

    def __init__(
        self,
        max_length: int,
        scalar_type: type,
        tile_size: int = 512,
        device=None,
        max_column_count: int = 1,
    ):
        self.tile_size = tile_size
        self.device = device
        self.max_column_count = max_column_count

        # sizes[i] is the (worst-case) length of the array being scanned at
        # level i: level 0 is the caller's array, level i>0 is the block sums
        # of level i-1. The last level always fits in a single tile, so it
        # needs no further reduction.
        sizes = [max_length]
        while sizes[-1] > tile_size:
            sizes.append((sizes[-1] + tile_size - 1) // tile_size)
        self.sizes = sizes
        self.num_levels = len(sizes)

        num_blocks0 = (max_length + tile_size - 1) // tile_size

        # One scratch buffer per level >= 1, plus a trailing size-1 dummy that
        # only exists to catch the (unused) block-sums output of the top
        # level. Buffer `level` doubles as: the block-sums output when the
        # kernel runs on level `level`'s data, and (in place) the local
        # exclusive scan / final result of level `level + 1`'s data once the
        # kernel runs on it.
        level_sizes = sizes[1:] + [1]
        self._levels = [
            wp.empty(shape=(max_column_count, sz), dtype=scalar_type, device=device) for sz in level_sizes
        ]

        self.scan_kernel, self.broadcast_kernel, self.add_kernel = _create_tiled_scan_kernels(tile_size)

        placeholder = self._levels[0]
        self.scan_launch: wp.Launch = wp.launch(
            self.scan_kernel,
            dim=(max_column_count, num_blocks0, tile_size),
            inputs=(placeholder,),
            outputs=(placeholder, placeholder),
            block_dim=tile_size,
            record_cmd=True,
        )
        self.broadcast_launch: wp.Launch = wp.launch(
            self.broadcast_kernel,
            dim=(max_column_count, max_length),
            inputs=(placeholder, placeholder),
            outputs=(placeholder,),
            record_cmd=True,
        )
        self.add_launch: wp.Launch = wp.launch(
            self.add_kernel,
            dim=(max_column_count, max_length),
            inputs=(placeholder, placeholder),
            outputs=(placeholder,),
            record_cmd=True,
        )

    def _level_buffer(self, level: int, column_count: int, col_offset: int, out: wp.array):
        if level == 0:
            return out
        return self._levels[level - 1][col_offset : col_offset + column_count]

    def compute_exclusive_scan(self, a: wp.array, out: wp.array, col_offset: int = 0) -> wp.array:
        a2 = a.reshape((1, -1)) if a.ndim == 1 else a
        out2 = out.reshape((1, -1)) if out.ndim == 1 else out

        column_count = a2.shape[0]
        length = a2.shape[1]

        if length == 0:
            return out

        lengths = [length]
        data_in = a2
        local_out = out2

        for level in range(self.num_levels):
            num_blocks = (length + self.tile_size - 1) // self.tile_size
            block_sums_out = self._levels[level][col_offset : col_offset + column_count, :num_blocks]

            self.scan_launch.set_param_at_index(0, data_in)
            self.scan_launch.set_param_at_index(1, local_out)
            self.scan_launch.set_param_at_index(2, block_sums_out)
            self.scan_launch.set_dim((column_count, num_blocks, self.tile_size))
            self.scan_launch.launch()

            lengths.append(num_blocks)
            data_in = block_sums_out
            local_out = block_sums_out
            length = num_blocks

        # Downsweep: broadcast each level's block offsets down into the level
        # below it, from the (already-correct, single-block) top level down
        # to the caller's own array.
        for level in range(self.num_levels - 2, -1, -1):
            local_excl = self._level_buffer(level, column_count, col_offset, out2)[:, : lengths[level]]
            block_offsets = self._level_buffer(level + 1, column_count, col_offset, out2)[:, : lengths[level + 1]]

            self.broadcast_launch.set_param_at_index(0, local_excl)
            self.broadcast_launch.set_param_at_index(1, block_offsets)
            self.broadcast_launch.set_param_at_index(2, local_excl)
            self.broadcast_launch.set_dim((column_count, lengths[level]))
            self.broadcast_launch.launch()

        return out

    def compute_inclusive_scan(self, a: wp.array, out: wp.array, col_offset: int = 0) -> wp.array:
        a2 = a.reshape((1, -1)) if a.ndim == 1 else a
        out2 = out.reshape((1, -1)) if out.ndim == 1 else out

        self.compute_exclusive_scan(a, out, col_offset=col_offset)

        if a2.shape[1] == 0:
            return out

        self.add_launch.set_param_at_index(0, out2)
        self.add_launch.set_param_at_index(1, a2)
        self.add_launch.set_param_at_index(2, out2)
        self.add_launch.set_dim((a2.shape[0], a2.shape[1]))
        self.add_launch.launch()

        return out


@functools.cache
def _create_tiled_scan_kernels(tile_size):
    @wp.kernel
    def block_scan_kernel(
        data: wp.array2d(dtype=Any),
        local_excl: wp.array2d(dtype=Any),
        block_sums: wp.array2d(dtype=Any),
    ):
        column, block_id, _tid_block = wp.tid()

        start = block_id * tile_size

        t = wp.tile_load(data[column], shape=tile_size, offset=start)
        excl = wp.tile_scan_exclusive(t)
        wp.tile_store(local_excl[column], excl, offset=start)

        total = wp.tile_sum(t)
        wp.tile_store(block_sums[column], total, offset=block_id)

    @wp.kernel
    def broadcast_add_kernel(
        local_excl: wp.array2d(dtype=Any),
        block_offsets: wp.array2d(dtype=Any),
        out: wp.array2d(dtype=Any),
    ):
        column, i = wp.tid()
        block_id = i // tile_size
        out[column, i] = local_excl[column, i] + block_offsets[column, block_id]

    @wp.kernel
    def elementwise_add_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array2d(dtype=Any),
        out: wp.array2d(dtype=Any),
    ):
        column, i = wp.tid()
        out[column, i] = a[column, i] + b[column, i]

    return block_scan_kernel, broadcast_add_kernel, elementwise_add_kernel
