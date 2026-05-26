import warp as wp


@wp.kernel
def test_kernel(out: wp.array2d[wp.int32]):
    tid = wp.tid()
    result = wp.vec3i(tid, tid, tid)
    out[tid, 0] = result.x
    out[tid, 1] = result.y
    out[tid, 2] = result.z


def main():
    a = wp.empty((10, 3), dtype=wp.int32)
    wp.launch(kernel=test_kernel, dim=a.shape[0], inputs=[a])
    print(a.numpy())

    print(a.shape)


if __name__ == "__main__":
    main()
