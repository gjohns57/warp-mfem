import warp as wp


class mat23(wp.types.matrix(shape=(2, 3), dtype=wp.float32)):
    """2 by 3 matrix"""


@wp.func
def point_triangle_distance_type(p: wp.vec3, t0: wp.vec3, t1: wp.vec3, t2: wp.vec3):
    normal = wp.cross(t1 - t0, t2 - t0)

    e01 = t1 - t0
    n01 = wp.cross(e01, normal)

    pass
