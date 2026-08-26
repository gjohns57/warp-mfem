import warp as wp


@wp.func
def capsule_sdf(point: wp.vec3, radius: wp.float32, half_height: wp.float32):
    """Signed distance, gradient, and Hessian from `point` to a capsule.

    The capsule is centered at the origin with its axis along Z, matching
    newton.geometry.sdf_capsule/sdf_capsule_grad called with up_axis=Axis.Z
    (the only axis used in this codebase). Combining them into one function
    avoids finding the closest point on the central axis twice, and adds the
    Hessian, which Newton doesn't provide.
    """
    eps = wp.float32(1.0e-8)

    # Closest point on the central axis segment, clamped to the two end caps.
    axis_z = wp.clamp(point[2], -half_height, half_height)
    diff = wp.vec3(point[0], point[1], point[2] - axis_z)
    rho = wp.length(diff)
    rho_safe = wp.max(rho, eps)

    n = wp.vec3(0.0, 0.0, 1.0)
    if rho > eps:
        n = diff / rho

    d = rho - radius

    identity = wp.identity(3, wp.float32)
    if point[2] > half_height or point[2] < -half_height:
        # End cap: closest axis point is fixed, so this is a plain
        # distance-to-point Hessian.
        hessian = (identity - wp.outer(n, n)) / rho_safe
    else:
        # Cylindrical section: the closest axis point tracks point[2] exactly,
        # so distance is independent of it -- zero that row/column out.
        ez = wp.vec3(0.0, 0.0, 1.0)
        hessian = (identity - wp.outer(ez, ez) - wp.outer(n, n)) / rho_safe

    return d, n, hessian
