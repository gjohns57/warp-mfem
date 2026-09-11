from newton import State, Model, GeoType
from mfem.refinement.additional_state import AdditionalState
from mfem.ipc.distance import capsule_sdf, plane_sdf
from mfem.collision import closest_point_segment_segment
import numpy as np
import warp as wp
import warp.sparse as ws



# We can see if it is cost effective to build a array that stores indices for surface points instead of just checking
# all points. If we end up building a surface tri mesh at each step we could at the same time build a surface points
# indices array


@wp.func
def barrier(d: wp.float32, d0: wp.float32, d1: wp.float32, b_d0: wp.float32, db_d0 : wp.float32, d2b_d0: wp.float32) -> tuple[wp.float32, wp.float32, wp.float32]:
    energy = wp.float32(0.0)
    d_energy = wp.float32(0.0)
    d2_energy = wp.float32(0.0)
    if d < d1 and d > d0:
        log_d_d_1 = wp.log(d / d1)
        energy = -(d - d1) * (d - d1) * log_d_d_1
        d_energy = -(2.0 * (d - d1) * log_d_d_1 + (d - d1) * (d - d1) / d)
        d2_energy = -(2.0 * log_d_d_1 + 4.0 * (d - d1) / d - (d - d1) * (d - d1) / (d * d))
    elif d <= d0:
        energy = b_d0 + db_d0 * (d - d0) + 0.5 *  d2b_d0 * (d - d0) * (d - d0)
        d_energy = db_d0 + d2b_d0 * (d - d0)
        d2_energy = d2b_d0

    return energy, d_energy, d2_energy


@wp.func
def friction_f0(y: wp.float32, ev: wp.float32) -> wp.float32:
    """Smoothed friction dissipation potential (IPC, Li et al. 2020).

    ``f0`` is the antiderivative of the mollifier ``f1``: below the sliding
    threshold ``ev = eps_v * dt`` it ramps quadratically (so the tangential
    force goes smoothly to zero at zero sliding -- static friction), above it
    ``f0(y) = y`` which yields the constant Coulomb force ``mu * lambda_n``.
    """
    if y >= ev:
        return y
    return -y * y * y / (3.0 * ev * ev) + y * y / ev + ev / 3.0


@wp.func
def friction_f1_over_y(y: wp.float32, ev: wp.float32) -> wp.float32:
    """``f1(y) / y`` with a finite ``y -> 0`` limit of ``2 / ev``.

    ``f1`` is the derivative of :func:`friction_f0`. Dividing by ``y`` keeps the
    quantity bounded at zero sliding (that is what makes the lagged friction
    gradient/Hessian well defined for a resting contact), and it is >= 0
    everywhere so ``(f1/y) * (I - n n^T)`` is a PSD Hessian block.
    """
    if y >= ev:
        return 1.0 / wp.max(y, wp.float32(1.0e-12))
    return -y / (ev * ev) + 2.0 / ev


@wp.func
def query_min_signed_distance(
    x: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
) -> wp.float32:
    """Find the minimum signed distance from `x` to the rigid bodies (distance only, no gradient/Hessian)."""
    d_min = wp.float32(1.0e8)

    for shape in range(shape_count):
        body = shape_body[shape]
        if body >= 0:
            body_transform = body_q[body]
        else:
            body_transform = wp.transform_identity()
        shape_world_transform = wp.transform_multiply(body_transform, shape_transform[shape])
        x_local = wp.transform_point(wp.transform_inverse(shape_world_transform), x)

        if shape_type[shape] == GeoType.CAPSULE:
            scale = shape_scale[shape]
            d, _n_local, _hess_local = capsule_sdf(x_local, scale[0], scale[1])
        elif shape_type[shape] == GeoType.PLANE:
            d, _n_local, _hess_local = plane_sdf(x_local)
        else:
            continue

        if d < d_min:
            d_min = d

    return d_min


@wp.kernel
def compute_distance(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    distance: wp.array[wp.float32],
    distance_gradient: wp.array[wp.vec3],
    distance_hessian: wp.array[wp.mat33],
    shape_id: wp.array[wp.int32],
):
    """Find the particle's minimum signed distance to the rigid bodies, along with its gradient and Hessian.

    ``shape_id`` records which shape realized the minimum (or -1 if there were
    none), so a later pass can look up a per-shape quantity -- e.g. a Coulomb
    friction coefficient that differs between the tool and the ground.
    """
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    x = particle_q[tid]

    d_min = wp.float32(1.0e8)
    grad_min = wp.vec3(0.0)
    hess_min = wp.mat33(0.0)
    id_min = wp.int32(-1)

    for shape in range(shape_count):
        # Transform x into shape local frame
        body = shape_body[shape]
        if body >= 0:
            body_transform = body_q[body]
        else:
            # Static shapes (e.g. the ground plane) have no parent body.
            body_transform = wp.transform_identity()
        shape_world_transform = wp.transform_multiply(body_transform, shape_transform[shape])
        x_local = wp.transform_point(wp.transform_inverse(shape_world_transform), x)

        if shape_type[shape] == GeoType.CAPSULE:
            scale = shape_scale[shape]
            d, n_local, hess_local = capsule_sdf(x_local, scale[0], scale[1])
        elif shape_type[shape] == GeoType.PLANE:
            d, n_local, hess_local = plane_sdf(x_local)
        else:
            continue

        # Gradient direction transforms as a normal (rotation only, no translation)
        grad_d = wp.transform_vector(shape_world_transform, n_local)

        # Hessian transforms as a bilinear form under the shape's rotation: R H R^T
        rotation = wp.quat_to_matrix(wp.transform_get_rotation(shape_world_transform))
        hess_d = rotation * hess_local * wp.transpose(rotation)

        if d < d_min:
            d_min = d
            grad_min = grad_d
            hess_min = hess_d
            id_min = shape

    distance[tid] = d_min
    distance_gradient[tid] = grad_min
    distance_hessian[tid] = hess_min
    shape_id[tid] = id_min


@wp.kernel
def compute_distance_only(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    distance: wp.array[wp.float32],
):
    """Recompute only the particle's minimum signed distance, without its gradient/Hessian. Used in the line search."""
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    x = particle_q[tid]

    distance[tid] = query_min_signed_distance(
        x, shape_transform, shape_type, shape_scale, shape_body, shape_count, body_q
    )


@wp.kernel
def evaluate_barrier_energy(
    active_particle_count: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    distance_gradient: wp.array[wp.vec3],
    distance_hessian: wp.array[wp.mat33],
    stiffness: wp.array[wp.float32],
    d0: wp.float32, # The distance closer to which (or if the signed distance is negative) we use a quadratic extrapolation of the log barrier
    d1: wp.float32, # The distance after which the barrier energy is 0
    quadratic_barrier_coefficients: wp.vec3,
    barrier_energy: wp.array[wp.float32],
    barrier_gradient: wp.array[wp.vec3],
    barrier_hessian: wp.array[wp.mat33],
):
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    k = stiffness[0]
    d = distance[tid]

    energy = wp.float32(0.0)
    gradient = wp.vec3(0.0)
    hessian = wp.mat33(0.0)


    if d < d1:
        grad_d = distance_gradient[tid]

        e, de, d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)
        energy = e
        gradient = de * grad_d
        # Gauss-Newton / PSD-projected contact Hessian: keep only the
        # d2e * grad_d grad_d^T term (rank-1, PSD once d2e is clamped >= 0) and
        # drop the de * hess_d curvature term. For a convex obstacle approached
        # from outside de < 0 while hess_d (e.g. the capsule's (I - nn^T)/rho) is
        # PSD and nonzero, so de * hess_d is negative-definite with magnitude
        # ~ k*|de|/rho -- that blows up as rho -> 0 near the capsule axis and
        # makes the assembled global matrix indefinite, which the plain-CG global
        # solve cannot handle. A plane has hess_d == 0 so this changes nothing there.
        hessian = wp.max(d2e, wp.float32(0.0)) * wp.outer(grad_d, grad_d)

    barrier_energy[tid] = k * energy
    barrier_gradient[tid] = k * gradient
    barrier_hessian[tid] = k * hessian


@wp.kernel
def evaluate_barrier_energy_only(
    active_particle_count: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    d0: wp.float32,
    d1: wp.float32,
    quadratic_barrier_coefficients: wp.vec3,
    barrier_energy: wp.array[wp.float32],
):
    """Recompute only the barrier energy (no gradient/Hessian). Used in the line search."""
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    k = stiffness[0]
    d = distance[tid]

    energy = wp.float32(0.0)
    if d < d1:
        e, de, d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)
        energy = e

    barrier_energy[tid] = k * energy


@wp.kernel
def compute_lagged_normal_force(
    active_particle_count: wp.array[wp.int32],
    distance: wp.array[wp.float32],
    distance_gradient: wp.array[wp.vec3],
    shape_id: wp.array[wp.int32],
    stiffness: wp.array[wp.float32],
    d0: wp.float32,
    d1: wp.float32,
    quadratic_barrier_coefficients: wp.vec3,
    normal_lagged: wp.array[wp.vec3],
    normal_force_lagged: wp.array[wp.float32],
    shape_id_lagged: wp.array[wp.int32],
):
    """Snapshot the normal contact force magnitude and direction at the
    start-of-step configuration. These are held fixed for the whole step and
    drive the (lagged) friction potential -- see :func:`evaluate_friction`.

    Also snapshots which shape realized the contact (``shape_id_lagged``), so
    friction can use that shape's own Coulomb coefficient (e.g. the tool vs.
    the ground) rather than one coefficient shared by every contact.
    """
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return
    k = stiffness[0]
    d = distance[tid]

    lam = wp.float32(0.0)
    n = wp.vec3(0.0)
    sid = wp.int32(-1)
    if d < d1:
        _e, de, _d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)
        # -k * b'(d) is the (outward, >= 0) normal force the barrier applies.
        lam = k * wp.max(-de, wp.float32(0.0))
        n = distance_gradient[tid]
        sid = shape_id[tid]

    normal_lagged[tid] = n
    normal_force_lagged[tid] = lam
    shape_id_lagged[tid] = sid


@wp.kernel
def evaluate_friction(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    x_start: wp.array[wp.vec3],
    normal_lagged: wp.array[wp.vec3],
    normal_force_lagged: wp.array[wp.float32],
    shape_id_lagged: wp.array[wp.int32],
    shape_friction_mu: wp.array[wp.float32],
    epsv_dt: wp.float32,  # eps_v * dt: tangential sliding below which friction is static
    friction_energy: wp.array[wp.float32],
    friction_gradient: wp.array[wp.vec3],
    friction_hessian: wp.array[wp.mat33],
):
    """Lagged semi-implicit friction: with the normal force ``lambda_n`` and
    the contact normal ``n`` frozen at their start-of-step values, the friction
    dissipation is a smooth potential in the tangential sliding displacement
    ``u_T = (x - x_start) - ((x - x_start) . n) n``. The Coulomb coefficient is
    looked up per-shape (``shape_id_lagged``, snapshotted alongside the normal
    force at the start of the step) so different rigid bodies -- e.g. the tool
    and the ground -- can carry different friction."""
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return

    lam = normal_force_lagged[tid]
    sid = shape_id_lagged[tid]

    energy = wp.float32(0.0)
    gradient = wp.vec3(0.0)
    hessian = wp.mat33(0.0)

    if lam > 0.0 and sid >= 0:
        mu = shape_friction_mu[sid]
        if mu > 0.0:
            n = normal_lagged[tid]
            u = particle_q[tid] - x_start[tid]
            u_t = u - wp.dot(u, n) * n
            y = wp.length(u_t)

            scale = mu * lam
            f1_over_y = friction_f1_over_y(y, epsv_dt)

            energy = scale * friction_f0(y, epsv_dt)
            gradient = scale * f1_over_y * u_t
            proj = wp.identity(3, wp.float32) - wp.outer(n, n)
            hessian = scale * f1_over_y * proj

    friction_energy[tid] = energy
    friction_gradient[tid] = gradient
    friction_hessian[tid] = hessian


@wp.kernel
def evaluate_friction_energy_only(
    active_particle_count: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    x_start: wp.array[wp.vec3],
    normal_lagged: wp.array[wp.vec3],
    normal_force_lagged: wp.array[wp.float32],
    shape_id_lagged: wp.array[wp.int32],
    shape_friction_mu: wp.array[wp.float32],
    epsv_dt: wp.float32,
    friction_energy: wp.array[wp.float32],
):
    """Recompute only the friction dissipation energy. Used in the line search."""
    tid = wp.tid()
    if tid >= active_particle_count[0]:
        return

    lam = normal_force_lagged[tid]
    sid = shape_id_lagged[tid]

    energy = wp.float32(0.0)
    if lam > 0.0 and sid >= 0:
        mu = shape_friction_mu[sid]
        if mu > 0.0:
            n = normal_lagged[tid]
            u = particle_q[tid] - x_start[tid]
            u_t = u - wp.dot(u, n) * n
            y = wp.length(u_t)
            energy = mu * lam * friction_f0(y, epsv_dt)

    friction_energy[tid] = energy

# ---------------------------------------------------------------------------
# Triangle-level contact against the capsule tool
#
# The per-particle barrier above only ever sees the SDF at the mesh vertices,
# so a capsule narrower than the surface edges can slip between them and
# submerge with no resisting force. The kernels below add one extra barrier
# term per active surface tri (AdditionalState.tri_indices), evaluated at the
# closest point of the tri to the capsule *axis segment*. For a convex tool
# that point realizes the minimum of the capsule SDF over the whole tri, so
# the barrier "sees" edges and face interiors, not only vertices.
#
# Writing the closest point as p = w0 x0 + w1 x1 + w2 x2 with the barycentric
# weights w frozen, everything reduces to the chain rule through a linear map:
#   dE/dx_i      = w_i * b'(d) * n
#   d2E/dx_i dx_j = w_i w_j * b''(d) * n n^T      (Gauss-Newton, PSD)
# The gradient is exact by the envelope theorem (the closest point minimizes
# over w); the Hessian drops the closest-point / SDF curvature terms exactly as
# the per-particle Hessian above already does.
#
# Only capsule shapes take part: a plane is convex and unbounded, so its SDF
# minimum over a tri always sits at a vertex, which the per-particle term
# already handles. Tris sharing an edge (or a vertex) that both pick the same
# closest point double-count that point; that is deliberate -- it keeps the
# energy continuous as the closest point crosses between face, edge and vertex
# regions, and a barrier's stiffness is not a physical quantity.
# ---------------------------------------------------------------------------


@wp.func
def closest_point_triangle_bary(x: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Barycentric weights (w_a, w_b, w_c) of the closest point on tri abc to x
    (Ericson, "Real-Time Collision Detection" 5.1.5)."""
    ab = b - a
    ac = c - a
    ax = x - a
    d1 = wp.dot(ab, ax)
    d2 = wp.dot(ac, ax)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bx = x - b
    d3 = wp.dot(ab, bx)
    d4 = wp.dot(ac, bx)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    cx = x - c
    d5 = wp.dot(ab, cx)
    d6 = wp.dot(ac, cx)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)

    total = va + vb + vc
    if total <= 0.0:
        # Degenerate (collinear) tri that slipped past the region tests.
        return wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    denom = 1.0 / total
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def segment_triangle_intersect(
    p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3
) -> tuple[wp.int32, wp.vec3]:
    """Moller-Trumbore segment/triangle test. Returns (hit, barycentric weights
    of the hit point on abc); hit == 0 means no intersection (weights zero)."""
    direction = p1 - p0
    e1 = b - a
    e2 = c - a
    pv = wp.cross(direction, e2)
    det = wp.dot(e1, pv)
    # Relative parallel test: det ~ |dir| |e1| |e2| for a general hit.
    scale = wp.length(direction) * wp.length(e1) * wp.length(e2)
    if wp.abs(det) <= 1.0e-7 * scale + 1.0e-30:
        return wp.int32(0), wp.vec3(0.0)
    inv_det = 1.0 / det
    tv = p0 - a
    u = wp.dot(tv, pv) * inv_det
    if u < 0.0 or u > 1.0:
        return wp.int32(0), wp.vec3(0.0)
    qv = wp.cross(tv, e1)
    v = wp.dot(direction, qv) * inv_det
    if v < 0.0 or u + v > 1.0:
        return wp.int32(0), wp.vec3(0.0)
    t = wp.dot(e2, qv) * inv_det
    if t < 0.0 or t > 1.0:
        return wp.int32(0), wp.vec3(0.0)
    return wp.int32(1), wp.vec3(1.0 - u - v, u, v)


@wp.func
def closest_point_triangle_segment(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, p0: wp.vec3, p1: wp.vec3
) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.int32]:
    """Closest points between tri abc and segment p0p1.

    Returns (barycentric weights of the point on the tri, the point on the
    segment, their distance, hit) where hit == 1 means the segment pierces the
    tri -- in which case the distance is 0 and the tri point is the piercing
    point. The minimum is taken over the five candidates that can realize a
    tri/segment distance: each segment endpoint vs. the tri (covers face
    interiors, and edges/vertices as a side effect) and each tri edge vs. the
    segment. Nothing here needs derivatives: they come from the envelope
    theorem in the caller."""
    best_bary = closest_point_triangle_bary(p0, a, b, c)
    best_q = p0
    best_d = wp.length(best_bary[0] * a + best_bary[1] * b + best_bary[2] * c - p0)

    bary = closest_point_triangle_bary(p1, a, b, c)
    d = wp.length(bary[0] * a + bary[1] * b + bary[2] * c - p1)
    if d < best_d:
        best_d = d
        best_bary = bary
        best_q = p1

    # Tri edges vs. the segment. closest_point_segment_segment returns the
    # parameter along its *first* segment, so the tri edge goes first.
    c1, c2, s = closest_point_segment_segment(a, b, p0, p1)
    d = wp.length(c1 - c2)
    if d < best_d:
        best_d = d
        best_bary = wp.vec3(1.0 - s, s, 0.0)
        best_q = c2

    c1, c2, s = closest_point_segment_segment(b, c, p0, p1)
    d = wp.length(c1 - c2)
    if d < best_d:
        best_d = d
        best_bary = wp.vec3(0.0, 1.0 - s, s)
        best_q = c2

    c1, c2, s = closest_point_segment_segment(c, a, p0, p1)
    d = wp.length(c1 - c2)
    if d < best_d:
        best_d = d
        best_bary = wp.vec3(s, 0.0, 1.0 - s)
        best_q = c2

    hit, hit_bary = segment_triangle_intersect(p0, p1, a, b, c)
    if hit == 1:
        best_d = wp.float32(0.0)
        best_bary = hit_bary
        best_q = hit_bary[0] * a + hit_bary[1] * b + hit_bary[2] * c

    return best_bary, best_q, best_d, hit


@wp.func
def tri_capsule_contact(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
) -> tuple[wp.float32, wp.vec3, wp.vec3, wp.int32]:
    """Minimum signed distance from tri abc to any capsule shape, with the
    barycentric weights of the closest tri point, the unit world-space
    direction the distance grows in (the SDF gradient at that point) and the
    shape index (-1 if there are no capsules)."""
    d_min = wp.float32(1.0e8)
    bary_min = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    n_min = wp.vec3(0.0)
    id_min = wp.int32(-1)

    for shape in range(shape_count):
        if shape_type[shape] != GeoType.CAPSULE:
            continue
        body = shape_body[shape]
        if body >= 0:
            body_transform = body_q[body]
        else:
            body_transform = wp.transform_identity()
        X = wp.transform_multiply(body_transform, shape_transform[shape])
        scale = shape_scale[shape]
        radius = scale[0]
        half_height = scale[1]
        # Capsule axis segment in world space (local axis is +Z, see capsule_sdf).
        p0 = wp.transform_point(X, wp.vec3(0.0, 0.0, -half_height))
        p1 = wp.transform_point(X, wp.vec3(0.0, 0.0, half_height))

        bary, q, dist, hit = closest_point_triangle_segment(a, b, c, p0, p1)
        d = dist - radius
        if d < d_min:
            p = bary[0] * a + bary[1] * b + bary[2] * c
            if hit == 0 and dist > 1.0e-8:
                n = (p - q) / dist
            else:
                # The axis pierces the tri (or the closest tri point sits on
                # the axis), where the capsule SDF gradient is undefined. Push
                # along the tri normal, oriented toward the nearer axis end so
                # the tri leaves by the shorter route. The energy (b(-radius),
                # the barrier's deepest value) is what matters here: it makes
                # the line search reject any step that tunnels the axis through
                # a face.
                n = wp.cross(b - a, c - a)
                n_len = wp.length(n)
                if n_len > 1.0e-20:
                    n = n / n_len
                else:
                    n = wp.vec3(0.0, 0.0, 1.0)
                axis = p1 - p0
                t = wp.dot(p - p0, axis) / wp.max(wp.dot(axis, axis), 1.0e-20)
                toward = axis
                if t < 0.5:
                    toward = -axis
                if wp.dot(n, toward) < 0.0:
                    n = -n
            d_min = d
            bary_min = bary
            n_min = n
            id_min = shape

    return d_min, bary_min, n_min, id_min


@wp.kernel
def compute_tri_distance(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    tri_distance: wp.array[wp.float32],
    tri_bary: wp.array[wp.vec3],
    tri_normal: wp.array[wp.vec3],
    tri_shape_id: wp.array[wp.int32],
):
    """Per active surface tri: signed distance to the nearest capsule, closest
    point weights and the world-space SDF gradient there."""
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        return
    a = particle_q[tri_indices[tid, 0]]
    b = particle_q[tri_indices[tid, 1]]
    c = particle_q[tri_indices[tid, 2]]
    d, bary, n, sid = tri_capsule_contact(
        a, b, c, shape_transform, shape_type, shape_scale, shape_body, shape_count, body_q
    )
    tri_distance[tid] = d
    tri_bary[tid] = bary
    tri_normal[tid] = n
    tri_shape_id[tid] = sid


@wp.kernel
def compute_tri_distance_only(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_count: wp.int32,
    body_q: wp.array[wp.transform],
    tri_distance: wp.array[wp.float32],
):
    """Line-search variant of compute_tri_distance: distance only."""
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        return
    a = particle_q[tri_indices[tid, 0]]
    b = particle_q[tri_indices[tid, 1]]
    c = particle_q[tri_indices[tid, 2]]
    d, _bary, _n, _sid = tri_capsule_contact(
        a, b, c, shape_transform, shape_type, shape_scale, shape_body, shape_count, body_q
    )
    tri_distance[tid] = d


@wp.kernel
def evaluate_tri_barrier_energy(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_distance: wp.array[wp.float32],
    tri_bary: wp.array[wp.vec3],
    tri_normal: wp.array[wp.vec3],
    stiffness: wp.array[wp.float32],
    d0: wp.float32,
    d1: wp.float32,
    quadratic_barrier_coefficients: wp.vec3,
    tri_barrier_energy: wp.array[wp.float32],
    barrier_energy: wp.array[wp.float32],
    barrier_gradient: wp.array[wp.vec3],
    hessian_rows: wp.array[wp.int32],
    hessian_cols: wp.array[wp.int32],
    hessian_values: wp.array[wp.mat33],
):
    """Barrier energy/gradient/Hessian of one tri contact, scattered to the
    tri's vertices with its barycentric weights.

    The energy is accumulated into the *per-particle* ``barrier_energy`` (as
    well as recorded per tri) so the solver's objective sum and the refinement
    vertex scores pick it up with no extra plumbing; the gradient is
    accumulated into the per-particle ``barrier_gradient`` for the same
    reason. Both must therefore run *after* the per-particle barrier kernel,
    which assigns those arrays. The 9 Hessian blocks go out as BSR triplets.
    """
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        # Inactive slots must be zero: the tests / diagnostics sum the array.
        tri_barrier_energy[tid] = 0.0
        return
    k = stiffness[0]
    d = tri_distance[tid]

    energy = wp.float32(0.0)
    de = wp.float32(0.0)
    d2e = wp.float32(0.0)
    if d < d1:
        energy, de, d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)

    w = tri_bary[tid]
    n = tri_normal[tid]
    energy *= k
    g = k * de * n
    h = k * wp.max(d2e, wp.float32(0.0)) * wp.outer(n, n)

    tri_barrier_energy[tid] = energy
    for i in range(3):
        vi = tri_indices[tid, i]
        wp.atomic_add(barrier_energy, vi, w[i] * energy)
        wp.atomic_add(barrier_gradient, vi, w[i] * g)
        for j in range(3):
            slot = tid * 9 + i * 3 + j
            hessian_rows[slot] = vi
            hessian_cols[slot] = tri_indices[tid, j]
            hessian_values[slot] = (w[i] * w[j]) * h


@wp.kernel
def evaluate_tri_barrier_energy_only(
    active_tri_count: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_distance: wp.array[wp.float32],
    tri_bary: wp.array[wp.vec3],
    stiffness: wp.array[wp.float32],
    d0: wp.float32,
    d1: wp.float32,
    quadratic_barrier_coefficients: wp.vec3,
    tri_barrier_energy: wp.array[wp.float32],
    barrier_energy: wp.array[wp.float32],
):
    """Line-search variant: tri barrier energy only, scattered to the
    per-particle ``barrier_energy`` (run after the per-particle kernel)."""
    tid = wp.tid()
    if tid >= active_tri_count[0]:
        tri_barrier_energy[tid] = 0.0
        return
    k = stiffness[0]
    d = tri_distance[tid]

    energy = wp.float32(0.0)
    if d < d1:
        energy, _de, _d2e = barrier(d, d0, d1, *quadratic_barrier_coefficients)
    energy *= k

    w = tri_bary[tid]
    tri_barrier_energy[tid] = energy
    for i in range(3):
        wp.atomic_add(barrier_energy, tri_indices[tid, i], w[i] * energy)



class Contact:
    def __init__(
        self,
        model: Model,
        max_particles: int,
        d0: float,
        d1: float,
        stiffness: float,
        friction_mu: float | list[float] | np.ndarray = 0.0,
        friction_eps_v: float = 1.0e-3,
        max_tris: int = 0,
        max_incident_tris: int = 0,
        tri_contact: bool = True,
    ):
        self.distance = wp.zeros(max_particles, dtype=wp.float32)
        self.distance_gradient = wp.zeros(max_particles, dtype=wp.vec3)
        self.distance_hessian = wp.zeros(max_particles, dtype=wp.mat33)
        self.shape_id = wp.zeros(max_particles, dtype=wp.int32)
        self.barrier_energy = wp.zeros(max_particles, dtype=wp.float32)
        self.barrier_gradient = wp.zeros(max_particles, dtype=wp.vec3)
        self.barrier_hessian_blocks = wp.zeros(max_particles, dtype=wp.mat33)
        self._quadratic_barrier_coefficients = wp.vec3(*barrier(d0, 0.0, d1, 0.0, 0.0, 0.0))
        self._stiffness = wp.array([stiffness], dtype=wp.float32)
        self.d0 = d0
        self.d1 = d1

        # ---- Lagged (semi-implicit) friction ------------------------------
        # mu: Coulomb coefficient, either one scalar shared by every rigid
        # shape or a per-shape sequence indexed the same way shapes were added
        # to the ModelBuilder (e.g. [tool_mu, ground_mu]) so different bodies
        # -- the poking tool vs. the table -- can be tuned independently.
        # eps_v: sliding *velocity* below which the contact is treated as
        # static; it is turned into a per-step sliding *distance* threshold
        # eps_v * dt inside evaluate()/evaluate_energy_only().
        self.shape_count = int(model.shape_count)
        mu = np.asarray(friction_mu, dtype=np.float32)
        if mu.ndim == 0:
            mu = np.full(self.shape_count, float(mu), dtype=np.float32)
        elif mu.shape[0] != self.shape_count:
            raise ValueError(
                f"friction_mu has {mu.shape[0]} entries but the model has "
                f"{self.shape_count} shapes"
            )
        self.friction_mu = mu  # per-shape, numpy (host-side "any friction?" checks)
        self._shape_friction_mu = wp.array(mu, dtype=wp.float32)
        self.friction_eps_v = float(friction_eps_v)
        # Start-of-step positions and the normal force / normal / contact
        # shape frozen there.
        self.x_start = wp.zeros(max_particles, dtype=wp.vec3)
        self.normal_lagged = wp.zeros(max_particles, dtype=wp.vec3)
        self.normal_force_lagged = wp.zeros(max_particles, dtype=wp.float32)
        self.shape_id_lagged = wp.zeros(max_particles, dtype=wp.int32)
        self.friction_energy = wp.zeros(max_particles, dtype=wp.float32)
        self.friction_gradient = wp.zeros(max_particles, dtype=wp.vec3)
        self.friction_hessian_blocks = wp.zeros(max_particles, dtype=wp.mat33)

        # ---- Triangle-level contact (see compute_tri_distance) -------------
        # One extra barrier term per active surface tri, evaluated at the
        # tri's closest point to the capsule axis, so the tool cannot slip
        # between vertices. Its energy/gradient are accumulated into the
        # per-particle barrier_energy / barrier_gradient arrays above; its
        # Hessian is a separate BSR matrix over the tris' (i, j) vertex pairs
        # (all of which already exist in the global lhs sparsity, so the
        # solver's padded bsr_axpy never grows topology). Disabled when the
        # caller has no surface tris (max_tris == 0) or asks for vertex-only
        # contact (tri_contact=False).
        self.tri_contact = bool(tri_contact) and max_tris > 0
        self.max_tris = int(max_tris)
        if self.tri_contact:
            self.tri_distance = wp.zeros(max_tris, dtype=wp.float32)
            self.tri_bary = wp.zeros(max_tris, dtype=wp.vec3)
            self.tri_normal = wp.zeros(max_tris, dtype=wp.vec3)
            self.tri_shape_id = wp.zeros(max_tris, dtype=wp.int32)
            self.tri_barrier_energy = wp.zeros(max_tris, dtype=wp.float32)
            self._tri_hessian_rows = wp.zeros(max_tris * 9, dtype=wp.int32)
            self._tri_hessian_cols = wp.zeros(max_tris * 9, dtype=wp.int32)
            self._tri_hessian_values = wp.zeros(max_tris * 9, dtype=wp.mat33)
            self._tri_hessian_block_count = wp.zeros(1, dtype=wp.int32)
            self.tri_barrier_hessian = ws.bsr_zeros(
                max_particles,
                max_particles,
                wp.mat33,
                topology="padded",
                row_capacity=3 * max(1, int(max_incident_tris)),
            )

    def _tri_shape_inputs(self, model: Model, state: State, additional_state: AdditionalState) -> list:
        return [
            additional_state.active_tri_count,
            additional_state.tri_indices,
            state.particle_q,
            model.shape_transform,
            model.shape_type,
            model.shape_scale,
            model.shape_body,
            model.shape_count,
            state.body_q,
        ]

    def compute_distance(self, model: Model, state: State, additional_state: AdditionalState):
        wp.launch(
            compute_distance,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_body,
                model.shape_count,
                state.body_q,
            ],
            outputs=[
                self.distance,
                self.distance_gradient,
                self.distance_hessian,
                self.shape_id,
            ],
        )
        if self.tri_contact:
            wp.launch(
                compute_tri_distance,
                dim=self.max_tris,
                inputs=self._tri_shape_inputs(model, state, additional_state),
                outputs=[
                    self.tri_distance,
                    self.tri_bary,
                    self.tri_normal,
                    self.tri_shape_id,
                ],
            )

    def compute_distance_only(self, model: Model, state: State, additional_state: AdditionalState):
        wp.launch(
            compute_distance_only,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_body,
                model.shape_count,
                state.body_q,
            ],
            outputs=[
                self.distance,
            ],
        )
        if self.tri_contact:
            wp.launch(
                compute_tri_distance_only,
                dim=self.max_tris,
                inputs=self._tri_shape_inputs(model, state, additional_state),
                outputs=[self.tri_distance],
            )

    def begin_step(self, model: Model, state: State, additional_state: AdditionalState):
        """Snapshot the start-of-step state that lagged friction needs: the
        particle positions (to measure this step's tangential sliding) and the
        normal contact force / normal frozen at that configuration. Call once
        per step, before the Newton iterations."""
        wp.copy(self.x_start, state.particle_q)

        # No rigid shapes (e.g. the unit-test grid) -> no contact, no friction.
        if self.shape_count == 0 or float(self.friction_mu.max()) <= 0.0:
            return

        self.compute_distance(model, state, additional_state)
        wp.launch(
            compute_lagged_normal_force,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                self.distance,
                self.distance_gradient,
                self.shape_id,
                self._stiffness,
                self.d0,
                self.d1,
                self._quadratic_barrier_coefficients,
            ],
            outputs=[
                self.normal_lagged,
                self.normal_force_lagged,
                self.shape_id_lagged,
            ],
        )

    def evaluate(self, model: Model, state: State, additional_state: AdditionalState, dt: float):
        """Recompute the distance and the full barrier + friction energy (energy, gradient, Hessian)."""
        self.compute_distance(model, state, additional_state)

        wp.launch(
            evaluate_barrier_energy,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                self.distance,
                self.distance_gradient,
                self.distance_hessian,
                self._stiffness,
                self.d0,
                self.d1,
                self._quadratic_barrier_coefficients,
            ],
            outputs=[
                self.barrier_energy,
                self.barrier_gradient,
                self.barrier_hessian_blocks,
            ]
        )

        if self.tri_contact:
            # Must follow evaluate_barrier_energy: it *accumulates* into the
            # per-particle barrier_energy / barrier_gradient that kernel assigns.
            wp.launch(
                evaluate_tri_barrier_energy,
                dim=self.max_tris,
                inputs=[
                    additional_state.active_tri_count,
                    additional_state.tri_indices,
                    self.tri_distance,
                    self.tri_bary,
                    self.tri_normal,
                    self._stiffness,
                    self.d0,
                    self.d1,
                    self._quadratic_barrier_coefficients,
                ],
                outputs=[
                    self.tri_barrier_energy,
                    self.barrier_energy,
                    self.barrier_gradient,
                    self._tri_hessian_rows,
                    self._tri_hessian_cols,
                    self._tri_hessian_values,
                ],
            )
            wp.copy(self._tri_hessian_block_count, additional_state.active_tri_count)
            self._tri_hessian_block_count *= 9
            ws.bsr_set_from_triplets(
                self.tri_barrier_hessian,
                self._tri_hessian_rows,
                self._tri_hessian_cols,
                self._tri_hessian_values,
                count=self._tri_hessian_block_count,
                topology="padded",
            )

        wp.launch(
            evaluate_friction,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                self.x_start,
                self.normal_lagged,
                self.normal_force_lagged,
                self.shape_id_lagged,
                self._shape_friction_mu,
                self.friction_eps_v * dt,
            ],
            outputs=[
                self.friction_energy,
                self.friction_gradient,
                self.friction_hessian_blocks,
            ],
        )

    def evaluate_energy_only(self, model: Model, state: State, additional_state: AdditionalState, dt: float):
        """Recompute the distance and the barrier + friction energy only (no gradient/Hessian). Used in the line search."""
        self.compute_distance_only(model, state, additional_state)

        wp.launch(
            evaluate_barrier_energy_only,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                self.distance,
                self._stiffness,
                self.d0,
                self.d1,
                self._quadratic_barrier_coefficients,
            ],
            outputs=[
                self.barrier_energy,
            ]
        )

        if self.tri_contact:
            wp.launch(
                evaluate_tri_barrier_energy_only,
                dim=self.max_tris,
                inputs=[
                    additional_state.active_tri_count,
                    additional_state.tri_indices,
                    self.tri_distance,
                    self.tri_bary,
                    self._stiffness,
                    self.d0,
                    self.d1,
                    self._quadratic_barrier_coefficients,
                ],
                outputs=[
                    self.tri_barrier_energy,
                    self.barrier_energy,
                ],
            )

        wp.launch(
            evaluate_friction_energy_only,
            dim=model.particle_count,
            inputs=[
                additional_state.active_particle_count,
                state.particle_q,
                self.x_start,
                self.normal_lagged,
                self.normal_force_lagged,
                self.shape_id_lagged,
                self._shape_friction_mu,
                self.friction_eps_v * dt,
            ],
            outputs=[
                self.friction_energy,
            ],
        )
