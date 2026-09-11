import newton
import numpy as np
import warp as wp
import warp.sparse as ws
from newton import Contacts, Mesh, Model
from newton._src.sim import contacts
from newton.geometry import BroadPhaseExplicit, BroadPhaseSAP, NarrowPhase
from newton.viewer import ViewerUSD
from nvtx import nvtx

# @wp.func
# def swept_aabb(
#     tri: wp.array[wp.vec3], x: wp.array[wp.vec3], dx: wp.array[wp.vec3]
# ) -> tuple[wp.vec3, wp.vec3]:

#     x0 = x[tri[0]]
#     x1 = x[tri[1]]
#     x2 = x[tri[2]]
#     px0 = x0 + dx[tri[0]]
#     px1 = x1 + dx[tri[1]]
#     px2 = x2 + dx[tri[2]]

#     return (
#         wp.min(wp.min(x0, px0), wp.min(wp.min(x1, px1), wp.min(x2, px2))),
#         wp.max(wp.max(x0, px0), wp.max(wp.max(x1, px1), wp.max(x2, px2))),
#     )


# @wp.kernel
# def compute_swept_aabbs(
#     tri_indices: wp.array2d[wp.int32],
#     x: wp.array[wp.vec3],
#     dx: wp.array[wp.vec3],
#     lower: wp.array[wp.vec3],
#     upper: wp.array[wp.vec3],
# ):
#     tid = wp.tid()

#     lower[tid], upper[tid] = swept_aabb(tri_indices[tid], x, dx)


# @wp.kernel
# def find_tool_tissue_collisions(
#     tool_tri_indices: wp.array2d[wp.int32],
#     collideable_tris: wp.array[wp.int32],
#     tool_x: wp.array[wp.vec3],
#     dx: wp.array[wp.vec3],
#     bvh: wp.uint64,
#     tool: wp.int32,
#     tissue: wp.int32,
# ):
#     lower, upper = swept_aabb(tool_tri_indices, x, dx)

#     wp.bvh_query_aabb(
#         bvh,
#         lower,
#         upper,

#     )
#     pass


# def CollsionEnergy:
#     def __init__()


# @wp.kernel
# def eval_shape_boundary(
#     x: wp.array[wp.vec3],
#     contact_count: wp.array[wp.int32],
#     contact_node: wp.array[wp.int32],
#     contact_shape: wp.array[wp.int32],
#     contact_normal: wp.array[wp.vec3],
#     energy: wp.array[wp.float32],
#     gradient: wp.array[wp.vec3],
#     hessian: wp.array[wp.mat33],
# ):

#     pass
#

# For some reason Newton hides its geometry types enum
MESH_TYPE = 8


# Build bvh (one for triangles and one for edges? points?) initially from primatives (consider using different roots for tool and tissue in case not interested in self collsion)
#
# for each time step:
#   potentially rebuild bvh
#
#   for each iteration:
#     for each node using last step's swept BVH query nearby primatives to evaluate boundary energy, gradient, and hessian
#
#
#     global solve
#     refit bvh based on dx to build swept AABBs
#     CCD line search
#       query point swept AABBs against shape AABBs
#     local solve
#     backtracking line search


def refit_swept_bvh(
    model: Model, bvh: wp.Bvh, x: wp.array[wp.vec3], dx: wp.array[wp.vec3]
):

    return


@wp.kernel
def compute_mesh_world_space_vertices_kernel(
    body_transfrom: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    vertices: wp.array[wp.vec3],
):
    pass


# def compute_mesh_world_space_vertices(mesh: newton.Mesh, body_transform: wp.transform) -> wp.array[wp.vec3]:
#     ws_vertices = wp.array(mesh.vertices, dtype=wp.vec3)
#     transform =


@wp.kernel
def compute_static_rigid_aabbs(
    points: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    point_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_transform: wp.array[wp.transform],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tid = wp.tid()
    shape = point_shape[tri_indices[tid, 0]]
    body = shape_body[shape]
    transform = wp.transform_multiply(body_transform[body], shape_transform[shape])

    x0 = wp.transform_point(transform, points[tri_indices[tid, 0]])
    x1 = wp.transform_point(transform, points[tri_indices[tid, 1]])
    x2 = wp.transform_point(transform, points[tri_indices[tid, 2]])

    lower[tid] = wp.min(wp.min(x0, x1), x2)
    upper[tid] = wp.max(wp.max(x0, x1), x2)


@wp.kernel
def compute_aabbs(
    points: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    tid = wp.tid()
    x0 = points[tri_indices[tid, 0]]
    x1 = points[tri_indices[tid, 1]]
    x2 = points[tri_indices[tid, 2]]

    lower[tid] = wp.min(wp.min(x0, x1), x2)
    upper[tid] = wp.max(wp.max(x0, x1), x2)


@wp.kernel
def transform_points_to_world(
    points_local: wp.array[wp.vec3],
    point_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_transform: wp.array[wp.transform],
    points_world: wp.array[wp.vec3],
):
    tid = wp.tid()
    shape = point_shape[tid]
    transform = shape_transform[shape]

    body = shape_body[shape]
    if body >= 0:
        transform = wp.transform_multiply(body_transform[body], transform)

    points_world[tid] = wp.transform_point(transform, points_local[tid])


@wp.kernel
def compute_rigid_tri_collision_groups(
    tri_indices: wp.array2d[wp.int32],
    point_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    collision_group: wp.array[wp.int32],
):
    # Negative collision groups collide with everything except triangles
    # sharing the same negative id, so every body (including the implicit
    # "world" body used by shapes with no parent) gets its own id here.
    # Offsetting by 2 keeps the id away from 0, which the broad phase treats
    # as "never collides".
    tid = wp.tid()
    shape = point_shape[tri_indices[tid, 0]]
    body = shape_body[shape]
    collision_group[tid] = -(body + 2)


# @wp.kernel
# def soft_surface_aabbs(
#     particle_q: wp.array[wp.vec3],
#     surface_tris: wp.array2d[wp.int32],
#     surface_indices: wp.array[wp.int32],
#     surface_particle_count: wp.array[wp.int32],
#     lower: wp.array[wp.vec3],
#     upper: wp.array[wp.vec3],
# ):
#     tid = wp.tid()


def get_surface_indices(tri_indices: wp.array[wp.int32]) -> wp.array[wp.int32]:
    return wp.array(
        np.unique(tri_indices.numpy().flatten(), sorted=True), dtype=wp.int32
    )


@wp.func
def closest_point_on_triangle(
    x: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3
) -> wp.vec3:
    ab = b - a
    ac = c - a
    ax = x - a

    d1 = wp.dot(ab, ax)
    d2 = wp.dot(ac, ax)

    # Vertex A
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bx = x - b
    d3 = wp.dot(ab, bx)
    d4 = wp.dot(ac, bx)

    # Vertex B
    if d3 >= 0.0 and d4 <= d3:
        return b

    cx = x - c
    d5 = wp.dot(ab, cx)
    d6 = wp.dot(ac, cx)

    # Vertex C
    if d6 >= 0.0 and d5 <= d6:
        return c

    # Edge AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab

    # Edge AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac

    # Edge BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)

    # Interior
    denom = 1.0 / (va + vb + vc)
    return a + (vb * denom) * ab + (vc * denom) * ac


@wp.func
def point_triangle_dist(x: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    return wp.length(x - closest_point_on_triangle(x, a, b, c))


@wp.func
def closest_point_segment_segment(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    # Ericson, "Real-Time Collision Detection" 5.1.9 - robust to the
    # degenerate/near-parallel case (denom ~ 0) that a naive linear solve
    # would divide by zero on.
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)

    epsilon = 1.0e-8

    s = wp.float32(0.0)
    t = wp.float32(0.0)

    if a <= epsilon and e <= epsilon:
        s = 0.0
        t = 0.0
    elif a <= epsilon:
        s = 0.0
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= epsilon:
            t = 0.0
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom > epsilon:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)

    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return c1, c2, s


@wp.func
def min_point_triangle_distance(
    x: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    particle: wp.int32,
    d_hat: wp.float32,
    distance: wp.array[wp.float32],
):
    d = point_triangle_dist(x, a, b, c)
    if d < d_hat:
        wp.atomic_min(distance, particle, d)


@wp.func
def min_triangle_point_distance(
    x: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    pa: wp.int32,
    pb: wp.int32,
    pc: wp.int32,
    d_hat: wp.float32,
    distance: wp.array[wp.float32],
):
    # Same distance as min_point_triangle_distance with x/triangle swapped;
    # the rigid point is close to the whole soft face, so every vertex of
    # that face is a candidate for taking this as its minimum.
    d = point_triangle_dist(x, a, b, c)
    if d < d_hat:
        wp.atomic_min(distance, pa, d)
        wp.atomic_min(distance, pb, d)
        wp.atomic_min(distance, pc, d)


@wp.func
def min_edge_edge_distance(
    sa: wp.vec3,
    sb: wp.vec3,
    ra: wp.vec3,
    rb: wp.vec3,
    pa: wp.int32,
    pb: wp.int32,
    d_hat: wp.float32,
    distance: wp.array[wp.float32],
):
    c1, c2, _s = closest_point_segment_segment(sa, sb, ra, rb)
    d = wp.length(c1 - c2)
    if d < d_hat:
        wp.atomic_min(distance, pa, d)
        wp.atomic_min(distance, pb, d)


@wp.kernel
def eval_soft_rigid_distance(
    points: wp.array[wp.vec3],
    tris: wp.array2d[wp.int32],
    rigid_points_count: wp.int32,
    rigid_tri_count: wp.int32,
    candidate_pairs: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    d_hat: wp.float32,
    distance: wp.array[wp.float32],
):
    # `distance` must be pre-filled with a sentinel >= d_hat (e.g. d_hat
    # itself) before this launches - atomic_min only ever lowers it, it
    # never resets it, and pairs/features farther than d_hat never write.
    tid = wp.tid()
    if tid >= candidate_pair_count[0]:
        return

    pair = candidate_pairs[tid]

    tri_a = pair[0]
    tri_b = pair[1]
    a_is_rigid = tri_a < rigid_tri_count
    b_is_rigid = tri_b < rigid_tri_count

    # Only handle rigid/soft pairs here; same-side pairs (two rigid bodies,
    # or - impossible today since there's one soft body group - two soft
    # triangles) belong to a different kernel.
    if a_is_rigid == b_is_rigid:
        return

    tri_rigid = tri_a
    tri_soft = tri_b
    if not a_is_rigid:
        tri_rigid = tri_b
        tri_soft = tri_a

    r0 = points[tris[tri_rigid, 0]]
    r1 = points[tris[tri_rigid, 1]]
    r2 = points[tris[tri_rigid, 2]]

    si0 = tris[tri_soft, 0]
    si1 = tris[tri_soft, 1]
    si2 = tris[tri_soft, 2]

    s0 = points[si0]
    s1 = points[si1]
    s2 = points[si2]

    p0 = si0 - rigid_points_count
    p1 = si1 - rigid_points_count
    p2 = si2 - rigid_points_count

    # For each point we want to keep track of the smallest distance

    # For each soft point
    #     get rigid triangle barycentric coordinates
    #     if two constraints are active i.e two of the triangle barycentric coordinate conditions are not satisfied
    #         use point point distance formula
    #     if one is active
    #         use point edge distance formula
    #     if none are active
    #         use edge edge distance formula
    #
    # For each soft edge rigid edge pair
    #      get normalized edge coordinates
    #
    #      if two constraints are active
    #          again point point distance
    #      if one constriant is active
    #          point edge
    #      if none:
    #          edge edge
    #
    # For each rigid point:
    #

    # 3 soft points vs rigid triangle
    min_point_triangle_distance(s0, r0, r1, r2, p0, d_hat, distance)
    min_point_triangle_distance(s1, r0, r1, r2, p1, d_hat, distance)
    min_point_triangle_distance(s2, r0, r1, r2, p2, d_hat, distance)

    # 3 rigid points vs soft triangle
    min_triangle_point_distance(r0, s0, s1, s2, p0, p1, p2, d_hat, distance)
    min_triangle_point_distance(r1, s0, s1, s2, p0, p1, p2, d_hat, distance)
    min_triangle_point_distance(r2, s0, s1, s2, p0, p1, p2, d_hat, distance)

    # 9 soft edges x rigid edges
    min_edge_edge_distance(s0, s1, r0, r1, p0, p1, d_hat, distance)
    min_edge_edge_distance(s0, s1, r1, r2, p0, p1, d_hat, distance)
    min_edge_edge_distance(s0, s1, r2, r0, p0, p1, d_hat, distance)
    min_edge_edge_distance(s1, s2, r0, r1, p1, p2, d_hat, distance)
    min_edge_edge_distance(s1, s2, r1, r2, p1, p2, d_hat, distance)
    min_edge_edge_distance(s1, s2, r2, r0, p1, p2, d_hat, distance)
    min_edge_edge_distance(s2, s0, r0, r1, p2, p0, d_hat, distance)
    min_edge_edge_distance(s2, s0, r1, r2, p2, p0, d_hat, distance)
    min_edge_edge_distance(s2, s0, r2, r0, p2, p0, d_hat, distance)


@wp.kernel
def eval_rigid_soft_barrier(
    rigid_points: wp.array[wp.vec3],
    rigid_tris: wp.array2d[wp.int32],
    point_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_transform: wp.array[wp.transform],
    rigid_bvh: wp.uint64,
    particle_q: wp.array[wp.vec3],
    surface_indices: wp.array[wp.int32],
    d_hat: wp.float32,
    stiffness: wp.array[wp.float32],
    energy: wp.array[wp.float32],
    grad: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    kappa = stiffness[0]
    tid = wp.tid()

    x = particle_q[surface_indices[tid]]

    d = d_hat + 1.0
    rigid_point = wp.vec3()

    # Here we are querying a point against the rigid aabbs
    # Instead we could query a triangle against the rigid aabbs
    # then we
    query = wp.bvh_query_aabb(
        rigid_bvh,
        x - wp.vec3(d_hat, d_hat, d_hat),
        x + wp.vec3(d_hat, d_hat, d_hat),
    )
    i = wp.int32(0)

    while wp.bvh_query_next(query, i):
        shape = point_shape[rigid_tris[i, 0]]
        transform = wp.transform_multiply(
            body_transform[shape_body[shape]], shape_transform[shape]
        )
        tri0 = wp.transform_point(transform, rigid_points[rigid_tris[i, 0]])
        tri1 = wp.transform_point(transform, rigid_points[rigid_tris[i, 1]])
        tri2 = wp.transform_point(transform, rigid_points[rigid_tris[i, 2]])

        tri_point = closest_point_on_triangle(x, tri0, tri1, tri2)
        tri_dist = wp.length(x - tri_point)
        if tri_dist < d:
            rigid_point = tri_point
            d = tri_dist

    if d < d_hat:
        logddhat = wp.log(d / d_hat)

        dbdd = -2.0 * (d - d_hat) * logddhat - (d - d_hat) * (d - d_hat) / d
        d2bdd2 = -2.0 * (logddhat + (d - d_hat) / d) - (
            2.0 * d * (d - d_hat) - (d - d_hat) * (d - d_hat)
        ) / (d * d)
        dddx = (x - rigid_point) / d
        d2ddx2 = (
            d * d * wp.identity(3, wp.float32)
            - wp.outer(x - rigid_point, x - rigid_point)
        ) / (d * d * d)

        energy[surface_indices[tid]] += -kappa * (d - d_hat) * (d - d_hat) * logddhat
        grad[surface_indices[tid]] += kappa * dbdd * dddx
        hessian[surface_indices[tid]] += (
            kappa * d2bdd2 * wp.outer(dddx, dddx) + dbdd * d2ddx2
        )


@wp.kernel
def eval_mesh_boundary_energy(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_src_ptrs: wp.array[wp.uint64],
    shape_count: wp.int32,
    particle_q: wp.array[wp.vec3],
    d_hat: wp.float32,
    stiffness: wp.array[wp.float32],
    energy: wp.array[wp.float32],
    grad: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    tid = wp.tid()
    x = particle_q[tid]
    kappa = stiffness[0]

    for i in range(shape_count):
        if shape_type[i] != MESH_TYPE:
            continue
        mesh = shape_src_ptrs[i]

        # In the kernel, bring x into mesh local space first
        body_transform = body_q[shape_body[i]]
        shape_world_transform = wp.transform_multiply(
            body_transform, shape_transform[i]
        )
        x_shape_local = wp.transform_point(
            wp.transform_inverse(shape_world_transform), x
        )
        query = wp.mesh_query_point_no_sign(mesh, x_shape_local, d_hat)

        if query.result:
            mesh_pos = wp.transform_point(
                shape_world_transform,
                wp.mesh_eval_position(mesh, query.face, query.u, query.v),
            )

            d = wp.length(x - mesh_pos)
            logddhat = wp.log(d / d_hat)

            dbdd = -2.0 * (d - d_hat) * logddhat - (d - d_hat) * (d - d_hat) / d
            d2bdd2 = -2.0 * (logddhat + (d - d_hat) / d) - (
                2.0 * d * (d - d_hat) - (d - d_hat) * (d - d_hat)
            ) / (d * d)
            dddx = (x - mesh_pos) / d
            d2ddx2 = (
                d * d * wp.identity(3, wp.float32)
                - wp.outer(x - mesh_pos, x - mesh_pos)
            ) / (d * d * d)

            energy[tid] += -kappa * (d - d_hat) * (d - d_hat) * logddhat
            grad[tid] += kappa * dbdd * dddx
            hessian[tid] += kappa * d2bdd2 * wp.outer(dddx, dddx) + dbdd * d2ddx2


@wp.kernel
def soft_point_rigid_tri_ccd(
    rigid_points: wp.array[wp.vec3],
    rigid_tris: wp.array2d[wp.int32],
    point_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_transform: wp.array[wp.transform],
    rigid_bvh: wp.uint64,
    particle_q: wp.array[wp.vec3],
    surface_indices: wp.array[wp.int32],
    d_hat: wp.float32,
):
    pass


@wp.kernel
def _compute_aabb_lines(
    tri_ids: wp.array[wp.int32],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    tid = wp.tid()
    tri = tri_ids[tid]
    lower = lowers[tri]
    upper = uppers[tri]

    x0 = lower
    x1 = wp.vec3(upper[0], lower[1], lower[2])
    x2 = wp.vec3(lower[0], upper[1], lower[2])
    x3 = wp.vec3(lower[0], lower[1], upper[2])
    x4 = wp.vec3(upper[0], upper[1], lower[2])
    x5 = wp.vec3(lower[0], upper[1], upper[2])
    x6 = wp.vec3(upper[0], lower[1], upper[2])
    x7 = upper

    starts[12 * tid + 0] = x0
    starts[12 * tid + 1] = x0
    starts[12 * tid + 2] = x0

    ends[12 * tid + 0] = x1
    ends[12 * tid + 1] = x2
    ends[12 * tid + 2] = x3

    starts[12 * tid + 3] = x1
    starts[12 * tid + 4] = x1

    ends[12 * tid + 3] = x4
    ends[12 * tid + 4] = x6

    starts[12 * tid + 5] = x2
    starts[12 * tid + 6] = x2

    ends[12 * tid + 5] = x4
    ends[12 * tid + 6] = x5

    starts[12 * tid + 7] = x3
    starts[12 * tid + 8] = x3

    ends[12 * tid + 7] = x5
    ends[12 * tid + 8] = x6

    starts[12 * tid + 9] = x4
    starts[12 * tid + 10] = x5
    starts[12 * tid + 11] = x6

    ends[12 * tid + 9] = x7
    ends[12 * tid + 10] = x7
    ends[12 * tid + 11] = x7


class BarrierEnergy:
    def __init__(self, model: Model):
        self._energy = wp.empty(shape=model.particle_q.shape, dtype=wp.float32)
        self._grad = wp.empty_like(model.particle_q)
        self._hessian_blocks = wp.empty(shape=model.particle_q.shape, dtype=wp.mat33)
        self._hessian = ws.bsr_diag(self._hessian_blocks)
        self._stiffness = wp.array([1.0], wp.float32)
        self._model = model
        self._d_hat = 0.1

        self._build_collision_bvh(model)
        self._build_collision_mesh()

    def _gather_rigid_shapes(self, model: Model):
        """Concatenate every rigid shape's local-space vertices and triangles
        into single arrays, along with the owning shape index of each point
        and triangle."""
        shape_src = model.shape_source
        rigid_points_count = 0
        rigid_tris_count = 0

        for mesh in shape_src:
            if mesh is None:
                continue
            rigid_points_count += mesh.vertices.shape[0]
            rigid_tris_count += mesh.indices.shape[0] // 3

        self._rigid_points = wp.empty(rigid_points_count, dtype=wp.vec3)
        self._rigid_tris = wp.empty((rigid_tris_count, 3), dtype=wp.int32)
        self._rigid_point_shape = wp.empty(rigid_points_count, dtype=wp.int32)
        self._rigid_points_count = rigid_points_count
        self._rigid_tri_count = rigid_tris_count
        pt_offset = 0
        tri_offset = 0
        for i in range(model.shape_count):
            mesh = model.shape_source[i]

            if mesh is None:
                continue

            wp.copy(
                self._rigid_points,
                wp.array(mesh.vertices, dtype=wp.vec3, device="cpu"),
                dest_offset=pt_offset,
            )
            mesh_tri_count = mesh.indices.shape[0] // 3
            wp.copy(
                self._rigid_tris[tri_offset : tri_offset + mesh_tri_count],
                wp.array(mesh.indices, dtype=wp.int32, shape=(mesh_tri_count, 3))
                + pt_offset,
            )
            self._rigid_point_shape[
                pt_offset : pt_offset + mesh.vertices.shape[0]
            ].fill_(i)

            pt_offset += mesh.vertices.shape[0]
            tri_offset += mesh_tri_count

    def _build_collision_bvh(self, model: Model):
        self._gather_rigid_shapes(model)

        self._rigid_lowers = wp.empty(self._rigid_tri_count, dtype=wp.vec3)
        self._rigid_uppers = wp.empty_like(self._rigid_lowers)

        wp.launch(
            compute_static_rigid_aabbs,
            dim=self._rigid_tri_count,
            inputs=[
                self._rigid_points,
                self._rigid_tris,
                self._rigid_point_shape,
                self._model.shape_body,
                self._model.shape_transform,
                self._model.body_q,
            ],
            outputs=[self._rigid_lowers, self._rigid_uppers],
        )

        self._rigid_bvh = wp.Bvh(
            lowers=self._rigid_lowers,
            uppers=self._rigid_uppers,
            constructor="sah",
        )

        self._soft_surface_indices = get_surface_indices(self._model.tri_indices)
        self._soft_lowers = wp.empty(self._model.tri_indices.shape[0], dtype=wp.vec3)
        self._soft_uppers = wp.empty_like(self._soft_lowers)

        wp.launch(
            compute_aabbs,
            dim=self._model.tri_indices.shape[0],
            inputs=[self._model.particle_q, self._model.tri_indices],
            outputs=[self._soft_lowers, self._soft_uppers],
        )

        # self._soft_bvh = wp.Bvh(
        #     lowers=self._soft_lowers, uppers=self._soft_uppers, constructor="lbvh"
        # )

    def _compute_rigid_points_world(self, points_world: wp.array[wp.vec3]):
        """Write every rigid shape's local-space vertices into `points_world`,
        in world space, by applying each point's shape and body transform."""
        wp.launch(
            transform_points_to_world,
            dim=self._rigid_points_count,
            inputs=[
                self._rigid_points,
                self._rigid_point_shape,
                self._model.shape_body,
                self._model.shape_transform,
                self._model.body_q,
            ],
            outputs=[points_world],
        )

    def _build_collision_mesh(self):
        """Gather the rigid and soft body surfaces into a single points/triangles
        pair, plus a per-triangle collision group used to keep the broad phase
        from producing candidates between triangles of the same body.

        Independent of `_build_collision_bvh` — gathers its own rigid shape
        data so it can be used on its own (e.g. by a SAP broad phase)."""
        model = self._model
        self._gather_rigid_shapes(model)

        self._collision_points_count = self._rigid_points_count + model.particle_count
        self._collision_tri_count = self._rigid_tri_count + model.tri_count

        self._collision_points = wp.empty(self._collision_points_count, dtype=wp.vec3)
        self._collision_tris = wp.empty((self._collision_tri_count, 3), dtype=wp.int32)
        self._collision_groups = wp.empty(self._collision_tri_count, dtype=wp.int32)

        self._compute_rigid_points_world(
            self._collision_points[: self._rigid_points_count]
        )
        wp.copy(
            self._collision_points,
            model.particle_q,
            dest_offset=self._rigid_points_count,
        )

        wp.copy(self._collision_tris[: self._rigid_tri_count], self._rigid_tris)
        wp.copy(
            self._collision_tris[self._rigid_tri_count :],
            model.tri_indices + self._rigid_points_count,
        )

        wp.launch(
            compute_rigid_tri_collision_groups,
            dim=self._rigid_tri_count,
            inputs=[self._rigid_tris, self._rigid_point_shape, model.shape_body],
            outputs=[self._collision_groups[: self._rigid_tri_count]],
        )
        # All soft body triangles share one collision group, distinct from every
        # rigid body's group, treating the soft body as the body right after the
        # last rigid body.
        self._collision_groups[self._rigid_tri_count :].fill_(-(model.body_count + 2))

        self._build_collision_broadphase()

    def _build_collision_broadphase(self):
        """Set up a SAP broad phase over the combined rigid + soft triangles
        built by `_build_collision_mesh`, along with the scratch buffers
        `evaluate` needs to refresh AABBs and query candidate pairs each call."""
        tri_count = self._collision_tri_count

        self._collision_lowers = wp.empty(tri_count, dtype=wp.vec3)
        self._collision_uppers = wp.empty_like(self._collision_lowers)

        # A single simulation world: every triangle collides freely with every
        # other triangle outside its own body, subject to `self._collision_groups`.
        self._collision_world = wp.zeros(tri_count, dtype=wp.int32)

        # Half the barrier distance per triangle, so a pair whose gap-expanded
        # boxes overlap is separated by no more than d_hat (the two halves sum
        # at the pair overlap test).
        self._collision_gap = wp.full(tri_count, self._d_hat / 2.0, dtype=wp.float32)

        # Upper bound on simultaneously overlapping pairs; matches the
        # worst-case sizing newton's own SAP-backed CollisionPipeline uses.
        self._collision_candidate_pair_max = tri_count * (tri_count - 1) // 2
        self._collision_candidate_pairs = wp.zeros(
            self._collision_candidate_pair_max, dtype=wp.vec2i
        )
        self._collision_candidate_pair_count = wp.zeros(1, dtype=wp.int32)

        self._collision_broadphase = BroadPhaseSAP(self._collision_world)

    def _update_collision_candidates(self, x: wp.array[wp.vec3]):
        """Refresh the combined collision mesh with the current rigid body and
        soft body positions, recompute triangle AABBs, and query the SAP broad
        phase for candidate pairs. Candidates aren't consumed yet — this just
        keeps `self._collision_candidate_pairs` ready to be read once the
        energy evaluation switches over from the BVH."""
        self._compute_rigid_points_world(
            self._collision_points[: self._rigid_points_count]
        )
        wp.copy(self._collision_points, x, dest_offset=self._rigid_points_count)

        wp.launch(
            compute_aabbs,
            dim=self._collision_tri_count,
            inputs=[self._collision_points, self._collision_tris],
            outputs=[self._collision_lowers, self._collision_uppers],
        )

        self._collision_broadphase.launch(
            self._collision_lowers,
            self._collision_uppers,
            self._collision_gap,
            self._collision_groups,
            self._collision_world,
            self._collision_tri_count,
            self._collision_candidate_pairs,
            self._collision_candidate_pair_count,
        )

    def evaluate(
        self, x: wp.array[wp.vec3]
    ) -> tuple[wp.array[wp.float32], wp.array[wp.vec3], ws.BsrMatrix]:

        # Not consumed yet — this keeps self._collision_candidate_pairs current
        # so the energy kernels below can be switched from the BVH query over
        # to reading candidate pairs directly.
        self._update_collision_candidates(x)

        wp.launch(
            compute_aabbs,
            dim=self._model.tri_indices.shape[0],
            inputs=[x, self._model.tri_indices],
            outputs=[self._soft_lowers, self._soft_uppers],
        )
        self._energy.zero_()
        self._grad.zero_()
        self._hessian_blocks.zero_()
        # print(self._collision_candidate_pair_count.numpy())

        # count = self._collision_candidate_pair_count.numpy()[0]
        # pairs = self._collision_candidate_pairs.numpy()[:count]
        # groups = self._collision_groups.numpy()
        # same_side = (groups[pairs[:, 0]] == groups[pairs[:, 1]]).sum()
        # print(
        #     f"{same_side}/{count} candidate pairs share a collision group (should be 0)"
        # )

        # wp.launch(
        #     eval_rigid_soft_barrier,
        #     dim=self._soft_surface_indices.shape[0],
        #     inputs=[
        #         self._rigid_points,
        #         self._rigid_tris,
        #         self._rigid_point_shape,
        #         self._model.shape_body,
        #         self._model.shape_transform,
        #         self._model.body_q,
        #         self._rigid_bvh.id,
        #         x,
        #         self._soft_surface_indices,
        #         self._d_hat,
        #         self._stiffness,
        #     ],
        #     outputs=[
        #         self._energy,
        #         self._grad,
        #         self._hessian_blocks,
        #     ],
        # )

        # wp.launch(
        #     eval_mesh_boundary_energy,
        #     dim=model.particle_count,
        #     inputs=[
        #         model.body_q,
        #         model.shape_body,
        #         model.shape_transform,
        #         model.shape_type,
        #         model.shape_source_ptr,
        #         model.shape_count,
        #         x,
        #         0.1,
        #         self._stiffness,
        #     ],
        #     outputs=[self._energy, self._grad, self._hessian_blocks],
        # )
        ws.bsr_set_diag(self._hessian, self._hessian_blocks)
        return self._energy, self._grad, self._hessian

    def log_aabbs(self, viewer: ViewerUSD):
        # Only draw AABBs for triangles actually touched by a broad-phase
        # candidate pair, not every rigid/soft triangle in the scene.
        count = int(self._collision_candidate_pair_count.numpy()[0])
        pairs = self._collision_candidate_pairs.numpy()[:count]
        tri_ids_np = np.unique(pairs.reshape(-1))

        starts = wp.empty(len(tri_ids_np) * 12, wp.vec3)
        ends = wp.empty_like(starts)
        colors = wp.full_like(starts, wp.vec3(1.0, 0.0, 0.0))

        if len(tri_ids_np) > 0:
            tri_ids = wp.array(tri_ids_np, dtype=wp.int32)
            wp.launch(
                _compute_aabb_lines,
                dim=len(tri_ids_np),
                inputs=[tri_ids, self._collision_lowers, self._collision_uppers],
                outputs=[starts, ends],
            )

        viewer.log_lines(
            name="aabbs",
            starts=starts,
            ends=ends,
            colors=colors,
            # ViewerUSD only authors new line positions when num_lines > 0, so
            # a USD time-sampled attribute holds its last value once we hit
            # zero candidates - explicitly hide the primitive so stale boxes
            # from an earlier frame don't keep rendering.
            hidden=len(tri_ids_np) == 0,
        )

    def ccd_line_search(self, x: wp.array[wp.vec3], dx: wp.array[wp.vec3]):
        # build swept AABBs
        # point triangle
        # triangle point
        # edge edge
        pass
