import numpy as np
import pytest
import warp as wp

from mfem.collision import (
    closest_point_segment_segment,
    eval_soft_rigid_distance,
    min_edge_edge_distance,
    min_point_triangle_distance,
    min_triangle_point_distance,
)

D_HAT = 2.0


# ---------------------------------------------------------------------------
# closest_point_segment_segment
# ---------------------------------------------------------------------------


@wp.kernel
def _wrap_closest_point_segment_segment(
    p1: wp.array(dtype=wp.vec3),
    q1: wp.array(dtype=wp.vec3),
    p2: wp.array(dtype=wp.vec3),
    q2: wp.array(dtype=wp.vec3),
    c1_out: wp.array(dtype=wp.vec3),
    c2_out: wp.array(dtype=wp.vec3),
    s_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    c1, c2, s = closest_point_segment_segment(p1[tid], q1[tid], p2[tid], q2[tid])
    c1_out[tid] = c1
    c2_out[tid] = c2
    s_out[tid] = s


def _run_closest_point_segment_segment(p1, q1, p2, q2):
    c1 = wp.empty(1, dtype=wp.vec3)
    c2 = wp.empty(1, dtype=wp.vec3)
    s = wp.empty(1, dtype=wp.float32)
    wp.launch(
        _wrap_closest_point_segment_segment,
        dim=1,
        inputs=[
            wp.array([p1], dtype=wp.vec3),
            wp.array([q1], dtype=wp.vec3),
            wp.array([p2], dtype=wp.vec3),
            wp.array([q2], dtype=wp.vec3),
        ],
        outputs=[c1, c2, s],
    )
    return c1.numpy()[0], c2.numpy()[0], s.numpy()[0]


def test_closest_point_segment_segment_perpendicular_skew():
    # Segment A along x at y=z=0; segment B vertical (along z) at x=0.5, y=1.
    # They cross "in projection" at x=0.5 with a 1.0 gap along y.
    c1, c2, s = _run_closest_point_segment_segment(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.5), (0.5, 1.0, -0.5)
    )
    assert c1 == pytest.approx([0.5, 0.0, 0.0], abs=1e-6)
    assert c2 == pytest.approx([0.5, 1.0, 0.0], abs=1e-6)
    assert s == pytest.approx(0.5, abs=1e-6)
    assert np.linalg.norm(c1 - c2) == pytest.approx(1.0, abs=1e-6)


def test_closest_point_segment_segment_parallel_offset():
    # Parallel segments (degenerate cross-product case) offset by 1 in y.
    c1, c2, _s = _run_closest_point_segment_segment(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.25, 1.0, 0.0), (0.75, 1.0, 0.0)
    )
    assert np.linalg.norm(c1 - c2) == pytest.approx(1.0, abs=1e-6)


def test_closest_point_segment_segment_endpoint_clamped():
    # Segment B sits entirely past the end of segment A, so both closest
    # points should clamp to A's q1 endpoint and B's nearer endpoint.
    c1, c2, s = _run_closest_point_segment_segment(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 1.0, 0.0), (4.0, 1.0, 0.0)
    )
    assert c1 == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert c2 == pytest.approx([3.0, 1.0, 0.0], abs=1e-6)
    # c1 = p1 + s * (q1 - p1) = (0,0,0) + s * (1,0,0) = (1,0,0) => s = 1
    assert s == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# min_point_triangle_distance / min_triangle_point_distance / min_edge_edge_distance
# ---------------------------------------------------------------------------

_TRI_A = np.array([-5.0, -5.0, 0.0])
_TRI_B = np.array([5.0, -5.0, 0.0])
_TRI_C = np.array([0.0, 5.0, 0.0])


@wp.kernel
def _wrap_min_point_triangle_distance(
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.vec3),
    d_hat: wp.float32,
    distance: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    min_point_triangle_distance(x[tid], a[tid], b[tid], c[tid], tid, d_hat, distance)


def _run_min_point_triangle_distance(x, d_hat=D_HAT, sentinel=D_HAT):
    distance = wp.full(1, sentinel, dtype=wp.float32)
    wp.launch(
        _wrap_min_point_triangle_distance,
        dim=1,
        inputs=[
            wp.array([x], dtype=wp.vec3),
            wp.array([_TRI_A], dtype=wp.vec3),
            wp.array([_TRI_B], dtype=wp.vec3),
            wp.array([_TRI_C], dtype=wp.vec3),
            d_hat,
        ],
        outputs=[distance],
    )
    return distance.numpy()[0]


def test_min_point_triangle_distance_matches_known_distance():
    # x sits directly above the triangle's interior at height 1.0.
    x = np.array([0.0, 0.0, 1.0])
    assert _run_min_point_triangle_distance(x) == pytest.approx(1.0, abs=1e-6)


def test_min_point_triangle_distance_beyond_d_hat_leaves_sentinel():
    x = np.array([0.0, 0.0, D_HAT + 1.0])
    assert _run_min_point_triangle_distance(x) == pytest.approx(D_HAT, abs=1e-9)


def test_min_point_triangle_distance_keeps_smaller_of_two_writes():
    # Two calls into the same slot: the second (larger) distance must not
    # clobber the first (smaller) one - that's the entire point of atomic_min.
    distance = wp.full(1, D_HAT, dtype=wp.float32)
    wp.launch(
        _wrap_min_point_triangle_distance,
        dim=1,
        inputs=[
            wp.array([[0.0, 0.0, 0.3]], dtype=wp.vec3),
            wp.array([_TRI_A], dtype=wp.vec3),
            wp.array([_TRI_B], dtype=wp.vec3),
            wp.array([_TRI_C], dtype=wp.vec3),
            D_HAT,
        ],
        outputs=[distance],
    )
    wp.launch(
        _wrap_min_point_triangle_distance,
        dim=1,
        inputs=[
            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
            wp.array([_TRI_A], dtype=wp.vec3),
            wp.array([_TRI_B], dtype=wp.vec3),
            wp.array([_TRI_C], dtype=wp.vec3),
            D_HAT,
        ],
        outputs=[distance],
    )
    assert distance.numpy()[0] == pytest.approx(0.3, abs=1e-6)


@wp.kernel
def _wrap_min_triangle_point_distance(
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.vec3),
    d_hat: wp.float32,
    distance: wp.array(dtype=wp.float32),
):
    min_triangle_point_distance(x[0], a[0], b[0], c[0], 0, 1, 2, d_hat, distance)


def test_min_triangle_point_distance_writes_all_three_vertices():
    x = np.array([0.0, 0.0, 1.0])
    distance = wp.full(3, D_HAT, dtype=wp.float32)
    wp.launch(
        _wrap_min_triangle_point_distance,
        dim=1,
        inputs=[
            wp.array([x], dtype=wp.vec3),
            wp.array([_TRI_A], dtype=wp.vec3),
            wp.array([_TRI_B], dtype=wp.vec3),
            wp.array([_TRI_C], dtype=wp.vec3),
            D_HAT,
        ],
        outputs=[distance],
    )
    # Same distance as the point-triangle case with x/triangle swapped -
    # written into all three vertices since the whole face is what's close.
    assert distance.numpy() == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


@wp.kernel
def _wrap_min_edge_edge_distance(
    sa: wp.array(dtype=wp.vec3),
    sb: wp.array(dtype=wp.vec3),
    ra: wp.array(dtype=wp.vec3),
    rb: wp.array(dtype=wp.vec3),
    d_hat: wp.float32,
    distance: wp.array(dtype=wp.float32),
):
    min_edge_edge_distance(sa[0], sb[0], ra[0], rb[0], 0, 1, d_hat, distance)


def test_min_edge_edge_distance_matches_known_distance():
    ra, rb = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    sa, sb = (0.0, 1.0, 0.5), (0.0, 1.0, -0.5)

    distance = wp.full(2, D_HAT, dtype=wp.float32)
    wp.launch(
        _wrap_min_edge_edge_distance,
        dim=1,
        inputs=[
            wp.array([sa], dtype=wp.vec3),
            wp.array([sb], dtype=wp.vec3),
            wp.array([ra], dtype=wp.vec3),
            wp.array([rb], dtype=wp.vec3),
            D_HAT,
        ],
        outputs=[distance],
    )
    # Both edge endpoints get the same shared distance.
    assert distance.numpy() == pytest.approx([1.0, 1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# eval_soft_rigid_distance (full kernel, one candidate pair)
# ---------------------------------------------------------------------------


def _run_soft_rigid_distance(rigid_tri, soft_tri, rigid_tri_count=1, d_hat=D_HAT):
    points = wp.array(np.vstack([rigid_tri, soft_tri]), dtype=wp.vec3)
    tris = wp.array2d(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32), dtype=wp.int32)
    candidate_pairs = wp.array([[0, 1]], dtype=wp.vec2i)
    candidate_pair_count = wp.array([1], dtype=wp.int32)

    distance = wp.full(3, d_hat, dtype=wp.float32)

    wp.launch(
        eval_soft_rigid_distance,
        dim=1,
        inputs=[
            points,
            tris,
            3,  # rigid_points_count
            rigid_tri_count,
            candidate_pairs,
            candidate_pair_count,
            d_hat,
        ],
        outputs=[distance],
    )
    return distance.numpy()


def test_eval_soft_rigid_distance_vertex_face_only():
    # Big rigid triangle in the z=0 plane; small soft triangle well inside
    # its interior footprint, offset by a height well below every other
    # feature's distance, so only the 3 point-triangle checks fire.
    rigid_tri = np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    h = 0.3
    soft_tri = np.array([[-0.2, 0.0, h], [0.2, 0.0, h], [0.0, 0.3, h]])

    d_hat = 0.5  # smaller than the ~10 unit distances from the other 12 checks
    distance = _run_soft_rigid_distance(rigid_tri, soft_tri, d_hat=d_hat)
    assert distance == pytest.approx([h, h, h], abs=1e-5)


def test_eval_soft_rigid_distance_ignores_same_side_pairs():
    # Sanity check on the rigid/rigid (or soft/soft) early-out: build a scene
    # where tri 0 and tri 1 are both "rigid" (rigid_tri_count=2) so the pair
    # should be skipped entirely, leaving the sentinel untouched.
    rigid_tri = np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    soft_tri = np.array([[-0.2, 0.0, 0.1], [0.2, 0.0, 0.1], [0.0, 0.3, 0.1]])

    distance = _run_soft_rigid_distance(rigid_tri, soft_tri, rigid_tri_count=2)
    assert distance == pytest.approx([D_HAT, D_HAT, D_HAT], abs=1e-9)


def test_eval_soft_rigid_distance_respects_candidate_pair_count():
    # candidate_pair_count[0] == 0 should make the kernel a no-op even
    # though candidate_pairs itself contains a valid-looking entry.
    rigid_tri = np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    soft_tri = np.array([[-0.2, 0.0, 0.1], [0.2, 0.0, 0.1], [0.0, 0.3, 0.1]])

    points = wp.array(np.vstack([rigid_tri, soft_tri]), dtype=wp.vec3)
    tris = wp.array2d(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32), dtype=wp.int32)
    candidate_pairs = wp.array([[0, 1]], dtype=wp.vec2i)
    candidate_pair_count = wp.array([0], dtype=wp.int32)

    distance = wp.full(3, D_HAT, dtype=wp.float32)
    wp.launch(
        eval_soft_rigid_distance,
        dim=1,
        inputs=[
            points,
            tris,
            3,
            1,
            candidate_pairs,
            candidate_pair_count,
            D_HAT,
        ],
        outputs=[distance],
    )

    assert distance.numpy() == pytest.approx([D_HAT, D_HAT, D_HAT], abs=1e-9)


def test_eval_soft_rigid_distance_takes_min_across_all_15_checks():
    # A configuration where the true minimum is the point-triangle distance
    # for just one soft vertex, while the other checks are all farther -
    # verifies the kernel doesn't overwrite a smaller distance with a larger
    # one from a different check touching the same particle.
    rigid_tri = np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    # Vertex 0 sits close to the plane; vertices 1 and 2 are farther away
    # (still within d_hat via edges/points, but at a larger distance).
    soft_tri = np.array([[0.0, 0.0, 0.1], [0.3, 0.0, 0.4], [0.0, 0.3, 0.4]])

    d_hat = 0.5
    distance = _run_soft_rigid_distance(rigid_tri, soft_tri, d_hat=d_hat)

    assert distance[0] == pytest.approx(0.1, abs=1e-5)
    assert np.all(distance <= d_hat)
