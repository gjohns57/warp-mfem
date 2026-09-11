"""Triangle-level contact barrier (mfem.refinement.contact, tri_* kernels).

Checks the tri/segment closest-point geometry, that the per-tri barrier
catches a capsule the per-vertex barrier cannot see (the tunnelling case), and
that the scattered gradient matches a finite difference of the scattered
energy -- i.e. the envelope-theorem chain rule through frozen barycentric
weights is right, both for a face-interior and an edge closest point.
"""

import numpy as np
import pytest
import warp as wp
from newton import GeoType

from mfem.refinement.contact import (
    barrier,
    closest_point_triangle_segment,
    compute_distance,
    compute_tri_distance,
    evaluate_tri_barrier_energy,
)

D0 = 0.05
D1 = 0.2
STIFFNESS = 10.0
RADIUS = 0.1
HALF_HEIGHT = 0.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@wp.kernel
def _wrap_closest_point_triangle_segment(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.vec3),
    p0: wp.array(dtype=wp.vec3),
    p1: wp.array(dtype=wp.vec3),
    bary_out: wp.array(dtype=wp.vec3),
    q_out: wp.array(dtype=wp.vec3),
    d_out: wp.array(dtype=wp.float32),
    hit_out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    bary, q, d, hit = closest_point_triangle_segment(a[tid], b[tid], c[tid], p0[tid], p1[tid])
    bary_out[tid] = bary
    q_out[tid] = q
    d_out[tid] = d
    hit_out[tid] = hit


def _closest(a, b, c, p0, p1):
    bary = wp.empty(1, dtype=wp.vec3)
    q = wp.empty(1, dtype=wp.vec3)
    d = wp.empty(1, dtype=wp.float32)
    hit = wp.empty(1, dtype=wp.int32)
    wp.launch(
        _wrap_closest_point_triangle_segment,
        dim=1,
        inputs=[wp.array([v], dtype=wp.vec3) for v in (a, b, c, p0, p1)],
        outputs=[bary, q, d, hit],
    )
    return bary.numpy()[0], q.numpy()[0], float(d.numpy()[0]), int(hit.numpy()[0])


TRI = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))


def test_closest_face_interior():
    # Vertical segment above the face interior: endpoint-vs-face wins.
    bary, q, d, hit = _closest(*TRI, (0.5, 0.5, 0.3), (0.5, 0.5, 1.3))
    assert hit == 0
    assert d == pytest.approx(0.3, abs=1e-6)
    assert bary == pytest.approx([0.5, 0.25, 0.25], abs=1e-6)
    assert q == pytest.approx([0.5, 0.5, 0.3], abs=1e-6)


def test_closest_edge_crossing():
    # Segment crossing over the a-b edge (y = 0) at x = 1, gap 0.4 in z.
    bary, q, d, hit = _closest(*TRI, (1.0, -1.0, 0.4), (1.0, 1.0, 0.4))
    assert hit == 0
    assert d == pytest.approx(0.4, abs=1e-6)
    # Closest tri point is on the edge... unless the endpoint-vs-face candidate
    # (1, 1, 0.4) -> face point (1, 1, 0) at 0.4 ties. Either way the distance
    # is 0.4 and the closest point is not a vertex.
    assert bary.max() < 1.0 - 1e-6


def test_closest_vertex():
    bary, q, d, hit = _closest(*TRI, (-1.0, -1.0, 0.0), (-1.0, -1.0, 1.0))
    assert hit == 0
    assert d == pytest.approx(np.sqrt(2.0), abs=1e-6)
    assert bary == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_segment_pierces_face():
    bary, q, d, hit = _closest(*TRI, (0.5, 0.5, -1.0), (0.5, 0.5, 1.0))
    assert hit == 1
    assert d == 0.0
    assert bary == pytest.approx([0.5, 0.25, 0.25], abs=1e-6)
    assert q == pytest.approx([0.5, 0.5, 0.0], abs=1e-6)


# ---------------------------------------------------------------------------
# Barrier kernels on a single tri vs. a single capsule
# ---------------------------------------------------------------------------


def _capsule_shapes(pos, rot=(0.0, 0.0, 0.0, 1.0)):
    """One capsule shape on a body at world pose (pos, rot); axis along local +Z."""
    shape_transform = wp.array([wp.transform_identity()], dtype=wp.transform)
    shape_type = wp.array([int(GeoType.CAPSULE)], dtype=wp.int32)
    shape_scale = wp.array([wp.vec3(RADIUS, HALF_HEIGHT, 0.0)], dtype=wp.vec3)
    shape_body = wp.array([0], dtype=wp.int32)
    body_q = wp.array([wp.transform(wp.vec3(*pos), wp.quat(*rot))], dtype=wp.transform)
    return shape_transform, shape_type, shape_scale, shape_body, 1, body_q


def _tri_barrier(verts, capsule_pos, rot=(0.0, 0.0, 0.0, 1.0)):
    """Run compute_tri_distance + evaluate_tri_barrier_energy on one tri.

    Returns (tri energy, per-particle scattered energy (3,), gradient (3, 3),
    Hessian blocks (3, 3, 3, 3), distance, bary, normal)."""
    shapes = _capsule_shapes(capsule_pos, rot)
    q = wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3)
    tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.int32)
    active = wp.array([1], dtype=wp.int32)

    dist = wp.zeros(1, dtype=wp.float32)
    bary = wp.zeros(1, dtype=wp.vec3)
    normal = wp.zeros(1, dtype=wp.vec3)
    sid = wp.zeros(1, dtype=wp.int32)
    wp.launch(
        compute_tri_distance,
        dim=1,
        inputs=[active, tri_indices, q, *shapes],
        outputs=[dist, bary, normal, sid],
    )

    coeffs = wp.vec3(*barrier(D0, 0.0, D1, 0.0, 0.0, 0.0))
    tri_energy = wp.zeros(1, dtype=wp.float32)
    energy = wp.zeros(3, dtype=wp.float32)
    gradient = wp.zeros(3, dtype=wp.vec3)
    rows = wp.zeros(9, dtype=wp.int32)
    cols = wp.zeros(9, dtype=wp.int32)
    values = wp.zeros(9, dtype=wp.mat33)
    wp.launch(
        evaluate_tri_barrier_energy,
        dim=1,
        inputs=[active, tri_indices, dist, bary, normal, wp.array([STIFFNESS], dtype=wp.float32), D0, D1, coeffs],
        outputs=[tri_energy, energy, gradient, rows, cols, values],
    )
    H = np.zeros((3, 3, 3, 3))
    for slot, (r, c) in enumerate(zip(rows.numpy(), cols.numpy())):
        H[r, c] += values.numpy()[slot]
    return (
        float(tri_energy.numpy()[0]),
        energy.numpy(),
        gradient.numpy(),
        H,
        float(dist.numpy()[0]),
        bary.numpy()[0],
        normal.numpy()[0],
    )


def _vertex_min_distance(verts, capsule_pos):
    """The per-particle contact kernel's min distance over the tri's vertices."""
    shapes = _capsule_shapes(capsule_pos)
    q = wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3)
    dist = wp.zeros(3, dtype=wp.float32)
    grad = wp.zeros(3, dtype=wp.vec3)
    hess = wp.zeros(3, dtype=wp.mat33)
    sid = wp.zeros(3, dtype=wp.int32)
    wp.launch(
        compute_distance,
        dim=3,
        inputs=[wp.array([3], dtype=wp.int32), q, *shapes],
        outputs=[dist, grad, hess, sid],
    )
    return float(dist.numpy().min())


# A tri much wider than the capsule, lying in z = 0; the capsule's lower cap
# hovers a small gap above the face centre.
BIG_TRI = [(-3.0, -3.0, 0.0), (3.0, -3.0, 0.0), (0.0, 3.0, 0.0)]
GAP = 0.08
CAPSULE_ABOVE_CENTRE = (0.0, -1.0, GAP + RADIUS + HALF_HEIGHT)


def test_tri_barrier_catches_what_vertices_miss():
    # Every vertex is far outside the barrier range ...
    assert _vertex_min_distance(BIG_TRI, CAPSULE_ABOVE_CENTRE) > D1
    # ... but the face is GAP away from the capsule, well inside it.
    e, e_scatter, g, H, d, bary, n = _tri_barrier(BIG_TRI, CAPSULE_ABOVE_CENTRE)
    assert d == pytest.approx(GAP, abs=1e-5)
    assert n == pytest.approx([0.0, 0.0, -1.0], abs=1e-5)
    assert e > 0.0
    assert e_scatter.sum() == pytest.approx(e, rel=1e-5)
    assert bary.sum() == pytest.approx(1.0, abs=1e-6)
    # Energy decreases when the tri moves *away* from the capsule (-z), so the
    # gradient of every vertex points toward the capsule (+z).
    assert (g[:, 2] > 0.0).all()
    total = g.sum(axis=0)
    assert np.abs(total[:2]).max() < 1e-4 * abs(total[2])


def test_tri_barrier_axis_through_face_is_deepest_penetration():
    e_touch, *_ = _tri_barrier(BIG_TRI, (0.0, -1.0, HALF_HEIGHT + RADIUS))  # cap just touching
    _, _, _, _, d, _, _ = _tri_barrier(BIG_TRI, (0.0, -1.0, 0.0))  # axis pierces the face
    assert d == pytest.approx(-RADIUS, abs=1e-6)
    e_pierce, *_ = _tri_barrier(BIG_TRI, (0.0, -1.0, 0.0))
    assert e_pierce > e_touch > 0.0


def _quat_axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) / 2.0
    return (*(np.sin(half) * axis), np.cos(half))


def _quat_mul(q1, q2):
    """(x, y, z, w) Hamilton product q1 * q2 (apply q2 first, then q1)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _fd_gradient(verts, capsule_pos, rot, h=1e-3):
    verts = np.asarray(verts, dtype=np.float64)
    fd = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            vp = verts.copy()
            vp[i, k] += h
            vm = verts.copy()
            vm[i, k] -= h
            ep = _tri_barrier(vp, capsule_pos, rot)[0]
            em = _tri_barrier(vm, capsule_pos, rot)[0]
            fd[i, k] = (ep - em) / (2.0 * h)
    return fd


@pytest.mark.parametrize(
    "verts,capsule_pos,rot",
    [
        # Face interior contact, capsule tilted a little so nothing is axis-aligned.
        (BIG_TRI, (0.3, -0.8, GAP + RADIUS + HALF_HEIGHT), (0.1, 0.05, 0.0, 0.9937)),
        # Edge contact: capsule lying roughly along x just outside the a-b
        # edge (y = -3), a little above it, its axis turned 15 degrees in the
        # plane so it is *not* parallel to the edge (a parallel axis has a
        # whole interval of equally-close edge points, i.e. a kink in the
        # energy where a finite difference is meaningless).
        (BIG_TRI, (0.4, -3.3, 0.1), _quat_mul(_quat_axis_angle((0.0, 0.0, 1.0), 15.0), _quat_axis_angle((0.0, 1.0, 0.0), 90.0))),
    ],
)
def test_tri_barrier_gradient_matches_finite_difference(verts, capsule_pos, rot):
    rot = np.asarray(rot) / np.linalg.norm(rot)
    e, _, g, H, d, bary, _ = _tri_barrier(verts, capsule_pos, tuple(rot))
    assert 0.0 < d < D1, d
    assert bary.max() < 1.0 - 1e-4, "test intends a non-vertex closest point"
    fd = _fd_gradient(verts, capsule_pos, tuple(rot))
    assert np.abs(g - fd).max() < 2e-2 * np.abs(fd).max()

    # Gauss-Newton Hessian: symmetric and PSD as a 9x9.
    H9 = H.transpose(0, 2, 1, 3).reshape(9, 9)
    assert np.allclose(H9, H9.T, atol=1e-6 * np.abs(H9).max())
    assert np.linalg.eigvalsh(H9).min() > -1e-5 * np.abs(H9).max()
