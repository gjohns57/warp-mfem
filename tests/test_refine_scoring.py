"""Geometric edge-refinement scoring (mfem.refinement.refinement,
populate_candidates_geometric / populate_tri_candidates_geometric).

Checks the properties that motivated it over the legacy score: scores are
per-edge maxima (no valence bias), edges below twice the minimum length are
never candidates, longer edges of a tet are preferred, only capsule (tool)
contact counts for the vertex term, and a surface tri in contact promotes the
edge nearest its closest point to the tool.
"""

import numpy as np
import pytest
import warp as wp
from newton import GeoType

from mfem.refinement.geometry_hash import get_hashtable_size
from mfem.refinement.refinement import (
    populate_candidates_geometric,
    populate_tri_candidates_geometric,
)

H_MIN = 0.1
D1 = 0.2


def _edge_scores(keys, scores):
    k = keys.numpy(); s = scores.numpy(); m = k != 0
    k = k[m].astype(np.uint64)
    lo = (k & np.uint64(0xFFFFFFFF)).astype(np.int64)
    hi = (k >> np.uint64(32)).astype(np.int64)
    return {(int(a), int(b)): float(v) for a, b, v in zip(lo, hi, s[m])}


def _dm_inv(q, tets):
    out = []
    for t in tets:
        Dm = np.stack([q[t[1]] - q[t[0]], q[t[2]] - q[t[0]], q[t[3]] - q[t[0]]], axis=1)
        out.append(np.linalg.inv(Dm))
    return np.array(out, dtype=np.float32)


def _run_tet_pass(q, tets, energy, distance=None, shape_id=None, shape_type=(GeoType.CAPSULE,),
                  inv_mass=None, mu=1.0, elastic_weight=1.0, vertex_contact_weight=1.0,
                  elastic_stats=(0.0, 0.0), hessian=None, curvature_weight=0.0):
    """elastic_stats = (sum of density / mu, tet count); zeros mean 'absolute
    density' (no mesh-wide mean to be relative to)."""
    q = np.asarray(q, dtype=np.float32); tets = np.asarray(tets, dtype=np.int32)
    n = len(q)
    size = get_hashtable_size(len(tets) * 6, 0.5)
    keys = wp.zeros(size, dtype=wp.uint64); scores = wp.zeros(size, dtype=wp.float32)
    wp.launch(
        populate_candidates_geometric,
        dim=len(tets),
        inputs=[
            wp.array([len(tets)], dtype=wp.int32),
            wp.array(tets, dtype=wp.int32),
            wp.array(np.asarray(energy, dtype=np.float32), dtype=wp.float32),
            wp.array(np.full((len(tets), 3), mu, dtype=np.float32), dtype=wp.float32),
            wp.array(_dm_inv(q, tets), dtype=wp.mat33),
            wp.array(np.ones(n, dtype=np.float32) if inv_mass is None else np.asarray(inv_mass, dtype=np.float32), dtype=wp.float32),
            wp.array(q, dtype=wp.vec3),
            wp.array(np.full(n, 1.0e8, dtype=np.float32) if distance is None else np.asarray(distance, dtype=np.float32), dtype=wp.float32),
            wp.array(np.full(n, -1, dtype=np.int32) if shape_id is None else np.asarray(shape_id, dtype=np.int32), dtype=wp.int32),
            wp.array(np.zeros((n, 3, 3), dtype=np.float32) if hessian is None else np.asarray(hessian, dtype=np.float32), dtype=wp.mat33),
            wp.array(np.array([int(t) for t in shape_type], dtype=np.int32), dtype=wp.int32),
            float(D1), float(H_MIN), float(elastic_weight), float(vertex_contact_weight), float(curvature_weight),
            wp.array(np.asarray(elastic_stats, dtype=np.float32), dtype=wp.float32), int(size),
        ],
        outputs=[keys, scores],
    )
    return keys, scores, _edge_scores(keys, scores)


# Two tets sharing the face (0, 1, 2); edge (0, 1) is the longest edge of both.
Q2 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.3, 0.6, 0.0), (0.3, 0.2, 0.6), (0.3, 0.2, -0.6)]
T2 = [(0, 1, 2, 3), (0, 1, 2, 4)]


def test_scores_are_per_edge_maxima_not_sums():
    """The shared edges must score what the most-strained incident tet gives
    them, not the sum over both tets (the legacy valence bias)."""
    _, _, one = _run_tet_pass(Q2, T2, energy=[1.0, 0.0])
    _, _, both = _run_tet_pass(Q2, T2, energy=[1.0, 1.0])
    for e in [(0, 1), (0, 2), (1, 2)]:
        assert one[e] > 0.0
        assert both[e] == pytest.approx(one[e], rel=1e-5)
    # Edges only in the strain-free tet get nothing.
    assert one[(0, 4)] == 0.0 and one[(1, 4)] == 0.0 and one[(2, 4)] == 0.0


def test_longest_edge_of_tet_scores_highest_and_score_is_dimensionless():
    _, _, sc = _run_tet_pass(Q2, T2, energy=[1.0, 1.0])
    q = np.asarray(Q2)
    lengths = {e: np.linalg.norm(q[e[0]] - q[e[1]]) for e in sc}
    longest = max(lengths, key=lengths.get)
    assert longest == (0, 1)
    assert sc[longest] == max(sc.values())
    # score = (L/h_min) * (L/L_max) * energy / (vol * mu) with L = L_max for the
    # longest edge, so it must equal (L / h_min) * energy density.
    t = np.asarray(T2[0]); Dm = np.stack([q[t[1]] - q[t[0]], q[t[2]] - q[t[0]], q[t[3]] - q[t[0]]], axis=1)
    vol = abs(np.linalg.det(Dm)) / 6.0
    assert sc[longest] == pytest.approx((lengths[longest] / H_MIN) * 1.0 / vol, rel=1e-4)


def test_short_edges_are_registered_but_never_candidates():
    q = np.asarray(Q2) * 0.15  # longest edge = 0.15 < 2 * H_MIN
    _, _, sc = _run_tet_pass(q, T2, energy=[1.0, 1.0])
    assert len(sc) == 9  # all edges present in the table ...
    assert all(v == 0.0 for v in sc.values())  # ... but none may be split


def test_fully_kinematic_edges_are_excluded():
    inv_mass = [0.0, 0.0, 1.0, 1.0, 1.0]
    _, _, sc = _run_tet_pass(Q2, T2, energy=[1.0, 1.0], inv_mass=inv_mass)
    assert (0, 1) not in sc
    assert (0, 2) in sc  # only one pinned endpoint: still splittable


def test_vertex_contact_counts_capsules_only():
    n = len(Q2)
    dist = np.full(n, 1.0e8); dist[3] = 0.5 * D1  # vertex 3 halfway into the barrier range
    sid = np.full(n, -1); sid[3] = 0
    _, _, capsule = _run_tet_pass(Q2, T2, energy=[0.0, 0.0], distance=dist, shape_id=sid, shape_type=(GeoType.CAPSULE,))
    _, _, plane = _run_tet_pass(Q2, T2, energy=[0.0, 0.0], distance=dist, shape_id=sid, shape_type=(GeoType.PLANE,))
    touching = [(0, 3), (1, 3), (2, 3)]
    for e in touching:
        assert capsule[e] > 0.0
        assert plane[e] == 0.0
    for e in capsule:
        if e not in touching:
            assert capsule[e] == 0.0
    # Penetration scores higher than surface contact (no separate override needed).
    dist[3] = -0.5 * D1
    _, _, pen = _run_tet_pass(Q2, T2, energy=[0.0, 0.0], distance=dist, shape_id=sid)
    assert pen[(0, 3)] > capsule[(0, 3)]


def test_tri_contact_promotes_edge_nearest_closest_point():
    # One tet whose face (0, 1, 2) is an equilateral surface tri, so the three
    # tri edges have the same length and only the bary weighting separates them.
    q = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, np.sqrt(3) / 2, 0.0), (0.5, 0.3, 0.8)], dtype=np.float32)
    tets = [(0, 1, 2, 3)]
    keys, scores, before = _run_tet_pass(q, tets, energy=[0.0])
    assert all(v == 0.0 for v in before.values())

    tris = np.array([(0, 1, 2)], dtype=np.int32)
    size = keys.shape[0]

    def tri_pass(bary, d):
        k = wp.clone(keys); s = wp.clone(scores)
        wp.launch(
            populate_tri_candidates_geometric,
            dim=1,
            inputs=[
                wp.array([1], dtype=wp.int32), wp.array(tris, dtype=wp.int32),
                wp.array(np.ones(4, dtype=np.float32), dtype=wp.float32), wp.array(q, dtype=wp.vec3),
                wp.array([d], dtype=wp.float32), wp.array([wp.vec3(*bary)], dtype=wp.vec3),
                wp.array(np.zeros((4, 3, 3), dtype=np.float32), dtype=wp.mat33),
                float(D1), float(H_MIN), 1.0, 0.0, int(size),
            ],
            outputs=[k, s],
        )
        return _edge_scores(k, s)

    # Closest point on edge (1, 2) (bary of vertex 0 is 0): that edge wins outright.
    sc = tri_pass((0.0, 0.5, 0.5), 0.5 * D1)
    assert sc[(1, 2)] > sc[(0, 1)] and sc[(1, 2)] > sc[(0, 2)]
    assert sc[(0, 1)] == pytest.approx(sc[(0, 2)], rel=1e-5)
    assert sc[(1, 2)] == pytest.approx((1.0 / H_MIN) * 1.0 * 0.5 * 1.0, rel=1e-4)
    # Tet edges to the interior vertex are untouched by the tri pass.
    assert sc[(0, 3)] == 0.0

    # Centroid contact: all three edges tie (2/3 weight each).
    sc = tri_pass((1 / 3, 1 / 3, 1 / 3), 0.5 * D1)
    assert sc[(0, 1)] == pytest.approx(sc[(1, 2)], rel=1e-5) == pytest.approx(sc[(0, 2)], rel=1e-5)

    # Outside the barrier range nothing changes.
    sc = tri_pass((0.0, 0.5, 0.5), 2.0 * D1)
    assert all(v == 0.0 for v in sc.values())

    # Scores combine by max with the tet pass, never sum.
    sc_a = tri_pass((0.0, 0.5, 0.5), 0.5 * D1)
    sc_b = tri_pass((0.0, 0.5, 0.5), 0.0)  # at the surface: proximity 1 > 0.5
    assert sc_b[(1, 2)] == pytest.approx(2.0 * sc_a[(1, 2)], rel=1e-5)


def test_elastic_term_is_excess_over_mean_density():
    """With mesh-wide stats the elastic term is max(0, rho / mean(rho) - 1):
    a uniformly strained body scores nothing, twice-the-mean scores 1."""
    q = np.asarray(Q2)
    t = np.asarray(T2[0]); Dm = np.stack([q[t[1]] - q[t[0]], q[t[2]] - q[t[0]], q[t[3]] - q[t[0]]], axis=1)
    vol = abs(np.linalg.det(Dm)) / 6.0
    rho = 1.0 / vol  # density of a tet with energy 1 and mu 1
    # Uniform: every tet at the mean -> nothing to refine.
    _, _, sc = _run_tet_pass(Q2, T2, energy=[1.0, 1.0], elastic_stats=(2.0 * rho, 2.0))
    assert all(v == 0.0 for v in sc.values())
    # Tet 0 at twice the mean -> excess 1 -> longest edge scores L / h_min.
    _, _, sc = _run_tet_pass(Q2, T2, energy=[1.0, 1.0], elastic_stats=(rho, 2.0))
    L = np.linalg.norm(q[0] - q[1])
    assert sc[(0, 1)] == pytest.approx(L / H_MIN, rel=1e-4)


def test_curvature_term_scales_edges_across_the_tool_only():
    """With curvature_weight w the contact score is multiplied by 1 + w L kappa,
    kappa = t^T H t along the edge: a capsule Hessian (I - nn^T)/rho gives
    1/rho for an edge across the axis and 0 for one along it; a plane (H = 0)
    and w = 0 leave the score unchanged."""
    # Single tet with vertex 0 and 1 in contact; axis of the capsule = z, so
    # edge (0, 1) along x runs *across* the axis and edge (0, 3) along z runs
    # *along* it (both endpoints get the same Hessian).
    q = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 1.0)], dtype=np.float32)
    tets = [(0, 1, 2, 3)]
    rho = 2.0
    n = np.array([0.0, 1.0, 0.0])            # outward normal at the contact points
    H = (np.eye(3) - np.outer(n, n)) / rho    # capsule Hessian: 1/rho across, 0 along z? no: 0 along n
    # (I - nn^T)/rho has eigenvalue 1/rho in every direction orthogonal to n,
    # so make the "along-axis" check explicit with a cylinder-style Hessian
    # that is only curved along x.
    Hx = np.zeros((3, 3)); Hx[0, 0] = 1.0 / rho
    hess = np.stack([Hx, Hx, np.zeros((3, 3)), Hx]).astype(np.float32)
    dist = [0.5 * D1, 0.5 * D1, 1.0e8, 0.5 * D1]
    sid = [0, 0, -1, 0]

    _, _, base = _run_tet_pass(q, tets, energy=[0.0], distance=dist, shape_id=sid)
    _, _, w0 = _run_tet_pass(q, tets, energy=[0.0], distance=dist, shape_id=sid, hessian=hess, curvature_weight=0.0)
    _, _, w1 = _run_tet_pass(q, tets, energy=[0.0], distance=dist, shape_id=sid, hessian=hess, curvature_weight=1.0)
    _, _, w2 = _run_tet_pass(q, tets, energy=[0.0], distance=dist, shape_id=sid, hessian=hess, curvature_weight=2.0)
    _, _, flat = _run_tet_pass(q, tets, energy=[0.0], distance=dist, shape_id=sid,
                               hessian=np.zeros((4, 3, 3)), curvature_weight=2.0)

    assert base[(0, 1)] > 0.0
    # Weight 0 and a flat (plane) Hessian are no-ops.
    for k in base:
        assert w0[k] == pytest.approx(base[k], rel=1e-6)
        assert flat[k] == pytest.approx(base[k], rel=1e-6)
    # Across the axis: factor 1 + w * L / rho with L = 1.
    assert w1[(0, 1)] == pytest.approx(base[(0, 1)] * (1.0 + 1.0 / rho), rel=1e-5)
    assert w2[(0, 1)] == pytest.approx(base[(0, 1)] * (1.0 + 2.0 / rho), rel=1e-5)
    # Along the axis (edge (0, 3) is along z, Hx has no zz curvature): unchanged.
    assert base[(0, 3)] > 0.0
    assert w2[(0, 3)] == pytest.approx(base[(0, 3)], rel=1e-6)
    # Non-contact edge (2 is out of range on both ends? no: (1, 2) has vertex 1
    # in contact) -- the edge whose endpoints are both out of contact stays 0.
    assert w2[(2, 3)] == pytest.approx(base[(2, 3)], rel=1e-6)
