"""Surface-tracking loss: MSE of the sim surface against the tracked mesh.

For every vertex on the running simulation's *surface* mesh, find the nearest
point on the dataset's fused tracked surface at the matching frame and average
the squared distances.  This is the one-directional (sim -> tracked) mean
squared error; ``symmetric=True`` also measures tracked -> sim and reports the
average of the two (a Chamfer-style distance).

"Nearest point on the tracked mesh" means the nearest point on its *triangles*
(a point-to-surface projection), not merely the nearest tracked vertex -- the
tracked template has ~3.4 mm edges, so the vertex-only approximation carries a
~1.7 mm bias.  Pass ``project_to_faces=False`` for the faster vertex-only form.

The tracked geometry is taken straight from :class:`TrackedSurfaceOverlay`
(same trajectory file, frame mapping and invalid-vertex handling), so the loss
is measured against exactly the see-through mesh drawn over the sim.  Frames
carry a per-vertex ``valid`` mask; by default untracked vertices and any
triangle touching one are excluded from the target so the loss is only taken
against actually-observed surface.

:class:`TrackedCorrespondenceLoss` is the complementary *material-point*
metric: each sim surface vertex is paired once, at the first frame, with the
closest point on the tracked template (triangle + barycentric weights), and on
every later frame is compared with where that same point of the tracked
surface went. Because the tracked template has fixed topology, that is a true
Lagrangian error and, unlike the nearest-surface distance, it charges the sim
for tangential motion (whole-body sliding, slip under the poker).
``--correspondence`` adds it to the curve as ``corr_*`` columns.

Standalone, this scores a recording written by ``sim_octopus.py --record``:

    python -m mfem.refinement.surface_loss simulation.npz
    python -m mfem.refinement.surface_loss run.npz --tracked-surface PlushOctopus_T1/mesh_trajectories_canonical.npy --out loss.npz
"""

import argparse
import math

import numpy as np

from mfem.refinement.tracked_surface import TrackedSurfaceOverlay


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _closest_point_on_triangles(p, a, b, c):
    """Closest point to each ``p`` on its triangle ``(a, b, c)``.

    Every argument is ``(..., 3)`` and broadcasts together; the result has the
    common broadcast shape.  Vectorised form of the region test in Ericson,
    *Real-Time Collision Detection*, sec. 5.1.5 -- the ``np.where`` cascade is
    ordered low-priority first so a vertex region wins over an edge region wins
    over the interior.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("...i,...i->...", ab, ap)
    d2 = np.einsum("...i,...i->...", ac, ap)
    bp = p - b
    d3 = np.einsum("...i,...i->...", ab, bp)
    d4 = np.einsum("...i,...i->...", ac, bp)
    cp = p - c
    d5 = np.einsum("...i,...i->...", ab, cp)
    d6 = np.einsum("...i,...i->...", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = va + vb + vc
    inv = np.where(denom == 0.0, 0.0, 1.0 / np.where(denom == 0.0, 1.0, denom))
    v = (vb * inv)[..., None]
    w = (vc * inv)[..., None]
    out = a + ab * v + ac * w                                    # interior / face

    def _safe_ratio(num, den):
        return num / np.where(den == 0.0, 1.0, den)

    # Edge BC
    m = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    t = _safe_ratio(d4 - d3, (d4 - d3) + (d5 - d6))[..., None]
    out = np.where(m[..., None], b + (c - b) * t, out)
    # Edge AC
    m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    t = _safe_ratio(d2, d2 - d6)[..., None]
    out = np.where(m[..., None], a + ac * t, out)
    # Edge AB
    m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    t = _safe_ratio(d1, d1 - d3)[..., None]
    out = np.where(m[..., None], a + ab * t, out)
    # Vertex C
    m = (d6 >= 0.0) & (d5 <= d6)
    out = np.where(m[..., None], c, out)
    # Vertex B
    m = (d3 >= 0.0) & (d4 <= d3)
    out = np.where(m[..., None], b, out)
    # Vertex A
    m = (d1 <= 0.0) & (d2 <= 0.0)
    out = np.where(m[..., None], a, out)
    return out


def _vertex_faces_padded(faces, n_vertices):
    """``(vert_faces, degree)``: for every vertex, the indices of the triangles
    it belongs to, right-padded with ``-1`` to the max vertex degree."""
    flat = faces.reshape(-1)
    fidx = np.repeat(np.arange(faces.shape[0]), 3)
    order = np.argsort(flat, kind="stable")
    v_sorted = flat[order]
    f_sorted = fidx[order]
    degree = np.bincount(v_sorted, minlength=n_vertices)
    max_deg = int(degree.max()) if degree.size else 0
    vert_faces = np.full((n_vertices, max(max_deg, 1)), -1, dtype=np.int64)
    starts = np.zeros(n_vertices, dtype=np.int64)
    starts[1:] = np.cumsum(degree)[:-1]
    within = np.arange(v_sorted.shape[0]) - starts[v_sorted]
    vert_faces[v_sorted, within] = f_sorted
    return vert_faces, degree


def surface_triangles_from_tets(tets):
    """Boundary triangulation of a tet mesh: the faces that belong to exactly
    one tet, wound outward.  Handy for scoring a recording that only stores
    tets (``replay_octopus``)."""
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.concatenate(
        [tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
         tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]], axis=0
    )
    key = np.sort(faces, axis=1)
    order = np.lexsort(key.T[::-1])
    key = key[order]
    faces = faces[order]
    keep = np.ones(len(key), dtype=bool)
    same = np.all(key[1:] == key[:-1], axis=1)
    keep[:-1] &= ~same
    keep[1:] &= ~same
    return faces[keep].astype(np.int32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class TrackedSurfaceLoss:
    """MSE of a sim surface against the tracked mesh at the matching frame.

    Args:
        overlay: a :class:`TrackedSurfaceOverlay`.  Its trajectory, triangle
            topology, frame mapping and invalid-vertex fill are reused as-is.
            When the overlay is disabled the loss is a no-op (``evaluate``
            returns ``None``).
        project_to_faces: measure distance to the tracked *triangles* (point to
            surface).  ``False`` = distance to the nearest tracked vertex only
            (faster, ~1.7 mm biased for this template).
        use_valid_mask: drop untracked vertices, and any triangle touching one,
            from the target frame (needs ``overlay.valid``).
        symmetric: also measure tracked -> sim (nearest sim vertex) and report
            the mean of both directions.
        k_neighbors: candidate tracked vertices per sim vertex for the
            face-projection search.  1 (nearest only, then its incident
            triangles) is enough for a template this dense.
    """

    def __init__(self, overlay, *, project_to_faces=True, use_valid_mask=True,
                 symmetric=False, k_neighbors=1, exclude_tool_margin=None):
        """``exclude_tool_margin`` (metres, or ``None``): when :meth:`evaluate`
        is given a tool signed-distance function, tracked vertices closer than
        this to the tool (negative = inside it) are treated as untracked, and
        sim vertices that close are left out of the sim -> tracked mean. The
        tracked mesh cannot see under the poker and interpolates straight
        through it (8 mm inside on average on PlushOctopus_T1), so without this
        a sim that correctly wraps the tool is charged for the dimple, and the
        charge grows with how many sim vertices sit there."""
        self.overlay = overlay
        self.exclude_tool_margin = (
            float(exclude_tool_margin) if exclude_tool_margin is not None else None
        )
        self.active = bool(getattr(overlay, "active", False))
        self.project_to_faces = bool(project_to_faces)
        self.symmetric = bool(symmetric)
        self.k_neighbors = max(int(k_neighbors), 1)
        if not self.active:
            self.faces = None
            return

        self.faces = np.asarray(overlay.faces, dtype=np.int64)
        n_vertices = int(overlay.traj.shape[1])
        self.use_valid_mask = bool(use_valid_mask) and overlay.valid is not None
        self.vert_faces, _ = _vertex_faces_padded(self.faces, n_vertices)

    # ------------------------------------------------------------------
    def _tracked_frame(self, frame):
        """``(positions, valid_mask_or_None)`` for a tracked frame index."""
        pos = np.asarray(self.overlay.positions(frame), dtype=np.float64)
        valid = None
        if self.use_valid_mask:
            valid = np.asarray(self.overlay.valid[frame], dtype=bool)
        return pos, valid

    def _sq_dist_to_mesh(self, query, verts, valid):
        """Squared distance from each ``query`` point to the tracked surface."""
        return self._sq_dist_to_tri_mesh(query, verts, valid, self.faces, self.vert_faces)

    def _sq_dist_to_tri_mesh(self, query, verts, valid, faces, vert_faces):
        """Squared distance from each ``query`` point to the triangle mesh
        ``(verts, faces)``; ``vert_faces`` is the padded vertex -> incident
        faces table. Falls back to nearest-vertex when ``faces`` is None or
        ``project_to_faces`` is off."""
        from scipy.spatial import cKDTree  # noqa: PLC0415

        query = np.asarray(query, dtype=np.float64)
        vmap = np.where(valid)[0] if valid is not None else np.arange(len(verts))
        tree = cKDTree(verts[vmap])
        k = min(self.k_neighbors, len(vmap))
        dist, loc = tree.query(query, k=k)
        dist = np.atleast_2d(dist.reshape(len(query), k))
        loc = np.atleast_2d(loc.reshape(len(query), k))
        near_vidx = vmap[loc]                                 # (Q, k) global idx
        vert_sq = dist[:, 0] ** 2

        if not self.project_to_faces or faces is None:
            return vert_sq

        cand = vert_faces[near_vidx].reshape(len(query), -1)   # (Q, k*deg)
        ok = cand >= 0
        if valid is not None:
            face_all_valid = valid[faces].all(axis=1)
            ok &= face_all_valid[np.where(cand >= 0, cand, 0)]
        cand_safe = np.where(ok, cand, 0)

        tri = verts[faces[cand_safe]]                         # (Q, C, 3, 3)
        qexp = np.broadcast_to(query[:, None, :], tri[..., 0, :].shape)
        cp = _closest_point_on_triangles(
            qexp, tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
        )
        face_sq = ((cp - qexp) ** 2).sum(-1)
        face_sq = np.where(ok, face_sq, np.inf).min(axis=1)
        # Sim vertices whose neighbourhood had no usable triangle fall back to
        # the nearest-vertex distance.
        return np.where(np.isfinite(face_sq), face_sq, vert_sq)

    # ------------------------------------------------------------------
    def evaluate(self, sim_points, sim_time=None, *, frame=None, tool_signed_distance=None,
                 sim_faces=None):
        """Loss for one sim surface point cloud.

        Args:
            sim_points: ``(V, 3)`` world positions of the sim *surface*
                vertices (interior tet vertices should be excluded by the
                caller; see :func:`surface_triangles_from_tets`).
            sim_time: wall-clock sim time; mapped to a tracked frame with
                ``overlay.frame_for_time``.  Ignored when ``frame`` is given.
            frame: tracked frame index, overriding ``sim_time``.
            tool_signed_distance: optional callable ``(N, 3) -> (N,)`` giving
                the signed distance to the tool surface; used with
                ``exclude_tool_margin`` (see the constructor).
            sim_faces: optional ``(F, 3)`` triangles indexing ``sim_points``.
                With ``symmetric`` the tracked -> sim distance is then taken to
                the sim *surface* rather than its nearest vertex, so it does
                not fall just because refinement adds vertices.

        Returns:
            ``dict`` with ``mse`` / ``rmse`` (metres), ``rmse_mm``, ``max_mm``,
            ``frame`` and ``n_sim``; plus ``mse_tracked_to_sim`` /
            ``mse_symmetric`` when ``symmetric``.  ``None`` if disabled.
        """
        if not self.active:
            return None
        if frame is None:
            if sim_time is None:
                raise ValueError("pass sim_time or frame")
            frame = self.overlay.frame_for_time(sim_time)
        frame = int(frame)

        verts, valid = self._tracked_frame(frame)
        sim_points = np.asarray(sim_points, dtype=np.float64)
        if not np.isfinite(sim_points).all():
            # A diverged sim: report NaN rather than crash inside the KD-tree,
            # so a --record / --surface-loss-out run still closes out cleanly
            # (the sweep treats a non-finite curve as an infinite objective).
            out = {"frame": frame, "n_sim": int(len(sim_points)), "n_sim_all": int(len(sim_points)),
                   "n_tracked_excluded": 0, "mse": math.nan, "rmse": math.nan,
                   "rmse_mm": math.nan, "max_mm": math.nan}
            if self.symmetric:
                out["mse_tracked_to_sim"] = math.nan
                out["mse_symmetric"] = math.nan
                out["rmse_symmetric_mm"] = math.nan
            return out
        sim_all = sim_points
        n_sim_all = int(len(sim_points))
        n_tracked_excluded = 0
        if self.exclude_tool_margin is not None and tool_signed_distance is not None:
            margin = self.exclude_tool_margin
            d_trk = np.asarray(tool_signed_distance(verts), dtype=np.float64)
            near_trk = d_trk < margin
            n_tracked_excluded = int(near_trk.sum())
            if valid is None:
                valid = ~near_trk
            else:
                valid = valid & ~near_trk
            d_sim = np.asarray(tool_signed_distance(sim_points), dtype=np.float64)
            sim_points = sim_points[d_sim >= margin]
            if len(sim_points) == 0:
                sim_points = np.zeros((0, 3))
        if len(sim_points):
            sq = self._sq_dist_to_mesh(sim_points, verts, valid)
            mse = float(sq.mean())
            max_sq = float(sq.max())
        else:
            mse = 0.0
            max_sq = 0.0
        out = {
            "frame": frame,
            "n_sim": int(len(sim_points)),
            "n_sim_all": n_sim_all,
            "n_tracked_excluded": n_tracked_excluded,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "rmse_mm": math.sqrt(mse) * 1e3,
            "max_mm": math.sqrt(max_sq) * 1e3,
        }
        if self.symmetric:
            keep = valid if valid is not None else slice(None)
            saved = self.project_to_faces
            self.project_to_faces = False       # sim has no face topology here
            if len(sim_all) and len(verts[keep]):
                if sim_faces is not None and len(sim_faces):
                    self.project_to_faces = saved
                    sim_faces = np.asarray(sim_faces, dtype=np.int64)
                    sim_vf, _ = _vertex_faces_padded(sim_faces, len(sim_all))
                    rev = self._sq_dist_to_tri_mesh(verts[keep], sim_all, None, sim_faces, sim_vf)
                else:
                    rev = self._sq_dist_to_mesh(verts[keep], sim_all, None)
                mse_rev = float(rev.mean())
            else:
                mse_rev = 0.0
            self.project_to_faces = saved
            out["mse_tracked_to_sim"] = mse_rev
            out["mse_symmetric"] = 0.5 * (mse + mse_rev)
            out["rmse_symmetric_mm"] = math.sqrt(out["mse_symmetric"]) * 1e3
        return out


# ---------------------------------------------------------------------------
# Correspondence (material-point) loss
# ---------------------------------------------------------------------------
def _barycentric_on_triangle(p, a, b, c):
    """Barycentric weights ``(u, v, w)`` of ``p`` (assumed to lie in the plane
    of triangle ``(a, b, c)``) with ``p = u a + v b + w c``; clipped to the
    simplex and renormalised so roundoff never pushes a weight negative."""
    ab = b - a
    ac = c - a
    ap = p - a
    d00 = np.einsum("...i,...i->...", ab, ab)
    d01 = np.einsum("...i,...i->...", ab, ac)
    d11 = np.einsum("...i,...i->...", ac, ac)
    d20 = np.einsum("...i,...i->...", ap, ab)
    d21 = np.einsum("...i,...i->...", ap, ac)
    denom = d00 * d11 - d01 * d01
    safe = np.where(denom == 0.0, 1.0, denom)
    v = np.where(denom == 0.0, 0.0, (d11 * d20 - d01 * d21) / safe)
    w = np.where(denom == 0.0, 0.0, (d00 * d21 - d01 * d20) / safe)
    bary = np.stack([1.0 - v - w, v, w], axis=-1)
    bary = np.clip(bary, 0.0, 1.0)
    return bary / bary.sum(axis=-1, keepdims=True)


class TrackedCorrespondenceLoss:
    """Material-point MSE: every sim surface vertex is paired *once* with the
    closest point on the tracked mesh at the first frame, and from then on is
    compared against where *that* point of the tracked surface went.

    The tracked trajectory is a canonical template with fixed topology, so a
    tracked vertex is the same material point in every frame and a
    (triangle, barycentric) pair fixed at the bind frame is a Lagrangian
    marker. Unlike :class:`TrackedSurfaceLoss` (distance to the nearest
    surface point, blind to tangential motion) this charges the sim for
    sliding along the ground or slipping sideways under the poker.

    By default the error is measured on *displacement* from the bind frame
    (``relative=True``): ``(sim_t - sim_0) - (target_t - target_0)``. The
    coarse tet mesh's surface vertices do not sit exactly on the tracked
    template (the bind residual, reported by :meth:`bind`), and that constant
    offset would otherwise be a floor on every frame. ``relative=False``
    compares raw positions instead.

    Only vertices that exist at bind time are scored -- refinement appends
    new vertices past the original count, so the bound set is a stable prefix
    of the surface and the metric cannot move just because refinement adds
    vertices. The caller indexes the same vertex set on every frame.

    Args:
        overlay: a :class:`TrackedSurfaceOverlay` (or duck-typed equivalent:
            ``active``, ``faces``, ``traj``, ``valid``, ``positions(frame)``,
            ``frame_for_time(t)``).
        use_valid_mask: at bind, only pair against triangles whose vertices
            are all tracked; per frame, skip a sim vertex whose target triangle
            touches an untracked vertex on that frame.
        exclude_tool_margin: metres, or ``None``. With a tool signed-distance
            callable at :meth:`evaluate`, a pair is skipped on frames where
            either the sim vertex or its tracked target is closer than this to
            the tool (the tracked mesh interpolates through the poker).
        k_neighbors: candidate tracked vertices per sim vertex at bind (their
            incident triangles are the candidate faces).
        relative: see above.
    """

    def __init__(self, overlay, *, use_valid_mask=True, exclude_tool_margin=None,
                 k_neighbors=4, relative=True):
        self.overlay = overlay
        self.active = bool(getattr(overlay, "active", False))
        self.exclude_tool_margin = (
            float(exclude_tool_margin) if exclude_tool_margin is not None else None
        )
        self.k_neighbors = max(int(k_neighbors), 1)
        self.relative = bool(relative)
        self.tri = None            # (N,) tracked face per bound sim vertex, -1 = unbound
        self.bary = None           # (N, 3)
        self.sim0 = None           # (N, 3) sim positions at bind
        self.target0 = None        # (N, 3) tracked targets at bind
        self.bind_frame = None
        if not self.active:
            self.faces = None
            return
        self.faces = np.asarray(overlay.faces, dtype=np.int64)
        n_vertices = int(overlay.traj.shape[1])
        self.use_valid_mask = bool(use_valid_mask) and getattr(overlay, "valid", None) is not None
        self.vert_faces, _ = _vertex_faces_padded(self.faces, n_vertices)

    @property
    def bound(self):
        return self.tri is not None

    def _tracked_frame(self, frame):
        pos = np.asarray(self.overlay.positions(frame), dtype=np.float64)
        valid = np.asarray(self.overlay.valid[frame], dtype=bool) if self.use_valid_mask else None
        return pos, valid

    def _resolve_frame(self, sim_time, frame):
        if frame is None:
            if sim_time is None:
                raise ValueError("pass sim_time or frame")
            frame = self.overlay.frame_for_time(sim_time)
        return int(frame)

    # ------------------------------------------------------------------
    def bind(self, sim_points, sim_time=None, *, frame=None):
        """Fix the correspondence from ``sim_points`` (``(N, 3)``, the sim
        surface vertices in their initial pose) to the tracked mesh at the
        matching frame. Returns a summary dict (``frame``, ``n_bound``,
        ``n_unbound``, ``bind_rmse_mm``, ``bind_max_mm``); ``None`` if the
        loss is inactive."""
        if not self.active:
            return None
        from scipy.spatial import cKDTree  # noqa: PLC0415

        frame = self._resolve_frame(sim_time, frame)
        verts, valid = self._tracked_frame(frame)
        sim_points = np.asarray(sim_points, dtype=np.float64)
        n = len(sim_points)
        if not np.isfinite(sim_points).all():
            raise ValueError("cannot bind a correspondence to non-finite sim points")

        vmap = np.where(valid)[0] if valid is not None else np.arange(len(verts))
        tree = cKDTree(verts[vmap])
        k = min(self.k_neighbors, len(vmap))
        _, loc = tree.query(sim_points, k=k)
        loc = np.atleast_2d(loc.reshape(n, k))
        cand = self.vert_faces[vmap[loc]].reshape(n, -1)       # (N, C)
        ok = cand >= 0
        if valid is not None:
            face_ok = valid[self.faces].all(axis=1)
            ok &= face_ok[np.where(ok, cand, 0)]
        cand_safe = np.where(ok, cand, 0)
        tri = verts[self.faces[cand_safe]]                      # (N, C, 3, 3)
        qexp = np.broadcast_to(sim_points[:, None, :], tri[..., 0, :].shape)
        cp = _closest_point_on_triangles(qexp, tri[..., 0, :], tri[..., 1, :], tri[..., 2, :])
        sq = np.where(ok, ((cp - qexp) ** 2).sum(-1), np.inf)
        best = sq.argmin(axis=1)
        rows = np.arange(n)
        has = np.isfinite(sq[rows, best])
        face = np.where(has, cand[rows, best], -1)
        cp_best = cp[rows, best]
        a, b, c = (tri[rows, best, i, :] for i in range(3))
        bary = _barycentric_on_triangle(cp_best, a, b, c)
        bary[~has] = 0.0

        self.tri = face
        self.bary = bary
        self.sim0 = sim_points.copy()
        self.target0 = np.where(has[:, None], cp_best, np.nan)
        self.bind_frame = frame
        resid = np.sqrt(sq[rows, best][has]) if has.any() else np.zeros(0)
        return {
            "frame": frame,
            "n_bound": int(has.sum()),
            "n_unbound": int((~has).sum()),
            "bind_rmse_mm": float(np.sqrt((resid ** 2).mean()) * 1e3) if resid.size else 0.0,
            "bind_max_mm": float(resid.max() * 1e3) if resid.size else 0.0,
        }

    # ------------------------------------------------------------------
    def targets(self, frame):
        """``(N, 3)`` tracked target positions at ``frame`` for the bound
        vertices (NaN for unbound ones) plus the per-frame validity mask of
        the target triangles."""
        verts, valid = self._tracked_frame(int(frame))
        tri_safe = np.where(self.tri >= 0, self.tri, 0)
        corners = verts[self.faces[tri_safe]]                    # (N, 3, 3)
        tgt = np.einsum("nk,nki->ni", self.bary, corners)
        ok = self.tri >= 0
        if valid is not None:
            ok &= valid[self.faces[tri_safe]].all(axis=1)
        tgt = np.where(ok[:, None], tgt, np.nan)
        return tgt, ok

    def evaluate(self, sim_points, sim_time=None, *, frame=None, tool_signed_distance=None):
        """Loss for one frame. ``sim_points`` are the *same* vertices, in the
        same order, that were passed to :meth:`bind` (a longer array is
        truncated to that prefix). Returns a dict with ``mse`` / ``rmse`` /
        ``rmse_mm`` / ``max_mm`` (metres or mm), ``frame``, ``n_sim`` (pairs
        scored), ``n_sim_all`` (pairs bound) and ``n_excluded``; ``None`` if
        the loss is inactive."""
        if not self.active:
            return None
        if not self.bound:
            raise RuntimeError("TrackedCorrespondenceLoss.bind() must be called first")
        frame = self._resolve_frame(sim_time, frame)
        n = len(self.tri)
        sim_points = np.asarray(sim_points, dtype=np.float64)
        if len(sim_points) < n:
            raise ValueError(f"expected at least {n} sim points (bound set), got {len(sim_points)}")
        sim_points = sim_points[:n]
        nan_row = {"frame": frame, "n_sim": 0, "n_sim_all": int(n), "n_excluded": int(n),
                   "mse": math.nan, "rmse": math.nan, "rmse_mm": math.nan, "max_mm": math.nan}
        if not np.isfinite(sim_points).all():
            return nan_row

        tgt, ok = self.targets(frame)
        if self.exclude_tool_margin is not None and tool_signed_distance is not None:
            m = self.exclude_tool_margin
            d_sim = np.asarray(tool_signed_distance(sim_points), dtype=np.float64)
            tgt_q = np.where(ok[:, None], tgt, 0.0)
            d_tgt = np.asarray(tool_signed_distance(tgt_q), dtype=np.float64)
            ok &= (d_sim >= m) & (d_tgt >= m)

        if self.relative:
            err = (sim_points - self.sim0) - (tgt - self.target0)
        else:
            err = sim_points - tgt
        sq = (err ** 2).sum(axis=1)[ok]
        if sq.size == 0:
            mse, max_sq = 0.0, 0.0
        else:
            mse, max_sq = float(sq.mean()), float(sq.max())
        return {
            "frame": frame,
            "n_sim": int(ok.sum()),
            "n_sim_all": int(n),
            "n_excluded": int(n - ok.sum()),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "rmse_mm": math.sqrt(mse) * 1e3,
            "max_mm": math.sqrt(max_sq) * 1e3,
        }


CORR_PREFIX = "corr_"


def merge_correspondence_row(row, corr):
    """Fold a :meth:`TrackedCorrespondenceLoss.evaluate` result into a
    :class:`TrackedSurfaceLoss` row under ``corr_*`` keys so both curves ride
    in one record."""
    if corr is None:
        return row
    for k in ("mse", "rmse", "rmse_mm", "max_mm", "n_sim", "n_excluded"):
        row[CORR_PREFIX + k] = corr[k]
    return row


# ---------------------------------------------------------------------------
# Standalone: score a --record recording
# ---------------------------------------------------------------------------
class RecordingOverlay:
    """A ``sim_octopus --record`` recording standing in for the tracked
    surface, so :class:`TrackedSurfaceLoss` can score one simulation against
    another -- typically a coarse run (with or without refinement) against a
    converged fine-mesh run of the same episode. Unlike the camera-tracked
    mesh this reference does see under the tool, so no tool exclusion is
    needed, and the comparison isolates discretisation error from modelling
    error (material, contact, tracking).

    Exposes the subset of the TrackedSurfaceOverlay interface the loss uses:
    ``active``, ``faces``, ``traj``, ``valid`` (None), ``positions(frame)``
    and ``frame_for_time(t)`` (nearest recorded time). The reference must have
    constant topology (no refinement) so its surface faces are fixed.
    """

    def __init__(self, path):
        self.path = path
        rec = _load_recording(path)
        tets0 = rec["tets_at"](0)
        n0 = len(rec["positions_at"](0))
        for i in range(rec["frame_count"]):
            t = rec["tets_at"](i)
            if t.shape != tets0.shape or not np.array_equal(t, tets0):
                raise ValueError(f"{path}: reference recording must have constant topology")
        self.faces = surface_triangles_from_tets(tets0)
        self.faces = self.faces[(self.faces < n0).all(axis=1)]
        self.traj = np.stack([rec["positions_at"](i)[:n0] for i in range(rec["frame_count"])])
        self.times = np.asarray(rec["times"], dtype=np.float64)
        self.T = int(self.traj.shape[0])
        self.valid = None
        self.active = True
        self.capsule = rec["capsule"]

    def positions(self, frame):
        return self.traj[int(frame)]

    def frame_for_time(self, t):
        return int(np.argmin(np.abs(self.times - float(t))))


def _load_recording(path):
    data = np.load(path, allow_pickle=True)
    positions = data["positions"]
    tet_indices = data["tet_indices"]
    frame_count = int(data["frame_count"]) if "frame_count" in data.files else len(positions)
    times = (
        np.asarray(data["times"], dtype=np.float64)
        if "times" in data.files
        else np.arange(frame_count, dtype=np.float64) / float(data["fps"])
    )
    tets_shared = tet_indices.dtype != object and tet_indices.ndim == 2
    capsule = None
    if "capsule_transform" in data.files:
        capsule = {
            "transform": np.asarray(data["capsule_transform"], dtype=np.float64),
            "radius": float(data["capsule_radius"]),
            "half_height": float(data["capsule_half_height"]),
        }
    return {
        "capsule": capsule,
        "fps": float(data["fps"]),
        "episode": str(data["episode"]) if "episode" in data.files else "octopus",
        "start_frame": int(data["start_frame"]) if "start_frame" in data.files else 0,
        "frame_count": frame_count,
        "times": times,
        "positions_at": lambda i: np.asarray(positions[i], dtype=np.float64),
        "tets_at": lambda i: np.asarray(
            tet_indices if tets_shared else tet_indices[i], dtype=np.int64
        ),
    }


def _quat_to_matrix(q):
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def capsule_signed_distance(points, transform7, radius, half_height):
    """Signed distance from ``points`` (N, 3) to a capsule surface given its
    (x, y, z, qx, qy, qz, qw) world transform; the capsule axis is local +Z.
    Negative = inside."""
    P = np.asarray(points, dtype=np.float64)
    pos = np.asarray(transform7[:3], dtype=np.float64)
    R = _quat_to_matrix(transform7[3:7])
    axis = R @ np.array([0.0, 0.0, 1.0])
    a = pos - axis * half_height
    ab = 2.0 * axis * half_height
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-20), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - radius


def score_recording(rec_path, loss, *, playback_rate=1.0, corr=None):
    """Per-frame loss for a recording; returns a dict of stacked arrays. The
    poker capsule pose stored in the recording feeds the loss's tool exclusion
    (see ``TrackedSurfaceLoss(exclude_tool_margin=...)``).

    ``corr``: an optional :class:`TrackedCorrespondenceLoss`. It is bound to
    the surface vertices of the recording's first frame (recordings capture
    after the first step, so that is one sim frame in) and its per-frame
    result is merged into each row under ``corr_*`` keys."""
    rec = _load_recording(rec_path)
    tris_cache = {}
    rows = []
    surf0 = None
    for i in range(rec["frame_count"]):
        pts = rec["positions_at"](i)
        tets = rec["tets_at"](i)
        key = tets.shape
        tris = tris_cache.get(key)
        if tris is None:
            tris = surface_triangles_from_tets(tets)
            tris_cache[key] = tris
        surf = np.unique(tris)
        surf = surf[surf < len(pts)]
        tool = None
        cap = rec["capsule"]
        if cap is not None and i < len(cap["transform"]):
            tf = cap["transform"][i]
            tool = lambda P, tf=tf: capsule_signed_distance(P, tf, cap["radius"], cap["half_height"])
        local_faces = np.searchsorted(surf, tris[(tris < len(pts)).all(axis=1)])
        t = rec["times"][i] * playback_rate
        res = loss.evaluate(pts[surf], t, tool_signed_distance=tool, sim_faces=local_faces)
        if corr is not None and corr.active:
            if surf0 is None:
                surf0 = surf
                info = corr.bind(pts[surf0], t)
                print(f"[correspondence] bound {info['n_bound']} surface vertices to tracked "
                      f"frame {info['frame']} (bind residual RMS {info['bind_rmse_mm']:.2f} mm, "
                      f"max {info['bind_max_mm']:.2f} mm, unbound {info['n_unbound']})")
            merge_correspondence_row(
                res, corr.evaluate(pts[surf0], t, tool_signed_distance=tool))
        res["time"] = float(rec["times"][i])
        rows.append(res)
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", nargs="?", default="simulation.npz",
                    help="recording .npz from sim_octopus.py --record")
    ap.add_argument("--no-project", action="store_true",
                    help="distance to nearest tracked vertex instead of surface")
    ap.add_argument("--symmetric", action="store_true",
                    help="also report the tracked -> sim direction")
    ap.add_argument("--keep-untracked", action="store_true",
                    help="do not drop untracked tracked vertices/faces")
    ap.add_argument("--exclude-tool", type=float, default=None, metavar="MARGIN_M",
                    help="treat tracked vertices closer than MARGIN_M (m, negative = "
                         "inside) to the recorded poker capsule as untracked and leave "
                         "sim vertices that close out of the sim -> tracked mean")
    ap.add_argument("--correspondence", action="store_true",
                    help="also report the material-point loss (each sim surface vertex "
                         "paired once with the closest tracked point at the first frame "
                         "and then compared with where that point went); added to the "
                         "curve as corr_* columns")
    ap.add_argument("--correspondence-absolute", action="store_true",
                    help="correspondence loss on raw positions instead of displacement "
                         "from the first frame (the bind residual then becomes a floor)")
    ap.add_argument("--out", type=str, default=None,
                    help="write the per-frame curve to this .npz")
    ap.add_argument("--reference", type=str, default=None, metavar="REC.npz",
                    help="score against this recording (e.g. a converged fine-mesh "
                         "run of the same episode) instead of the tracked surface")
    ap.add_argument("--episode", default=None,
                    help="PokeFlex episode of the recording (default: its stored "
                         "'episode' key, else octopus); picks the default "
                         "--tracked-surface trajectory.")
    TrackedSurfaceOverlay.add_cli_args(ap)
    ap.set_defaults(tracked_surface=None)
    args = ap.parse_args()

    try:
        import polyscope as ps  # noqa: PLC0415
        ps.init()
    except Exception:
        pass

    rec = _load_recording(args.recording)
    rec_start = rec["start_frame"]
    if args.tracked_surface is None and not args.reference:
        from mfem.refinement.pokeflex_episodes import get_episode  # noqa: PLC0415
        try:
            args.tracked_surface = get_episode(
                args.episode or rec["episode"]
            ).tracked_surface_npy()
        except KeyError:
            args.tracked_surface = None
    if args.reference:
        overlay = RecordingOverlay(args.reference)
    else:
        overlay = TrackedSurfaceOverlay(
            args.tracked_surface, alpha=0.0, start_frame=rec_start,
            detrend=args.tracked_surface_detrend,
        )
    if not overlay.active:
        raise SystemExit("tracked surface unavailable; pass --tracked-surface")

    loss = TrackedSurfaceLoss(
        overlay,
        project_to_faces=not args.no_project,
        use_valid_mask=not args.keep_untracked,
        symmetric=args.symmetric,
        exclude_tool_margin=args.exclude_tool,
    )
    corr = None
    if args.correspondence:
        corr = TrackedCorrespondenceLoss(
            overlay,
            use_valid_mask=not args.keep_untracked,
            exclude_tool_margin=args.exclude_tool,
            relative=not args.correspondence_absolute,
        )
    curve = score_recording(args.recording, loss, corr=corr)

    rmse_mm = curve["rmse_mm"]
    print(
        f"{args.recording}: {len(rmse_mm)} frames  "
        f"RMSE mm  mean {rmse_mm.mean():.2f}  min {rmse_mm.min():.2f}  "
        f"max {rmse_mm.max():.2f}   (frame-mean MSE "
        f"{curve['mse'].mean():.3e} m^2)"
    )
    if args.symmetric:
        s = curve["rmse_symmetric_mm"]
        print(f"  symmetric RMSE mm  mean {s.mean():.2f}  max {s.max():.2f}")
    if args.correspondence:
        c = curve["corr_rmse_mm"]
        print(f"  correspondence RMSE mm  mean {c.mean():.2f}  min {c.min():.2f}  "
              f"max {c.max():.2f}   (frame-mean MSE {curve['corr_mse'].mean():.3e} m^2)")
    if args.out:
        np.savez_compressed(args.out, **curve)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
