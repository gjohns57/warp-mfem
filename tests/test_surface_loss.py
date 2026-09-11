"""TrackedSurfaceLoss (mfem.refinement.surface_loss): the tool-exclusion mask
and the sim-face projection of the tracked -> sim direction, on a synthetic
flat tracked mesh so no dataset is needed."""

import numpy as np
import pytest

from mfem.refinement.surface_loss import (
    RecordingOverlay, TrackedSurfaceLoss, capsule_signed_distance, score_recording,
)


class _FlatOverlay:
    """Duck-typed stand-in for TrackedSurfaceOverlay: a 21x21 grid in z=0."""

    def __init__(self, n=21, size=1.0):
        xs = np.linspace(0.0, size, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        self.verts = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], axis=1)
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
        self.faces = np.concatenate([
            np.stack([a.ravel(), b.ravel(), d.ravel()], 1),
            np.stack([a.ravel(), d.ravel(), c.ravel()], 1),
        ])
        self.traj = self.verts[None]
        self.valid = np.ones((1, n * n), dtype=bool)
        self.active = True

    def positions(self, frame):
        return self.verts

    def frame_for_time(self, t):
        return 0


def _tool(center, radius):
    tf = np.array([*center, 0.0, 0.0, 0.0, 1.0])
    return lambda P: capsule_signed_distance(P, tf, radius, 0.0)


def test_capsule_signed_distance_sphere_case():
    d = capsule_signed_distance(np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0]]),
                                np.array([0, 0, 0, 0, 0, 0, 1.0]), 0.1, 0.0)
    assert d == pytest.approx([0.2, -0.1])


def test_tool_exclusion_drops_dimple_region():
    """A sim surface with a 2 cm dimple under a 'tool' is charged for it
    unless the tool region is excluded on both sides."""
    ov = _FlatOverlay()
    tool = _tool((0.5, 0.5, 0.0), 0.1)
    sim = ov.verts.copy()
    d = np.linalg.norm(sim[:, :2] - 0.5, axis=1)
    sim[d < 0.15, 2] = -0.02                     # dimple the sim under the tool
    plain = TrackedSurfaceLoss(ov, symmetric=True)
    fair = TrackedSurfaceLoss(ov, symmetric=True, exclude_tool_margin=0.06)
    r0 = plain.evaluate(sim, frame=0, tool_signed_distance=tool, sim_faces=ov.faces)
    r1 = fair.evaluate(sim, frame=0, tool_signed_distance=tool, sim_faces=ov.faces)
    assert r0["mse"] > 0.0 and r0["mse_tracked_to_sim"] > 0.0
    assert r1["n_tracked_excluded"] > 0
    assert r1["n_sim"] < r1["n_sim_all"]
    assert r1["mse"] == pytest.approx(0.0, abs=1e-12)
    assert r1["mse_tracked_to_sim"] == pytest.approx(0.0, abs=1e-12)
    # Without a tool distance the margin is inert.
    r2 = fair.evaluate(sim, frame=0, sim_faces=ov.faces)
    assert r2["mse"] == pytest.approx(r0["mse"])


def test_tracked_to_sim_uses_sim_faces_not_vertex_density():
    """Tracked -> sim against a coarse sim surface must not improve just
    because vertices are added on the same plane."""
    ov = _FlatOverlay()
    coarse = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    coarse_faces = np.array([[0, 1, 3], [0, 3, 2]])
    loss = TrackedSurfaceLoss(ov, symmetric=True)
    with_faces = loss.evaluate(coarse, frame=0, sim_faces=coarse_faces)["mse_tracked_to_sim"]
    vertex_only = loss.evaluate(coarse, frame=0)["mse_tracked_to_sim"]
    assert with_faces == pytest.approx(0.0, abs=1e-12)   # the plane is exactly covered
    assert vertex_only > 1e-3                              # nearest-vertex gaps


def _write_recording(path, positions, tets, times):
    np.savez(path, fps=30.0, start_frame=0, frame_count=len(positions), times=times,
             positions=np.asarray(positions, dtype=np.float32), tet_indices=np.asarray(tets, dtype=np.int32),
             capsule_transform=np.zeros((len(positions), 7), dtype=np.float32) + np.array([9, 9, 9, 0, 0, 0, 1], dtype=np.float32),
             capsule_radius=0.01, capsule_half_height=0.01)


def test_recording_as_reference(tmp_path):
    """A recording scored against itself is exact; against a translated copy
    the symmetric RMSE is the translation."""
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    times = np.array([0.0, 1 / 30])
    ref = tmp_path / "ref.npz"; sim = tmp_path / "sim.npz"
    _write_recording(ref, [q, q], tets, times)
    _write_recording(sim, [q, q + [0, 0, 0.02]], tets, times)
    overlay = RecordingOverlay(str(ref))
    assert overlay.T == 2 and len(overlay.faces) == 6
    loss = TrackedSurfaceLoss(overlay, symmetric=True)
    curve = score_recording(str(ref), loss)
    assert np.allclose(curve["mse_symmetric"], 0.0, atol=1e-12)
    curve = score_recording(str(sim), loss)
    assert curve["rmse_mm"][0] == pytest.approx(0.0, abs=1e-6)
    # vertices projected onto faces: the shift is at most 2 cm, less on slanted faces
    assert 0.0 < curve["rmse_mm"][1] <= 20.0 + 1e-6


def test_non_finite_sim_points_give_nan_not_crash():
    ov = _FlatOverlay()
    sim = ov.verts.copy(); sim[0, 2] = np.nan
    row = TrackedSurfaceLoss(ov, symmetric=True).evaluate(sim, frame=0, sim_faces=ov.faces)
    assert np.isnan(row["mse"]) and np.isnan(row["mse_symmetric"])


# ---------------------------------------------------------------------------
# TrackedCorrespondenceLoss
# ---------------------------------------------------------------------------
from mfem.refinement.surface_loss import TrackedCorrespondenceLoss  # noqa: E402


class _MovingOverlay(_FlatOverlay):
    """Flat grid whose frame k is frame 0 rigidly translated by k * shift."""

    def __init__(self, shift, n_frames=3, **kw):
        super().__init__(**kw)
        self.shift = np.asarray(shift, dtype=float)
        self.traj = np.stack([self.verts + k * self.shift for k in range(n_frames)])
        self.valid = np.ones((n_frames, len(self.verts)), dtype=bool)

    def positions(self, frame):
        return self.traj[int(frame)]


def test_correspondence_sees_tangential_slide_surface_loss_ignores():
    """A sim that slides along the plane is invisible to the nearest-surface
    loss but is charged exactly the slide by the correspondence loss."""
    ov = _FlatOverlay()
    sim = ov.verts.copy()
    surf = TrackedSurfaceLoss(ov)
    corr = TrackedCorrespondenceLoss(ov)
    info = corr.bind(sim, frame=0)
    assert info["n_bound"] == len(sim) and info["bind_rmse_mm"] == pytest.approx(0.0, abs=1e-9)
    slid = sim + [0.01, 0.0, 0.0]
    inner = sim[:, 0] < 0.9            # points that stay on the plane after sliding
    assert surf.evaluate(slid[inner], frame=0)["rmse_mm"] == pytest.approx(0.0, abs=1e-9)
    r = corr.evaluate(slid, frame=0)
    assert r["rmse_mm"] == pytest.approx(10.0, abs=1e-9)
    assert r["max_mm"] == pytest.approx(10.0, abs=1e-9)


def test_correspondence_follows_tracked_motion():
    """When the tracked surface moves and the sim moves with it the relative
    error is zero; a sim that stays put is charged the tracked motion."""
    ov = _MovingOverlay(shift=[0.0, 0.0, 0.005])
    sim = ov.verts.copy()
    corr = TrackedCorrespondenceLoss(ov)
    corr.bind(sim, frame=0)
    assert corr.evaluate(sim + 2 * ov.shift, frame=2)["rmse_mm"] == pytest.approx(0.0, abs=1e-9)
    assert corr.evaluate(sim, frame=2)["rmse_mm"] == pytest.approx(10.0, abs=1e-9)


def test_correspondence_binds_to_faces_and_relative_removes_offset():
    """Off-lattice sim points bind to the closest point on a triangle (not a
    vertex); the constant bind residual is removed in relative mode and kept
    as a floor in absolute mode."""
    ov = _MovingOverlay(shift=[0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    sim = rng.uniform(0.1, 0.9, size=(200, 3))
    sim[:, 2] = 0.003                                     # 3 mm above the plane
    rel = TrackedCorrespondenceLoss(ov, relative=True)
    ab = TrackedCorrespondenceLoss(ov, relative=False)
    info = rel.bind(sim, frame=0); ab.bind(sim, frame=0)
    assert info["bind_rmse_mm"] == pytest.approx(3.0, abs=1e-6)   # face projection, not vertex
    assert np.allclose(rel.bary.sum(axis=1), 1.0)
    tgt, ok = rel.targets(0)
    assert ok.all() and np.allclose(tgt[:, :2], sim[:, :2], atol=1e-9) and np.allclose(tgt[:, 2], 0.0)
    assert rel.evaluate(sim, frame=0)["rmse_mm"] == pytest.approx(0.0, abs=1e-9)
    assert ab.evaluate(sim, frame=0)["rmse_mm"] == pytest.approx(3.0, abs=1e-6)


def test_correspondence_prefix_and_valid_and_tool_masks():
    ov = _MovingOverlay(shift=[0.0, 0.0, 0.0])
    sim = ov.verts.copy()
    corr = TrackedCorrespondenceLoss(ov, exclude_tool_margin=0.06)
    corr.bind(sim, frame=0)
    # Extra (refinement-added) vertices past the bound prefix are ignored.
    more = np.concatenate([sim, np.full((50, 3), 5.0)])
    assert corr.evaluate(more, frame=0)["n_sim"] == len(sim)
    # Targets whose triangle touches an untracked vertex on this frame drop out.
    ov.valid[1, :] = False
    ov.valid[1, ov.verts[:, 0] < 0.5] = True
    r = corr.evaluate(sim, frame=1)
    assert 0 < r["n_sim"] < len(sim) and r["n_excluded"] == len(sim) - r["n_sim"]
    # Tool exclusion: a dimple under the tool is not charged.
    dimpled = sim.copy()
    d = np.linalg.norm(sim[:, :2] - 0.5, axis=1)
    dimpled[d < 0.03, 2] = -0.02
    tool = _tool((0.5, 0.5, 0.0), 0.1)
    assert corr.evaluate(dimpled, frame=0)["max_mm"] == pytest.approx(20.0, abs=1e-9)
    r = corr.evaluate(dimpled, frame=0, tool_signed_distance=tool)
    assert r["rmse_mm"] == pytest.approx(0.0, abs=1e-9) and r["n_excluded"] > 0
    # Non-finite input -> NaN, not a crash.
    bad = sim.copy(); bad[3, 1] = np.nan
    assert np.isnan(corr.evaluate(bad, frame=0)["mse"])


def test_score_recording_with_correspondence(tmp_path):
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    times = np.array([0.0, 1 / 30])
    ref = tmp_path / "ref.npz"; sim = tmp_path / "sim.npz"
    _write_recording(ref, [q, q], tets, times)
    _write_recording(sim, [q, q + [0.02, 0, 0]], tets, times)
    overlay = RecordingOverlay(str(ref))
    curve = score_recording(str(sim), TrackedSurfaceLoss(overlay),
                            corr=TrackedCorrespondenceLoss(overlay))
    assert curve["corr_rmse_mm"][0] == pytest.approx(0.0, abs=1e-6)
    # every vertex moved 2 cm along x while its tracked partner stayed
    assert curve["corr_rmse_mm"][1] == pytest.approx(20.0, abs=1e-3)
