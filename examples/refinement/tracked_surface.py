"""Semi-transparent overlay of the PokeFlex dataset's fused surface tracking.

``PlushOctopus_T1/mesh_trajectories_canonical.npy`` is the fused 7-view surface
tracking for the episode: one ``(15002, 3)`` vertex cloud per frame, in the
PokeFlex world frame (metres), sharing frame 0 with the sim's rest pose. The
triangle topology and a rest-pose template come from ``template_canonical.obj``
(authored in millimetres), and ``valid_mask_canonical.npy`` flags which vertices
were actually tracked on each frame.

:class:`TrackedSurfaceOverlay` loads that trajectory and registers it as one
polyscope surface mesh, drawn see-through so it can be compared frame-by-frame
against the running / replayed simulation mesh. Only numpy + polyscope are
imported here (no Warp / Newton), so the lightweight ``replay_octopus`` tool can
use it too.

The fused tracking carries a slow whole-body drift (the reconstructed cloud
translates ~5 cm and yaws several degrees over the episode, with a transient
blow-up to ~18 cm / 25 deg near the end) that is a tracking artefact rather than
physical rigid motion of the plush body. ``detrend`` removes it: every frame is
mapped back onto the ``start_frame`` pose by a per-frame rigid transform fitted
(robustly, so the poke dent is not absorbed) to the vertices the two frames both
tracked. This runs inside :meth:`_positions`, so the overlay *and* the surface
loss that reuses it both see de-drifted geometry.
"""

import os

import numpy as np
import polyscope as ps

# Capture rate of mesh_trajectories_canonical.npy (PokeFlex forward-sim
# frame_dt = 0.03333 s); used to map a wall-clock sim time onto a track frame.
NATIVE_FPS = 30.0

DEFAULT_TRAJECTORY = "PlushOctopus_T1/mesh_trajectories_canonical.npy"


def _read_obj_vertices(path):
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64)


def _read_obj_faces(path):
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(faces, dtype=np.int32)


def _vertex_adjacency(faces, n_vertices):
    """Undirected mesh-graph edges as (src, dst) plus per-vertex degree."""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.unique(np.concatenate([edges, edges[:, ::-1]], axis=0), axis=0)
    src, dst = edges[:, 0], edges[:, 1]
    degree = np.maximum(np.bincount(src, minlength=n_vertices), 1)
    return src, dst, degree


def _fill_invalid(positions, invalid, src, dst, degree, init, iters=10):
    """Replace untracked vertices with an average of their mesh neighbours
    (Jacobi relaxation, seeded from the rest-pose template) so they ride along
    with nearby tracked geometry instead of visibly lagging."""
    filled = positions.copy()
    filled[invalid] = init[invalid]
    n = degree.shape[0]
    for _ in range(iters):
        neighbour_avg = np.stack(
            [np.bincount(src, weights=filled[dst, d], minlength=n) for d in range(3)],
            axis=1,
        ) / degree[:, None]
        filled[invalid] = neighbour_avg[invalid]
    return filled


def _fit_rigid_robust(ref, cur, *, iters=6, tukey_c=3.0):
    """Rigid transform ``(R, t)`` with ``cur ~= ref @ R.T + t``, fitted by
    iteratively reweighted Kabsch so a localised deformation (the poke) is
    treated as an outlier and does not tilt the fit.

    ``ref`` / ``cur`` are ``(M, 3)`` corresponding point sets (the vertices both
    frames tracked). Weights start uniform and are knocked down for points whose
    residual exceeds a few median-absolute-deviations (a Tukey biweight)."""
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    R = np.eye(3)
    t = np.zeros(3)
    if ref.shape[0] < 3:
        return R, t
    w = np.ones(ref.shape[0])
    for _ in range(iters):
        wsum = w.sum()
        if wsum < 1e-12:
            break
        ref_c = (w[:, None] * ref).sum(0) / wsum
        cur_c = (w[:, None] * cur).sum(0) / wsum
        h = (w[:, None] * (ref - ref_c)).T @ (cur - cur_c)
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        t = cur_c - R @ ref_c
        resid = np.linalg.norm(ref @ R.T + t - cur, axis=1)
        scale = 1.4826 * np.median(resid) + 1e-9
        x = resid / (tukey_c * scale)
        w = np.where(x < 1.0, (1.0 - x * x) ** 2, 0.0)
    return R, t


class TrackedSurfaceOverlay:
    """Playback of the dataset surface tracking as one see-through polyscope
    surface mesh, kept in lockstep with the sim (frame 0 = the shared rest pose).

    Args:
        path: ``mesh_trajectories_canonical.npy`` (or ``None`` / ``"none"`` to
            disable the overlay). ``template_canonical.obj`` and
            ``valid_mask_canonical.npy`` are read from the same directory.
        alpha: overlay opacity (0 = invisible, 1 = opaque).
        rate: track seconds advanced per sim second (matches
            ``--tool-playback-rate``).
        loop: wrap around to frame 0 after the last track frame.
        start_frame: track frame shown at ``sim_time`` 0. The sim starts from
            the tracked frame its initial mesh was built off of (not the
            episode's frame 0), so the overlay is advanced to the same frame.
            Also the reference pose the ``detrend`` rigid fit maps every frame
            back onto (zero correction at ``sim_time`` 0).
        detrend: whole-body drift removal applied in :meth:`_positions`.
            ``"none"`` = raw tracking; ``"translation"`` = subtract the per-frame
            shift of the common-tracked centroid; ``"rigid"`` = subtract a
            per-frame robustly-fitted rigid transform (translation + rotation).
        color: flat RGB for the overlay mesh.
        name: polyscope structure name.
    """

    def __init__(self, path=DEFAULT_TRAJECTORY, alpha=0.3, *, rate=1.0,
                 loop=False, start_frame=0, detrend="none",
                 color=(0.20, 0.50, 0.90), name="Tracked surface"):
        self.active = bool(path) and str(path).lower() not in ("", "none", "off")
        self.ps_mesh = None
        self.name = name
        self.start_frame = int(start_frame)
        self.detrend = str(detrend or "none").lower()
        self._detrend_cache = {}
        if not self.active:
            return

        if not os.path.exists(path):
            print(f"tracked surface: {path} not found; overlay disabled")
            self.active = False
            return

        data_dir = os.path.dirname(path)
        obj_path = os.path.join(data_dir, "template_canonical.obj")
        mask_path = os.path.join(data_dir, "valid_mask_canonical.npy")
        if not os.path.exists(obj_path):
            print(
                f"tracked surface: no template_canonical.obj next to {path}; "
                "overlay disabled"
            )
            self.active = False
            return

        self.traj = np.load(path, mmap_mode="r")            # (T, N, 3) metres
        self.T = int(self.traj.shape[0])
        n_vertices = int(self.traj.shape[1])
        self.alpha = float(alpha)
        self.rate = float(rate) if rate else 1.0
        self.loop = bool(loop)

        self.faces = _read_obj_faces(obj_path)
        self.template = _read_obj_vertices(obj_path) / 1000.0   # obj is in mm
        self.valid = (
            np.load(mask_path, mmap_mode="r") if os.path.exists(mask_path) else None
        )
        self.src, self.dst, self.degree = _vertex_adjacency(self.faces, n_vertices)

        start = max(0, min(self.start_frame, self.T - 1))
        # Reference pose the detrend fit maps every frame back onto (raw, so the
        # fit does not recurse through _positions), plus its tracked mask.
        self._detrend_start = start
        self._detrend_ref = (
            self._raw_positions(start).astype(np.float64)
            if self.detrend != "none" else None
        )
        self._detrend_ref_valid = (
            np.asarray(self.valid[start], dtype=bool)
            if (self.detrend != "none" and self.valid is not None) else None
        )
        if self.detrend != "none":
            print(f"tracked surface: whole-body detrend = {self.detrend} "
                  f"(reference frame {start})")

        self.ps_mesh = ps.register_surface_mesh(
            name, self._positions(start), self.faces, smooth_shade=True
        )
        self.ps_mesh.set_edge_width(0.0)
        self.ps_mesh.set_color(list(color))
        self.ps_mesh.set_transparency(self.alpha)
        print(
            f"tracked surface overlay: {self.T} frames, {n_vertices} verts, "
            f"{len(self.faces)} tris, alpha {self.alpha:g}  ({path})"
        )

    # ------------------------------------------------------------------
    def frame_for_time(self, sim_time):
        """Track frame index for a wall-clock ``sim_time`` (seconds), measured
        from ``start_frame`` (the frame shown at ``sim_time`` 0)."""
        idx = self.start_frame + int(round(float(sim_time) * NATIVE_FPS * self.rate))
        if self.loop and self.T:
            return idx % self.T
        return max(0, min(idx, self.T - 1))

    def positions(self, idx):
        """Tracked vertex positions at frame ``idx`` (metres), with untracked
        vertices filled from their mesh neighbours. Public entry point for
        consumers such as :mod:`mfem.refinement.surface_loss`."""
        return self._positions(idx)

    def _raw_positions(self, idx):
        """Tracked positions at frame ``idx`` with untracked vertices filled,
        before any whole-body detrend."""
        pos = np.asarray(self.traj[idx], dtype=np.float64)
        if self.valid is not None:
            invalid = ~np.asarray(self.valid[idx], dtype=bool)
            if invalid.any():
                pos = _fill_invalid(
                    pos, invalid, self.src, self.dst, self.degree, self.template
                )
        return pos.astype(np.float32)

    def _detrend_transform(self, idx):
        """``(R, t)`` such that ``raw @ R.T + t`` maps frame ``idx`` onto the
        reference pose (drift removed). Cached per frame; ``R`` is identity for
        ``detrend='translation'`` and ``(eye, 0)`` when the fit is degenerate."""
        if self.detrend == "none" or idx == self._detrend_start:
            return np.eye(3), np.zeros(3)
        cached = self._detrend_cache.get(idx)
        if cached is not None:
            return cached

        cur = self._raw_positions(idx).astype(np.float64)
        ref = self._detrend_ref
        mask = np.asarray(self.valid[idx], dtype=bool) if self.valid is not None \
            else np.ones(cur.shape[0], dtype=bool)
        if self._detrend_ref_valid is not None:
            mask = mask & self._detrend_ref_valid

        if mask.sum() < 3:
            R, t = np.eye(3), np.zeros(3)
        elif self.detrend == "translation":
            R = np.eye(3)
            t = ref[mask].mean(0) - cur[mask].mean(0)
        else:  # "rigid": fit cur -> ref, robustly
            R_fwd, t_fwd = _fit_rigid_robust(cur[mask], ref[mask])
            R, t = R_fwd, t_fwd

        self._detrend_cache[idx] = (R, t)
        return R, t

    def _positions(self, idx):
        pos = self._raw_positions(idx)
        if self.detrend == "none":
            return pos
        R, t = self._detrend_transform(idx)
        return (pos.astype(np.float64) @ R.T + t).astype(np.float32)

    def set_frame(self, idx):
        if not self.active or self.ps_mesh is None:
            return
        idx = int(idx) % self.T if (self.loop and self.T) else max(0, min(int(idx), self.T - 1))
        self.ps_mesh.update_vertex_positions(self._positions(idx))

    def update_for_time(self, sim_time):
        self.set_frame(self.frame_for_time(sim_time))

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument(
            "--tracked-surface",
            help="Path to the dataset's fused surface-tracking trajectory "
                 "(mesh_trajectories_canonical.npy). Overlaid on the sim as a "
                 "semi-transparent reference mesh, played back in lockstep "
                 "(frame 0 = the shared rest pose). 'none' disables it.",
            type=str,
            default=DEFAULT_TRAJECTORY,
        )
        parser.add_argument(
            "--tracked-surface-alpha",
            help="Opacity of the tracked-surface overlay (0 = invisible, "
                 "1 = opaque).",
            type=float,
            default=0.3,
        )
        parser.add_argument(
            "--tracked-surface-detrend",
            choices=("none", "translation", "rigid"),
            default="rigid",
            help="Remove the tracked mesh's whole-body drift (a tracking "
                 "artefact) by mapping every frame back onto the start frame: "
                 "'rigid' subtracts a per-frame robustly-fitted rotation + "
                 "translation, 'translation' only the centroid shift, 'none' "
                 "leaves the raw tracking. Affects both the overlay and the "
                 "surface-tracking loss.",
        )
