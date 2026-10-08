"""Replay a mesh recording written by ``sim_octopus.py --record``.

``sim_octopus.py --record run.npz`` dumps, for every rendered frame, the active
soft-body tet mesh (vertex positions + tet indices) and the poker capsule
transform. This script loads that ``.npz`` back into a polyscope window and
plays it as an animation -- no solver, no Warp, just the recorded geometry.
Only the boundary surface of the tet mesh is drawn (recovered per topology
with ``surface_triangles_from_tets``); pass ``--show-tets`` to draw the full
volume mesh instead.

    python -m mfem.refinement.replay_octopus run.npz
    python -m mfem.refinement.replay_octopus run.npz --fps 15 --paused

Recording layout (keys in the ``.npz``)
--------------------------------------
==================  ===================================================  =======
key                 shape / dtype                                        notes
==================  ===================================================  =======
fps                 scalar float                                         capture rate
frame_count         scalar int                                           number of frames F
times               (F,) float64                                         sim_time per frame
positions           (F, N, 3) float32, or (F,) object of (Ni, 3)         per-frame vertices
tet_indices         (M, 4) int32, or (F, M, 4), or (F,) object of (Mi,4) topology
capsule_transform   (F, 7) float32                                       [px py pz qx qy qz qw]
capsule_radius      scalar float                                         poker capsule radius
capsule_half_height scalar float                                         poker capsule half-height
==================  ===================================================  =======

``positions`` / ``tet_indices`` are stored as ``dtype=object`` arrays when
adaptive refinement changed the mesh size mid-run (hence ``allow_pickle=True``
on load); ``tet_indices`` is a single ``(M, 4)`` array when the topology never
changed.
"""

import argparse
import math

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial.transform import Rotation

from mfem.refinement.tracked_surface import TrackedSurfaceOverlay
from mfem.refinement.surface_loss import TrackedSurfaceLoss, surface_triangles_from_tets
from mfem.refinement.pokeflex_episodes import get_episode


def _load(path):
    data = np.load(path, allow_pickle=True)
    positions = data["positions"]
    tet_indices = data["tet_indices"]
    capsule_transform = np.asarray(data["capsule_transform"], dtype=np.float64)
    frame_count = int(data["frame_count"]) if "frame_count" in data.files else len(positions)
    times = (
        np.asarray(data["times"], dtype=np.float64)
        if "times" in data.files
        else np.arange(frame_count, dtype=np.float64) / float(data["fps"])
    )

    # tet_indices: (M, 4) shared topology, or one array per frame.
    tets_shared = bool(tet_indices.dtype != object and tet_indices.ndim == 2)

    def positions_at(i):
        return np.asarray(positions[i], dtype=np.float32)

    def tets_at(i):
        return np.asarray(tet_indices if tets_shared else tet_indices[i], dtype=np.int32)

    return {
        "fps": float(data["fps"]),
        "episode": str(data["episode"]) if "episode" in data.files else "octopus",
        "start_frame": int(data["start_frame"]) if "start_frame" in data.files else 0,
        "frame_count": frame_count,
        "times": times,
        "tets_shared": tets_shared,
        "positions_at": positions_at,
        "tets_at": tets_at,
        "capsule_transform": capsule_transform,
        "capsule_radius": float(data["capsule_radius"]),
        "capsule_half_height": float(data["capsule_half_height"]),
    }


def _capsule_nodes(xform7, half_height):
    """The two capsule cap centres in world space (capsule long axis is local +Z,
    matching ``sim_octopus``)."""
    pos = np.asarray(xform7[:3], dtype=np.float64)
    rot = Rotation.from_quat(xform7[3:7]).as_matrix()
    axis = rot[:, 2] * half_height
    return np.array([pos - axis, pos + axis], dtype=np.float32)


def _frame_camera(pts, *, fill_fraction=0.7, view_dir=(1.0, -0.5, 1.0)):
    """Aim the camera at ``pts`` and back it off so the body fills the view
    (a trimmed copy of ``sim_octopus.frame_camera_on_soft_body``)."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6)
    try:
        fov_deg = ps.get_view_camera_parameters().get_fov_vertical_deg()
    except Exception:
        fov_deg = 45.0
    distance = radius / math.sin(math.radians(fov_deg) * 0.5) / max(fill_fraction, 1e-3)
    d = np.asarray(view_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    ps.look_at(tuple(center - d * distance), tuple(center))


class _Player:
    def __init__(self, rec, args):
        self.rec = rec
        self.fps = float(args.fps) if args.fps else rec["fps"]
        self.loop = not args.no_loop
        self.playing = not args.paused
        self.speed = 1.0
        self.frame = 0
        self._accum = 0.0
        self._nv = -1
        self._surf_tris_cache = {}

        self.show_tets = bool(getattr(args, "show_tets", False))
        self.ps_body = None
        nodes = _capsule_nodes(rec["capsule_transform"][0], rec["capsule_half_height"])
        self.ps_capsule = ps.register_curve_network(
            "poker capsule", nodes, np.array([[0, 1]], dtype=np.int32)
        )
        self.ps_capsule.set_radius(rec["capsule_radius"], relative=False)
        self.ps_capsule.set_color([0.85, 0.55, 0.15])

        # See-through overlay of the dataset's fused surface tracking, advanced
        # by each frame's recorded sim_time and offset by the recording's
        # start_frame (the tracked frame the sim's initial mesh was built from).
        self.tracked_surface = TrackedSurfaceOverlay(
            getattr(args, "tracked_surface", None),
            float(getattr(args, "tracked_surface_alpha", 0.3)),
            loop=self.loop,
            start_frame=rec.get("start_frame", 0),
            detrend=getattr(args, "tracked_surface_detrend", "rigid"),
        )
        if self.tracked_surface.active:
            try:
                ps.set_transparency_mode("pretty")
            except Exception:
                pass

        # Per-frame surface-tracking loss (sim surface -> nearest point on the
        # tracked mesh). Enabled with --surface-loss; the recording only stores
        # tets, so the surface triangulation is recovered per topology.
        self.surface_loss = None
        self._loss_by_frame = {}
        if getattr(args, "surface_loss", False) and self.tracked_surface.active:
            self.surface_loss = TrackedSurfaceLoss(
                self.tracked_surface,
                project_to_faces=not getattr(args, "surface_loss_no_project", False),
                use_valid_mask=not getattr(args, "surface_loss_keep_untracked", False),
                symmetric=bool(getattr(args, "surface_loss_symmetric", False)),
            )

        self._show_frame(0)
        _frame_camera(rec["positions_at"](0))
        ps.set_user_callback(self._gui)

    def _surface_tris(self, tets):
        """Boundary triangles of ``tets``, cached per topology (the recording
        only stores tets; the surface is recovered once per mesh size)."""
        tris = self._surf_tris_cache.get(tets.shape)
        if tris is None:
            tris = surface_triangles_from_tets(tets)
            self._surf_tris_cache[tets.shape] = tris
        return tris

    def _show_frame(self, i):
        rec = self.rec
        pts = rec["positions_at"](i)
        if self.ps_body is None or not rec["tets_shared"] or pts.shape[0] != self._nv:
            # First frame, or a changing topology (adaptive refinement) -- (re-)register.
            tets = rec["tets_at"](i)
            if self.show_tets:
                self.ps_body = ps.register_volume_mesh("Soft body", pts, tets)
            else:
                self.ps_body = ps.register_surface_mesh(
                    "Soft body", pts, self._surface_tris(tets)
                )
            self.ps_body.set_edge_width(1.0)
            self.ps_body.set_edge_color([0.0, 0.0, 0.0])
            self._nv = pts.shape[0]
        else:
            self.ps_body.update_vertex_positions(pts)
        self.ps_capsule.update_node_positions(
            _capsule_nodes(rec["capsule_transform"][i], rec["capsule_half_height"])
        )
        if self.tracked_surface.active:
            self.tracked_surface.update_for_time(rec["times"][i])
        self.frame = i

        if self.surface_loss is not None and i not in self._loss_by_frame:
            surf = np.unique(self._surface_tris(rec["tets_at"](i)))
            surf = surf[surf < len(pts)]
            row = self.surface_loss.evaluate(pts[surf], rec["times"][i])
            row["time"] = float(rec["times"][i])
            self._loss_by_frame[i] = row

    def _advance(self, dt):
        if not self.playing or self.rec["frame_count"] <= 1:
            return
        self._accum += dt * self.fps * self.speed
        steps = int(self._accum)
        if steps == 0:
            return
        self._accum -= steps
        nxt = self.frame + steps
        n = self.rec["frame_count"]
        if nxt >= n:
            if self.loop:
                nxt %= n
            else:
                nxt = n - 1
                self.playing = False
        self._show_frame(nxt)

    def _gui(self):
        rec = self.rec
        n = rec["frame_count"]
        psim.TextUnformatted(
            f"frame {self.frame + 1}/{n}   t = {rec['times'][self.frame]:.3f} s"
        )
        _, self.playing = psim.Checkbox("play", self.playing)
        psim.SameLine()
        if psim.Button("restart"):
            self._accum = 0.0
            self._show_frame(0)
        psim.SameLine()
        if psim.Button("step"):
            self.playing = False
            self._show_frame(min(self.frame + 1, n - 1))

        changed, new_frame = psim.SliderInt("scrub", self.frame, 0, max(n - 1, 0))
        if changed:
            self.playing = False
            self._show_frame(int(new_frame))

        _, self.speed = psim.SliderFloat("speed", self.speed, 0.05, 4.0, "%.2fx")
        _, self.loop = psim.Checkbox("loop", self.loop)
        psim.TextUnformatted(f"playback fps: {self.fps:g}")

        if self.surface_loss is not None:
            row = self._loss_by_frame.get(self.frame)
            if row is not None:
                extra = (
                    f"  sym {row['rmse_symmetric_mm']:.2f}"
                    if "rmse_symmetric_mm" in row else ""
                )
                psim.TextUnformatted(
                    f"surface loss -> tracked #{row['frame']}: "
                    f"RMSE {row['rmse_mm']:.2f} mm  max {row['max_mm']:.2f} mm{extra}"
                )
            if self._loss_by_frame:
                allr = np.array([r["rmse_mm"] for r in self._loss_by_frame.values()])
                psim.TextUnformatted(
                    f"  seen {len(allr)}/{n} frames  RMSE mm mean {allr.mean():.2f} "
                    f"max {allr.max():.2f}"
                )

        try:
            dt = float(psim.GetIO().DeltaTime)
        except Exception:
            dt = 1.0 / max(self.fps, 1e-6)
        if not (0.0 < dt < 0.25):
            dt = 1.0 / max(self.fps, 1e-6)
        self._advance(dt)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", nargs="?", default="simulation.npz",
        help="recording .npz written by sim_octopus.py --record (default: simulation.npz)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="playback frame rate (default: the capture fps stored in the file)",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="stop on the last frame instead of looping",
    )
    parser.add_argument(
        "--paused", action="store_true", help="start paused on frame 0",
    )
    parser.add_argument(
        "--show-tets", action="store_true",
        help="draw the full tet mesh instead of its boundary surface",
    )
    parser.add_argument(
        "--episode", default=None,
        help="PokeFlex episode the recording is from (default: the 'episode' "
             "key stored in the .npz, else octopus). Only used to pick the "
             "default --tracked-surface trajectory.",
    )
    # Dataset surface-tracking overlay (see mfem.refinement.tracked_surface).
    # Default --tracked-surface to a None sentinel so main() can resolve it from
    # the recording's episode.
    TrackedSurfaceOverlay.add_cli_args(parser)
    parser.set_defaults(tracked_surface=None)

    # Surface-tracking loss (see mfem.refinement.surface_loss).
    parser.add_argument(
        "--surface-loss", action="store_true",
        help="Score each replayed frame's surface against the nearest point on "
             "the tracked mesh (MSE); shown in the GUI and summarised at exit.",
    )
    parser.add_argument(
        "--surface-loss-no-project", action="store_true",
        help="Distance to the nearest tracked vertex instead of the tracked surface.",
    )
    parser.add_argument(
        "--surface-loss-keep-untracked", action="store_true",
        help="Keep untracked tracked vertices / faces in the target.",
    )
    parser.add_argument(
        "--surface-loss-symmetric", action="store_true",
        help="Also report the tracked -> sim direction (Chamfer-style).",
    )
    parser.add_argument(
        "--surface-loss-out", type=str, default=None,
        help="After the window closes, write the per-frame loss curve to this .npz.",
    )
    return parser


def main():
    args = create_parser().parse_args()
    rec = _load(args.path)
    episode_key = args.episode or rec["episode"]
    if args.tracked_surface is None:
        try:
            args.tracked_surface = get_episode(episode_key).tracked_surface_npy()
        except KeyError:
            args.tracked_surface = None
    print(
        f"{args.path}: episode {episode_key}, {rec['frame_count']} frames, "
        f"{rec['times'][-1] - rec['times'][0]:.2f} s, capture {rec['fps']:g} fps, "
        f"start frame {rec['start_frame']}"
    )

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
    playback_fps = float(args.fps) if args.fps else rec["fps"]
    ps.set_max_fps(int(round(max(playback_fps, 1.0))))

    player = _Player(rec, args)

    if player.surface_loss is not None:
        # Score every frame up front so a summary is available even if the
        # window is closed before playback visits them all.
        for i in range(rec["frame_count"]):
            player._show_frame(i)
        player._show_frame(0)

    ps.show()

    if player.surface_loss is not None and player._loss_by_frame:
        rows = [player._loss_by_frame[i] for i in sorted(player._loss_by_frame)]
        keys = rows[0].keys()
        curve = {k: np.array([r[k] for r in rows]) for k in keys}
        rmse_mm = curve["rmse_mm"]
        print(
            f"[surface loss] {len(rmse_mm)} frames  RMSE mm  "
            f"mean {rmse_mm.mean():.2f}  min {rmse_mm.min():.2f}  "
            f"max {rmse_mm.max():.2f}   (frame-mean MSE {curve['mse'].mean():.3e} m^2)"
        )
        if args.surface_loss_out:
            np.savez_compressed(args.surface_loss_out, **curve)
            print(f"[surface loss] wrote {args.surface_loss_out}")


if __name__ == "__main__":
    main()
