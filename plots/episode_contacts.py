"""One frame per PokeFlex episode, tool at its deepest point of contact (IEEE figure).

Renders the coarse-mesh-plus-refinement recording (``octopus_refinement.py --record -r``,
the same recordings ``refinement_loss.py`` scores as "with refinement") for
each of the four tracked episodes -- octopus, dice, turtle, tp_roll -- and lays out
one panel per episode in a grid, each showing the poker capsule touching the body.
Unlike ``refinement_frames.py`` (one object, several mesh conditions / poke
phases, one shared camera), this is a gallery of four different objects, so each
panel gets its own camera fitted to its own body and is cropped independently.

The contact frame is auto-picked per episode as the recording frame minimizing the
capsule tip's signed distance to the nearest body vertex (deepest penetration /
closest approach), skipping frames too close to a refinement event (the mesh just
changed topology, so the render can visibly pop) or with a localized per-vertex
displacement spike (a sign of solver instability rather than the poke's own smooth
motion) -- see ``deepest_contact_frame``. Overridable per episode with ``--frame``.
Same rendering style as the other IEEE figures: clay-shaded grey body with black
wireframe, orange poker, Times-family serif, no in-figure title, subfigure letters
below the panels, PDF next to the PNG. Needs a display for polyscope (a window
opens briefly). Usage::

    uv run python -m plots.episode_contacts --out notebooks/episode_contacts_ieee.png
    uv run python -m plots.episode_contacts --grid 1 4 --out ...
    uv run python -m plots.episode_contacts --frame octopus 28 --out ...
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from plots.step_timing import ROOT, COLUMN_WIDTH, ieee_rcparams, save_figure
from plots.refinement_frames import (
    capsule_geometry, content_mask, flatten_on_white, load_recording,
)
from examples.refinement.surface_loss import surface_triangles_from_tets

# Coarse mesh + refinement recordings (the benchmark-best geometric-scoring setting;
# "r1" of the 3 repeats refinement_loss.py bands together for the new episodes).
RECORDINGS = {
    "octopus": "results/sweep_refine_geometric_work/rec_refine_geometric_threshold=2.0_refine_min_edge_length=0.004.npz",
    "dice": "results/refinement_loss_runs/dice_refine_r1.npz",
    "turtle": "results/refinement_loss_runs/turtle_refine_r1.npz",
    "tp_roll": "results/refinement_loss_runs/tp_roll_refine_r1.npz",
}
EPISODE_LABEL = {"octopus": "Octopus", "dice": "Dice", "turtle": "Turtle", "tp_roll": "Paper roll"}
DEFAULT_ORDER = ("octopus", "dice", "turtle", "tp_roll")

BODY_COLOR = (0.86, 0.85, 0.80)
EDGE_COLOR = (0.05, 0.05, 0.05)
TOOL_COLOR = (0.85, 0.55, 0.15)


# --------------------------------------------------------------------------
# Contact-frame detection
# --------------------------------------------------------------------------
def capsule_front(xform7, half_height):
    """World position of the poking (+Z) cap -- see capsule_geometry's docstring."""
    pos = np.asarray(xform7[:3], dtype=np.float64)
    axis = Rotation.from_quat(xform7[3:7]).as_matrix()[:, 2]
    return pos + axis * half_height


def topology_change_frames(rec):
    """Frames where the vertex count differs from the previous frame -- a
    refinement event under ``--refine`` (fixed-topology recordings return [])."""
    changes = []
    prev_n = rec["positions_at"](0).shape[0]
    for f in range(1, rec["frame_count"]):
        n = rec["positions_at"](f).shape[0]
        if n != prev_n:
            changes.append(f)
        prev_n = n
    return changes


def worst_tet_volume_ratio(rec, f):
    """Smallest signed tet volume at frame ``f``, as a fraction of that frame's own
    median (absolute) tet volume. Negative means at least one element has flipped
    inside-out -- the direct geometric signature of a solver blow-up -- while a
    small but positive ratio is just an ordinary thin/sliver element and not a
    defect. This is deliberately *not* based on vertex displacement: the poke
    itself legitimately moves only the handful of vertices right under the tip in
    a single frame (the elastic wave hasn't reached the rest of the body yet),
    which would look exactly like a displacement spike while being completely
    physical. These "coarse + refinement" recordings do have real, extended
    inverted-element stretches after some refinement events (as much as an entire
    recording's back half, e.g. tp_roll_refine_r1 from frame ~84 on) -- this is
    what actually needs filtering out, not fast contact motion."""
    p = rec["positions_at"](f)
    tets = rec["tets_at"](f)
    v0, v1, v2, v3 = p[tets[:, 0]], p[tets[:, 1]], p[tets[:, 2]], p[tets[:, 3]]
    vol = np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
    med = np.median(np.abs(vol))
    return float(vol.min() / med) if med > 0 else float(vol.min())


def deepest_contact_frame(rec, *, avoid_window=0, quality_thresh=-0.5, verbose=True):
    """Recording frame minimizing the capsule tip's signed distance to the nearest
    body vertex (most negative = deepest penetration, most positive = closest
    approach if the poke never actually penetrates), skipping frames within
    ``avoid_window`` of a refinement event (topology just changed) or with a badly
    inverted element (``worst_tet_volume_ratio`` below ``quality_thresh``).

    ``quality_thresh`` is deliberately not 0: in these recordings a coarse element
    right under the rigid poker commonly dips slightly negative from ordinary
    contact compression without there being any visible defect (it is a small
    interior sliver, not a surface fold), and requiring a perfectly non-negative
    mesh at every actual contact frame turns out to exclude *all* of them for
    every episode -- there is no frame in these recordings where the tool visibly
    touches the body and every tet is non-inverted. -0.5 keeps that everyday
    contact compression while still excluding the large (multi-unit-ratio),
    multi-frame blow-ups these coarse+refinement runs sometimes have after a
    refinement event (e.g. tp_roll_refine_r1 goes to -34 x median volume around
    frame 120 and never recovers by the end of the recording)."""
    changes = topology_change_frames(rec)

    def unstable(f):
        if any(abs(f - c) <= avoid_window for c in changes):
            return True
        return worst_tet_volume_ratio(rec, f) < quality_thresh

    best_f, best_d, skipped = None, math.inf, []
    for f in range(rec["frame_count"]):
        if unstable(f):
            skipped.append(f)
            continue
        p = rec["positions_at"](f)
        tip = capsule_front(rec["capsule"][f], rec["half_height"])
        d = float(np.linalg.norm(p - tip, axis=1).min()) - rec["radius"]
        if d < best_d:
            best_f, best_d = f, d
    if best_f is None:   # every frame flagged; fall back to the unfiltered best
        best_f = min(range(rec["frame_count"]), key=lambda f: np.linalg.norm(
            rec["positions_at"](f) - capsule_front(rec["capsule"][f], rec["half_height"]), axis=1).min())
    if verbose and skipped:
        print(f"[plot]   skipped {len(skipped)} unstable/near-refinement frame(s): {skipped}")
    return best_f


# --------------------------------------------------------------------------
# polyscope rendering: one independently-framed, independently-cropped panel
# --------------------------------------------------------------------------
def init_polyscope(*, pixels, ssaa):
    """ps.init() is meant to be called exactly once per process -- re-initializing
    it per panel (as a naive port of refinement_frames' per-frame loop would)
    hangs polyscope on the second call. Camera + geometry are set up fresh per
    panel; only this one-time window/backend setup lives here."""
    import polyscope as ps

    ps.set_program_name("episode contact frame")
    ps.init()
    ps.set_window_size(pixels, pixels)
    ps.set_build_gui(False)
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((1.0, 1.0, 1.0))
    try:
        ps.set_SSAA_factor(ssaa)
    except Exception:
        pass


def render_panel(rec, frame, *, view_dir, fill, edge_width):
    import polyscope as ps

    p = rec["positions_at"](frame)
    tets = rec["tets_at"](frame)
    tris = surface_triangles_from_tets(tets)
    caps, shaft = capsule_geometry(rec["capsule"][frame], rec["half_height"])

    pts = np.concatenate([p, caps])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))

    fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
    dist = radius / math.sin(math.radians(fov) * 0.5) / fill
    d = np.asarray(view_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    ps.look_at(tuple(center - d * dist), tuple(center), fly_to=False)

    body = ps.register_surface_mesh("body", p, tris, smooth_shade=False)
    body.set_color(BODY_COLOR)
    body.set_edge_width(edge_width)
    body.set_edge_color(EDGE_COLOR)
    body.set_material("clay")

    tool = ps.register_curve_network("tool", caps, np.array([[0, 1]], dtype=np.int32))
    tool.set_radius(rec["radius"], relative=False)
    tool.set_color(TOOL_COLOR)
    tool.set_material("clay")
    rod = ps.register_curve_network("shaft", shaft, np.array([[0, 1]], dtype=np.int32))
    rod.set_radius(0.25 * rec["radius"], relative=False)
    rod.set_color(TOOL_COLOR)
    rod.set_material("clay")

    buf = ps.screenshot_to_buffer(transparent_bg=True)
    im = np.asarray(buf, dtype=np.uint8).copy()
    ps.remove_all_structures()

    ys, xs = np.nonzero(content_mask(im))
    if not ys.size:
        return im
    pad = 0.03 * max(ys.max() - ys.min(), xs.max() - xs.min())
    y0, x0 = max(int(ys.min() - pad), 0), max(int(xs.min() - pad), 0)
    y1, x1 = min(int(ys.max() + pad) + 1, im.shape[0]), min(int(xs.max() + pad) + 1, im.shape[1])
    if y0 <= 0 or x0 <= 0 or y1 >= im.shape[0] or x1 >= im.shape[1]:
        print("[plot] warning: panel touches the render border; lower --fill")
    return im[y0:y1, x0:x1]


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot(order, images, verts, out: Path, *, dpi, column, grid, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(ieee_rcparams(column))
    width = COLUMN_WIDTH[column]
    nrow, ncol = grid
    fs = plt.rcParams["font.size"]
    gap = 0.08                # inches between panels, both axes
    letter_h = 0.28           # inches reserved below each panel for caption + letter

    aspect = np.array([im.shape[0] / im.shape[1] for im in images], dtype=float)  # h/w per panel
    col_w = (width - (ncol - 1) * gap) / ncol
    row_h = col_w * aspect.reshape(nrow, ncol).max(axis=1)   # tallest panel sets its row's height
    fig_h = row_h.sum() + (nrow - 1) * gap + nrow * letter_h
    fig = plt.figure(figsize=(width, fig_h))

    letters = "abcdefghijkl"
    for i, (key, im) in enumerate(zip(order, images)):
        r, c = divmod(i, ncol)
        panel_h = col_w * (im.shape[0] / im.shape[1])
        left = c * (col_w + gap)
        row_top = row_h[:r].sum() + r * (gap + letter_h)
        bottom = fig_h - (row_top + row_h[r])    # bottom-align inside the row: one caption line per row
        ax = fig.add_axes([left / width, bottom / fig_h, col_w / width, panel_h / fig_h])
        ax.imshow(flatten_on_white(im), interpolation="lanczos")
        ax.set_axis_off()
        caption = f"({letters[i]}) {EPISODE_LABEL[key]}"
        if counts:
            caption += f", {verts[i]} vertices"
        fig.text((left + 0.5 * col_w) / width, (bottom - 0.05) / fig_h,
                 caption, ha="center", va="top", fontsize=fs)
    save_figure(fig, out, dpi)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="episode_contacts.png", help="output PNG path (PDF written alongside)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--column", choices=list(COLUMN_WIDTH), default="double")
    ap.add_argument("--order", nargs="+", default=list(DEFAULT_ORDER), choices=list(RECORDINGS),
                    help="episodes, row-major through the grid (default: octopus dice turtle tp_roll)")
    ap.add_argument("--grid", nargs=2, type=int, default=(2, 2), metavar=("ROWS", "COLS"))
    for key in RECORDINGS:
        ap.add_argument(f"--{key}", default=RECORDINGS[key], metavar="NPZ", help=f"recording for '{key}'")
    ap.add_argument("--frame", nargs=2, action="append", default=[], metavar=("EPISODE", "F"),
                    help="override the auto-picked contact frame for one episode (repeatable)")
    ap.add_argument("--avoid-window", type=int, default=0,
                    help="frames to skip on either side of a refinement event (default 0: the event frame only)")
    ap.add_argument("--quality-thresh", type=float, default=-0.5,
                    help="flag a frame when its worst signed tet volume, as a fraction of the "
                         "frame's median tet volume, drops below this (default -0.5)")
    ap.add_argument("--view-dir", nargs=3, type=float, default=(1.0, -0.62, 0.85), metavar="D",
                    help="camera looks along -D at each body (default 1 -0.62 0.85)")
    ap.add_argument("--fill", type=float, default=1.05, help="fraction of the view each panel's scene fills")
    ap.add_argument("--pixels", type=int, default=1400, help="polyscope render size (square)")
    ap.add_argument("--ssaa", type=int, default=2, help="polyscope supersampling factor")
    ap.add_argument("--edge-width", type=float, default=1.0)
    ap.add_argument("--panels-dir", default=None, help="also write each panel as PNG here")
    ap.add_argument("--no-counts", action="store_true", help="omit the per-panel vertex count")
    args = ap.parse_args(argv)

    if len(args.order) != args.grid[0] * args.grid[1]:
        raise SystemExit(f"--order has {len(args.order)} episodes but --grid is {args.grid[0]}x{args.grid[1]}")
    frame_overrides = {ep: int(f) for ep, f in args.frame}

    recs = {key: load_recording(getattr(args, key)) for key in args.order}
    frames = {}
    for key, rec in recs.items():
        if key in frame_overrides:
            f = frame_overrides[key]
            if not 0 <= f < rec["frame_count"]:
                raise SystemExit(f"frame {f} out of range for {key} ({rec['frame_count']} frames)")
        else:
            f = deepest_contact_frame(rec, avoid_window=args.avoid_window, quality_thresh=args.quality_thresh)
        frames[key] = f
        print(f"[plot] {key}: contact frame {f}/{rec['frame_count']} (t={rec['times'][f]:.2f}s)")

    init_polyscope(pixels=args.pixels, ssaa=args.ssaa)
    images, verts = [], []
    for key in args.order:
        im = render_panel(recs[key], frames[key], view_dir=args.view_dir, fill=args.fill,
                          edge_width=args.edge_width)
        images.append(im)
        verts.append(recs[key]["positions_at"](frames[key]).shape[0])

    if args.panels_dir:
        from PIL import Image
        pdir = Path(args.panels_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for key, im in zip(args.order, images):
            p = pdir / f"{key}_contact_f{frames[key]:03d}.png"
            Image.fromarray(im).save(p)
            print(f"[plot] wrote {p}")

    plot(args.order, images, verts, Path(args.out), dpi=args.dpi, column=args.column,
         grid=tuple(args.grid), counts=not args.no_counts)


if __name__ == "__main__":
    main()
