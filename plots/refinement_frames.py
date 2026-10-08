"""Before / during / after-contact frames of the octopus poke (IEEE figure).

Renders the recorded soft-body surface (``octopus_refinement.py --record`` .npz, the same
benchmark recordings as ``refinement_loss.py``) with polyscope at three episode
frames and lays them out as a rows x 3 grid: one row per mesh condition (coarse,
coarse + refinement, optionally the fine mesh), one column per phase of the poke.
The surface wireframe is drawn so the refinement of the contact patch is visible; the
poker capsule is drawn in orange. Camera, crop and scale are identical in every panel.

Default frames (recording index, 30 fps, sim starts at tracked frame 19):

  before  0    poker hovering ~5 mm above the plush, initial 601-vertex mesh
  during  20   deepest point of the first indentation (poker tip lowest)
  after   140  poker lifted ~55 mm clear, plush relaxed, refinement complete

Formatted like ``step_timing.py`` (Times-family serif, IEEE column width, no
in-figure title, subfigure letters below the panels, PDF next to the PNG). Needs a
display for polyscope (a window opens briefly); every panel is also written as its own
PNG under ``--panels-dir`` for hand layout. Usage::

    uv run python -m plots.refinement_frames --out notebooks/refinement_frames_octopus_ieee.png
    uv run python -m plots.refinement_frames --rows coarse refine fine --out ...
    uv run python -m plots.refinement_frames --frames 0 55 140 ...
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from plots.step_timing import ROOT, COLUMN_WIDTH, ieee_rcparams, save_figure
from examples.refinement.surface_loss import surface_triangles_from_tets

RECORDINGS = {
    "coarse": "results/sweep_refine_legacy_work/rec_norefine.npz",
    "refine": "results/sweep_refine_geometric_work/rec_refine_geometric_threshold=2.0_refine_min_edge_length=0.004.npz",
    "fine": "results/reference_runs/fine_default.npz",
}
ROW_LABEL = {"coarse": "Coarse mesh", "refine": "Coarse + refinement", "fine": "Fine mesh"}
PHASES = ("Before contact", "During contact", "After contact")
DEFAULT_FRAMES = (0, 20, 140)

BODY_COLOR = (0.86, 0.85, 0.80)
EDGE_COLOR = (0.05, 0.05, 0.05)
TOOL_COLOR = (0.85, 0.55, 0.15)


# --------------------------------------------------------------------------
# Recording access
# --------------------------------------------------------------------------
def load_recording(path):
    d = np.load(ROOT / path if not os.path.isabs(path) else path, allow_pickle=True)
    pos, tets = d["positions"], d["tet_indices"]
    shared_tets = tets.dtype != object and tets.ndim == 2

    def positions_at(i):
        return np.asarray(pos[i], dtype=np.float32)

    def tets_at(i):
        return np.asarray(tets if shared_tets else tets[i], dtype=np.int32)

    return {
        "positions_at": positions_at, "tets_at": tets_at,
        "times": np.asarray(d["times"], dtype=np.float64),
        "capsule": np.asarray(d["capsule_transform"], dtype=np.float64),
        "radius": float(d["capsule_radius"]), "half_height": float(d["capsule_half_height"]),
        "frame_count": int(d["frame_count"]),
    }


def capsule_geometry(xform7, half_height, shaft_len=0.03):
    """Cap centres of the poker capsule plus a thin shaft leaving its back cap
    (capsule long axis = local +Z, +Z toward the poke, as in ``octopus_refinement``)."""
    pos = np.asarray(xform7[:3], dtype=np.float64)
    axis = Rotation.from_quat(xform7[3:7]).as_matrix()[:, 2]
    caps = np.array([pos - axis * half_height, pos + axis * half_height], dtype=np.float32)
    back = pos - axis * half_height
    shaft = np.array([back, back - axis * shaft_len], dtype=np.float32)
    return caps, shaft


# --------------------------------------------------------------------------
# polyscope rendering
# --------------------------------------------------------------------------
def render_panels(recs, frames, *, view_dir, fill, pixels, ssaa, edge_width):
    """Return {(row, frame): (H, W, 4) uint8 RGBA} screenshots, one fixed camera."""
    import polyscope as ps

    ps.set_program_name("refinement frames")
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

    # One camera for every panel: frame the union of all rendered bodies + capsules.
    pts = []
    for key, rec in recs.items():
        for f in frames:
            pts.append(rec["positions_at"](f))
            pts.append(capsule_geometry(rec["capsule"][f], rec["half_height"])[0])
    pts = np.concatenate(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
    dist = radius / math.sin(math.radians(fov) * 0.5) / fill
    d = np.asarray(view_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    ps.look_at(tuple(center - d * dist), tuple(center), fly_to=False)

    out = {}
    tris_cache = {}
    for key, rec in recs.items():
        for f in frames:
            p = rec["positions_at"](f)
            tets = rec["tets_at"](f)
            tris = tris_cache.get((key, tets.shape))
            if tris is None:
                tris = surface_triangles_from_tets(tets)
                tris_cache[(key, tets.shape)] = tris
            body = ps.register_surface_mesh("body", p, tris, smooth_shade=False)
            body.set_color(BODY_COLOR)
            body.set_edge_width(edge_width)
            body.set_edge_color(EDGE_COLOR)
            body.set_material("clay")

            caps, shaft = capsule_geometry(rec["capsule"][f], rec["half_height"])
            tool = ps.register_curve_network("tool", caps, np.array([[0, 1]], dtype=np.int32))
            tool.set_radius(rec["radius"], relative=False)
            tool.set_color(TOOL_COLOR)
            tool.set_material("clay")
            rod = ps.register_curve_network("shaft", shaft, np.array([[0, 1]], dtype=np.int32))
            rod.set_radius(0.25 * rec["radius"], relative=False)
            rod.set_color(TOOL_COLOR)
            rod.set_material("clay")

            buf = ps.screenshot_to_buffer(transparent_bg=True)
            out[(key, f)] = np.asarray(buf, dtype=np.uint8).copy()
            ps.remove_all_structures()
    return out


def content_mask(im):
    """Pixels that are not the white background (polyscope's screenshot ignores the
    transparent-background request, so the alpha channel is unusable)."""
    return (im[..., :3].astype(np.int16) < 250).any(axis=-1) & (im[..., 3] > 8)


def column_crop(images, frames, pad_frac=0.015):
    """Crop each column (one frame, all rows) to the union bounding box of its opaque
    pixels. Every panel keeps the same pixel scale (one camera), but columns get their
    own box so the figure does not carry the union whitespace of all frames."""
    h, w = next(iter(images.values())).shape[:2]
    out = {}
    for f in frames:
        lo = np.array([np.inf, np.inf])
        hi = np.array([-np.inf, -np.inf])
        for (key, ff), im in images.items():
            if ff != f:
                continue
            ys, xs = np.nonzero(content_mask(im))
            if ys.size:
                lo = np.minimum(lo, [ys.min(), xs.min()])
                hi = np.maximum(hi, [ys.max(), xs.max()])
        pad = pad_frac * max(hi - lo)
        y0, x0 = np.maximum(lo - pad, 0).astype(int)
        y1, x1 = np.minimum(hi + pad, [h - 1, w - 1]).astype(int) + 1
        if lo.min() <= 0 or hi[0] >= h - 1 or hi[1] >= w - 1:
            print(f"[plot] warning: frame {f} touches the render border; lower --fill")
        for (key, ff), im in images.items():
            if ff == f:
                out[(key, ff)] = im[y0:y1, x0:x1]
    return out


def flatten_on_white(im):
    a = im[..., 3:4].astype(np.float32) / 255.0
    rgb = im[..., :3].astype(np.float32) / 255.0
    return rgb * a + (1.0 - a)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot(recs, images, rows, frames, out: Path, *, dpi, column, title, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(ieee_rcparams(column))
    single = column == "single"
    width = COLUMN_WIDTH[column]
    nrow, ncol = len(rows), len(frames)
    fs = plt.rcParams["font.size"]
    row_label_w = 0.13                      # inches reserved for the rotated row labels
    col_gap = 0.04                          # inches between columns
    header_h = 0.0 if title else 0.15       # column headers (phase + time)
    letter_h = 0.14                         # "(a)" line under each panel
    # Column widths follow the per-column crop; one pixel scale for the whole figure.
    px_w = np.array([images[(rows[0], f)].shape[1] for f in frames], dtype=float)
    px_h = np.array([images[(rows[0], f)].shape[0] for f in frames], dtype=float)
    scale = (width - row_label_w - (ncol - 1) * col_gap) / px_w.sum()   # inches per pixel
    col_w = px_w * scale
    panel_h = float(px_h.max() * scale)
    fig_h = header_h + nrow * (panel_h + letter_h)
    fig = plt.figure(figsize=(width, fig_h))

    times = recs[rows[0]]["times"]
    letters = "abcdefghijkl"
    for r, key in enumerate(rows):
        row_top = header_h + r * (panel_h + letter_h)
        for c, f in enumerate(frames):
            im = images[(key, f)]
            ih = im.shape[0] * scale
            left = row_label_w + col_w[:c].sum() + c * col_gap
            # bottom-align inside the row: the plush rests on the same ground in every frame
            bottom = fig_h - (row_top + panel_h)
            ax = fig.add_axes([left / width, bottom / fig_h, col_w[c] / width, ih / fig_h])
            ax.imshow(flatten_on_white(im), interpolation="lanczos")
            ax.set_axis_off()
            caption = f"({letters[r * ncol + c]})"
            if counts:   # vertex count on the subfigure line, so the render stays clean
                caption += f" {recs[key]['positions_at'](f).shape[0]} vertices"
            if r == 0:
                fig.text((left + 0.5 * col_w[c]) / width, 1.0 - 0.45 * header_h / fig_h,
                         f"{PHASES[c]} ($t$ = {times[f]:.2f} s)", ha="center", va="center", fontsize=fs)
            fig.text((left + 0.5 * col_w[c]) / width, (bottom - 0.05) / fig_h,
                     caption, ha="center", va="top", fontsize=fs)
        y_mid = fig_h - (row_top + 0.5 * panel_h)
        fig.text(0.45 * row_label_w / width, y_mid / fig_h, ROW_LABEL[key], rotation=90,
                 ha="center", va="center", fontsize=fs)
    save_figure(fig, out, dpi)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="refinement_frames_octopus.png", help="output PNG path (PDF written alongside)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--column", choices=list(COLUMN_WIDTH), default="double")
    ap.add_argument("--rows", nargs="+", default=["coarse", "refine"], choices=list(RECORDINGS),
                    help="mesh conditions, top to bottom (default: coarse refine)")
    ap.add_argument("--frames", nargs=3, type=int, default=list(DEFAULT_FRAMES), metavar="F",
                    help="recording frame indices for before / during / after (default 0 20 140)")
    for key in RECORDINGS:
        ap.add_argument(f"--{key}", default=RECORDINGS[key], metavar="NPZ", help=f"recording for the '{key}' row")
    ap.add_argument("--view-dir", nargs=3, type=float, default=(1.0, -0.62, 0.85), metavar="D",
                    help="camera looks along -D at the body (default 1 -0.62 0.85)")
    ap.add_argument("--fill", type=float, default=1.05, help="fraction of the view the scene fills")
    ap.add_argument("--pixels", type=int, default=1400, help="polyscope render size (square)")
    ap.add_argument("--ssaa", type=int, default=2, help="polyscope supersampling factor")
    ap.add_argument("--edge-width", type=float, default=1.0)
    ap.add_argument("--panels-dir", default=None, help="also write each panel as PNG here")
    ap.add_argument("--no-counts", action="store_true", help="omit the per-panel vertex count")
    ap.add_argument("--title", action="store_true", help="(unused; kept for parity with the other plot scripts)")
    args = ap.parse_args(argv)

    recs = {key: load_recording(getattr(args, key)) for key in args.rows}
    for key, rec in recs.items():
        for f in args.frames:
            if not 0 <= f < rec["frame_count"]:
                raise SystemExit(f"frame {f} out of range for {key} ({rec['frame_count']} frames)")

    images = render_panels(recs, args.frames, view_dir=args.view_dir, fill=args.fill,
                           pixels=args.pixels, ssaa=args.ssaa, edge_width=args.edge_width)
    images = column_crop(images, args.frames)

    if args.panels_dir:
        from PIL import Image
        pdir = Path(args.panels_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for (key, f), im in images.items():
            p = pdir / f"octopus_{key}_f{f:03d}.png"
            Image.fromarray(im).save(p)
            print(f"[plot] wrote {p}")

    plot(recs, images, args.rows, args.frames, Path(args.out), dpi=args.dpi, column=args.column,
         title=args.title, counts=not args.no_counts)


if __name__ == "__main__":
    main()
