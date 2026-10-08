"""Close-up before/after screenshots of a refinement event on the octopus
surface, edges colored by refinement score (paper/debug figure).

Runs the live octopus sim (examples.refinement.octopus_refinement, contact-driven
"geometric" scoring by default) and steps it one frame at a time. Each frame,
before stepping, the current topology is cloned and its edges are scored
with mfem.refinement.refinement.edge_refinement_scores -- the same per-edge
score octopus_refinement.py's live "Refinement scores" overlay uses, taken from
refine()'s populate_candidates* pass on *that* topology. "before" is the
starting frame; "after" is a later frame once the mesh has gained at least
--min-vertex-gain vertices (spanning many refinement passes, not just the
first one, for a visibly bigger vertex-count difference between panels).
Both are rendered as a close-up around the poker tip, edges drawn as a curve
network colored by score, capsule translucent.

A recorded (octopus_refinement.py --record) .npz has no score data -- only
positions/tets -- so this needs a live run instead, not the older
refinement_frames-style recording playback.

Needs a display for polyscope (a window opens briefly).

Run with::

    uv run python -m examples.refinement.octopus_refinement_closeup
    uv run python -m examples.refinement.octopus_refinement_closeup --episode octopus --out-prefix <path>
"""
from __future__ import annotations

from examples.config import apply_config

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BODY_COLOR = (0.30, 0.55, 0.80)
TOOL_COLOR = (0.85, 0.55, 0.15)
SCORE_CMAP = "reds"


def capsule_transform_matrix(xform7) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(xform7[3:7]).as_matrix()
    mat[:3, 3] = xform7[:3]
    return mat


def build_sim(argv):
    """Construct a live MFEMRefinementSim exactly like octopus_refinement.py's own
    __main__ block, but from an explicit argv instead of sys.argv, and
    without entering its interactive run() loop -- we drive .step() /
    .render() ourselves below."""
    import warp as wp
    from examples.refinement import octopus_refinement as so

    old_argv = sys.argv
    sys.argv = ["octopus_refinement_closeup"] + list(argv)
    try:
        parser = so.MFEMRefinementSim.create_parser()
        args = so.init(parser)
    finally:
        sys.argv = old_argv

    episode = so.get_episode(args)
    mesh_path = episode.mesh_paths[args.mesh]
    refinement_model = so.MFEMRefinementModel.load(mesh_path, 1.0, wp.vec3(0.0, 0.0, 0.0))
    sim = so.MFEMRefinementSim(refinement_model, args)
    return sim, args


def setup_polyscope(pixels: int, ssaa: int) -> None:
    """One-time polyscope display config. ps.init() itself already ran inside
    build_sim() -> octopus_refinement.init() (non-headless => a real GL window), so
    this must not call it again."""
    import polyscope as ps

    ps.set_window_size(pixels, pixels)
    ps.set_build_gui(False)
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_transparency_mode("pretty")
    try:
        ps.set_SSAA_factor(ssaa)
    except Exception:
        pass


def collect_before_after(sim, max_frames: int, min_vertex_gain: int):
    """Step ``sim`` frame by frame (--substeps 1 => exactly one refine() call
    per frame), snapshotting each frame's pre-step topology + edge scores.
    "before" is frame 0 (the starting mesh); "after" is the first later frame
    whose vertex count has grown by at least ``min_vertex_gain`` over
    "before" (or, failing that, the last frame reached by ``max_frames`` --
    spanning many refinement passes instead of stopping at the very first
    one gives a much more visible vertex-count difference between panels."""
    from mfem.refinement.refinement import edge_refinement_scores

    before = None
    latest = None
    for i in range(max_frames):
        add0 = sim.solver._additional_state_0
        n = int(add0.active_particle_count.numpy()[0])
        tri_ct = int(add0.active_tri_count.numpy()[0])
        pre_state = add0.clone()
        pre_particle_q = sim.state_0.particle_q[:n].numpy().copy()
        pre_tris = add0.tri_indices[:tri_ct].numpy().copy()
        pre_capsule = sim.state_0.body_q.numpy()[0].copy()

        sim.step()

        edges, scores = edge_refinement_scores(pre_state, sim.solver._refine_buffers)
        latest = dict(
            frame=i, particle_q=pre_particle_q, tris=pre_tris,
            edges=edges, scores=scores, capsule_xform=pre_capsule,
        )
        if before is None:
            before = latest
        print(f"[octopus closeup] frame {i}: {n} vertices, "
              f"max edge score {scores.max() if scores.size else 0.0:.3g}")

        if n - before["particle_q"].shape[0] >= min_vertex_gain:
            return before, latest

    print(f"[octopus closeup] only gained {latest['particle_q'].shape[0] - before['particle_q'].shape[0]} "
          f"vertices in {max_frames} frames (wanted {min_vertex_gain}); using the last frame reached")
    return before, latest


def surface_edge_scores(tris: np.ndarray, edges: np.ndarray, scores: np.ndarray):
    """Restrict edge_refinement_scores()' full tet-edge list to the surface
    triangulation. Interior tet edges are never visible through the opaque
    body from outside anyway, and drawing all of them (a curve network with
    6 edges per tet) triggered a polyscope OIT compositing glitch once the
    translucent capsule was also in the scene -- stray interior edges bled
    through the body as a dense tangle instead of staying occluded."""
    score_by_edge = {}
    for (a, b), s in zip(edges.tolist(), scores.tolist()):
        key = (a, b) if a < b else (b, a)
        score_by_edge[key] = s  # duplicates (one per incident tet) share the same canonical-edge score

    surf_keys = set()
    for a, b, c in tris.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            surf_keys.add((u, v) if u < v else (v, u))
    surf_keys = sorted(surf_keys)

    surf_edges = np.array(surf_keys, dtype=np.int32)
    surf_scores = np.array([score_by_edge.get(k, 0.0) for k in surf_keys], dtype=np.float32)
    return surf_edges, surf_scores


def render_closeup(before, after, capsule_mesh, capsule_alpha: float, close_radius: float,
                    out_prefix: str) -> dict[str, np.ndarray]:
    import polyscope as ps

    # A TrackedSurfaceOverlay structure is auto-registered at sim construction
    # time (see MFEMRefinementSim.__init__); we never call sim.render() to
    # refresh/hide it, so it would otherwise show up as a stray green patch
    # in the first screenshot below.
    ps.remove_all_structures()

    # Center the close-up on the poker tip (local +Z cap -- "+Z toward the
    # poke", see refinement_frames.capsule_geometry) at the "after" pose.
    xform7 = after["capsule_xform"]
    tip_axis = Rotation.from_quat(xform7[3:7]).as_matrix()[:, 2]
    center = np.asarray(xform7[:3], dtype=np.float64) + tip_axis * capsule_mesh.half_height

    # Mostly top-down (dominant -Y component => camera sits high above center
    # looking down), with a little tilt in X/Z so the capsule doesn't hide
    # the patch directly beneath it and the view still reads as 3D.
    view_dir = np.array([0.2, -0.9, 0.35])
    view_dir /= np.linalg.norm(view_dir)
    fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
    dist = close_radius / math.sin(math.radians(fov) * 0.5) / 0.85
    ps.look_at(tuple(center - view_dir * dist), tuple(center), fly_to=False)

    surf = {tag: surface_edge_scores(snap["tris"], snap["edges"], snap["scores"])
            for tag, snap in (("before", before), ("after", after))}

    # Scores span ~2 orders of magnitude (a handful of edges near the tool
    # score far above the rest of the mesh), so coloring the raw value makes
    # everything but those few edges look identical at the low end. log1p
    # compresses that range so the gradient leading up to the split edge
    # stays visible instead of being all-one-color.
    for tag in surf:
        edges, scores = surf[tag]
        surf[tag] = (edges, np.log1p(scores))

    # Common (log-scaled) score range across both panels so the same color
    # means the same score in "before" and "after".
    all_scores = np.concatenate([s for _, s in surf.values()])
    vmax = float(all_scores.max()) if all_scores.size and all_scores.max() > 0 else 1.0

    from PIL import Image

    out = {}
    for tag, snap in (("before", before), ("after", after)):
        surf_edges, surf_scores = surf[tag]

        body = ps.register_surface_mesh("body", snap["particle_q"], snap["tris"], smooth_shade=False)
        body.set_color(BODY_COLOR)
        body.set_edge_width(0.0)  # edges are drawn by the colored curve network below instead
        body.set_material("clay")

        edges_net = ps.register_curve_network("edge scores", snap["particle_q"], surf_edges)
        edges_net.set_radius(0.0006, relative=False)
        edges_net.add_scalar_quantity(
            "log(1 + score)", surf_scores, defined_on="edges", enabled=True,
            cmap=SCORE_CMAP, vminmax=(0.0, vmax),
        )

        capsule = ps.register_surface_mesh(
            "capsule", capsule_mesh.vertices, capsule_mesh.indices.reshape(-1, 3)
        )
        capsule.set_color(TOOL_COLOR)
        capsule.set_material("clay")
        capsule.set_transparency(capsule_alpha)
        capsule.set_transform(capsule_transform_matrix(snap["capsule_xform"]))

        buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=True), dtype=np.uint8).copy()
        out[tag] = buf
        ps.remove_all_structures()

        panel_path = Path(f"{out_prefix}_{tag}.png")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(buf).save(panel_path)
        print(f"[octopus closeup] frame {snap['frame']} ({snap['particle_q'].shape[0]} vertices) -> {panel_path}")

    ps.shutdown()
    return out


def compose(images: dict[str, np.ndarray], before, after, out_path: str) -> None:
    """Crop both panels to their shared content box (one camera, so one box
    is enough) and lay them out side by side with matplotlib."""
    import matplotlib.pyplot as plt

    def content_mask(im):
        return (im[..., :3].astype(np.int16) < 250).any(axis=-1)

    h, w = images["before"].shape[:2]
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for im in images.values():
        ys, xs = np.nonzero(content_mask(im))
        if ys.size:
            lo = np.minimum(lo, [ys.min(), xs.min()])
            hi = np.maximum(hi, [ys.max(), xs.max()])
    pad = 0.04 * max(hi - lo)
    y0, x0 = np.maximum(lo - pad, 0).astype(int)
    y1, x1 = np.minimum(hi + pad, [h - 1, w - 1]).astype(int) + 1

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.9))
    for ax, tag, snap in zip(axes, ("before", "after"), (before, after)):
        ax.imshow(images[tag][y0:y1, x0:x1])
        ax.set_axis_off()
        ax.set_title(f"{tag.capitalize()} refinement ({snap['particle_q'].shape[0]} vertices)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[octopus closeup] wrote {out_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="octopus")
    ap.add_argument("--refine-scoring", default="geometric", choices=["legacy", "geometric"])
    ap.add_argument("--refine-geometric-threshold", type=float, default=1.0,
                     help="lower = more edges qualify as candidates each pass (default sim value is 2.0)")
    ap.add_argument("--refine-min-edge-length", type=float, default=0.0015,
                     help="edges shorter than 2x this are never split; lowering it lets the contact patch "
                          "keep subdividing through more levels instead of stalling after ~1 round "
                          "(default sim value is ~0.004, i.e. --contact-d1)")
    ap.add_argument("--max-frames", type=int, default=150, help="give up after this many frames")
    ap.add_argument("--min-vertex-gain", type=int, default=400,
                     help="keep stepping past the first split until the mesh has gained at least this "
                          "many vertices over the starting frame, for a visibly bigger before/after diff")
    ap.add_argument("--out-prefix", default="notebooks/octopus_refinement_closeup")
    ap.add_argument("--capsule-alpha", type=float, default=0.45)
    ap.add_argument("--closeup-radius", type=float, default=0.05,
                     help="half-width (m) of the close-up view around the poker tip")
    ap.add_argument("--pixels", type=int, default=1000)
    ap.add_argument("--ssaa", type=int, default=2)
    apply_config(ap, "octopus_refinement_closeup", argv)
    args = ap.parse_args(argv)

    sim_argv = [
        "--episode", args.episode,
        "--refine",                     # -r: adaptive refinement is opt-in, off by default
        "--graph-capture",              # -g: ~10x faster steps (CUDA graph over the CG solve)
        "--line-search", "--preconditioner",
        "--energy", "neohookean",
        "--quiet",
        "--refine-every", "1",           # apply a split every step instead of every 10th (the sim default)
        "--refine-scoring", args.refine_scoring,
        "--refine-geometric-threshold", str(args.refine_geometric_threshold),
        "--refine-min-edge-length", str(args.refine_min_edge_length),
        "--capsule-alpha", str(args.capsule_alpha),
    ]
    sim, _ = build_sim(sim_argv)
    capsule_mesh = _CapsuleMesh(sim)
    setup_polyscope(args.pixels, args.ssaa)

    before, after = collect_before_after(sim, args.max_frames, args.min_vertex_gain)
    print(f"[octopus closeup] refinement span: frame {before['frame']} "
          f"({before['particle_q'].shape[0]} verts) -> frame {after['frame']} "
          f"({after['particle_q'].shape[0]} verts)")

    images = render_closeup(
        before, after, capsule_mesh, args.capsule_alpha, args.closeup_radius, args.out_prefix,
    )
    compose(images, before, after, f"{args.out_prefix}_compare.png")


class _CapsuleMesh:
    """capsule_mesh.vertices/.indices (newton.Mesh's own fields) plus
    half_height, bundled together so render_closeup doesn't also need sim."""

    def __init__(self, sim):
        import newton
        mesh = newton.Mesh.create_capsule(sim.capsule_radius, sim.capsule_half_height, up_axis=sim.capsule_axis)
        self.vertices = mesh.vertices
        self.indices = mesh.indices
        self.half_height = sim.capsule_half_height


if __name__ == "__main__":
    main()
