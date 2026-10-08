"""Slow-motion top-down close-up GIF of the octopus poke, edges colored by
refinement score.

Live-runs the octopus sim (same flags as octopus_refinement_closeup)
at a high --sim-fps so the sim itself resolves more frames per second of
real time, snapshotting every frame's pre-step topology + edge scores, then
renders each frame as a fixed-camera close-up around the poker tip. Playing
the frames at --gif-fps < --sim-fps gives a sim-fps/gif-fps x slow-motion.

Needs a display for polyscope (a window opens briefly).

    uv run python -m examples.refinement.octopus_refinement_slowmo
"""
from __future__ import annotations

from examples.config import apply_config

import argparse
import math
import pickle
from pathlib import Path

import numpy as np

from examples.refinement.octopus_refinement_closeup import (
    BODY_COLOR, TOOL_COLOR, _CapsuleMesh, build_sim,
    capsule_transform_matrix, setup_polyscope, surface_edge_scores,
)


def capture(args):
    from mfem.refinement.refinement import edge_refinement_scores

    sim, _ = build_sim([
        "--episode", args.episode, "--refine", "--graph-capture",
        "--line-search", "--preconditioner", "--energy", "neohookean", "--quiet",
        "--fps", str(args.sim_fps),
    ] + (["--start-frame", str(args.start_frame)] if args.start_frame is not None else []) + [
        "--refine-every", str(args.refine_every),
        "--refine-scoring", "geometric",
        "--capsule-alpha", "0.45",
    ] + (["--refine-geometric-threshold", str(args.refine_geometric_threshold)]
         if args.refine_geometric_threshold is not None else [])
      + (["--refine-min-edge-length", str(args.refine_min_edge_length)]
         if args.refine_min_edge_length is not None else []))
    capsule = _CapsuleMesh(sim)
    snaps = []
    for i in range(args.frames):
        add0 = sim.solver._additional_state_0
        n = int(add0.active_particle_count.numpy()[0])
        tri_ct = int(add0.active_tri_count.numpy()[0])
        pre_state = add0.clone()
        q = sim.state_0.particle_q[:n].numpy().copy()
        tris = add0.tri_indices[:tri_ct].numpy().copy()
        xf = sim.state_0.body_q.numpy()[0].copy()
        sim.step()
        edges, scores = edge_refinement_scores(pre_state, sim.solver._refine_buffers)
        se, ss = surface_edge_scores(tris, edges, scores)
        snaps.append(dict(q=q, tris=tris, se=se, ss=np.log1p(ss), xf=xf))
        if i % 20 == 0:
            print(f"[slowmo] frame {i}: {n} verts, max score {scores.max() if scores.size else 0:.3g}")
    return sim, capsule, snaps


def render(args, sim, capsule, snaps):
    import polyscope as ps
    from PIL import Image
    from scipy.spatial.transform import Rotation

    setup_polyscope(args.pixels, args.ssaa)
    ps.remove_all_structures()

    def tip(xf):
        return np.asarray(xf[:3], float) + Rotation.from_quat(xf[3:7]).as_matrix()[:, 2] * capsule.half_height

    tips = np.array([tip(s["xf"]) for s in snaps])
    # First contact = first poke: from frame 0 until the poker has bottomed out
    # and come back to (near) its starting height.
    y = tips[:, 1]
    bottom = int(np.argmin(y[: int(args.first_poke_search * args.sim_fps)]))
    back = np.nonzero(y[bottom:] >= y[0] - 0.002)[0]
    end = bottom + (int(back[0]) if back.size else len(y) - 1 - bottom) + 1
    # Start at the top of the approach (the path can lift off first when the sim
    # starts from an early --start-frame), then run through the first poke.
    begin = 0 if args.from_start else int(np.argmax(y[:bottom + 1]))
    snaps, tips = snaps[begin:end], tips[begin:end]
    bottom -= begin
    print(f"[slowmo] first contact: sim frames {begin}-{end - 1} (poker bottoms out at {begin + bottom})")
    view_dir = np.array([0.0, -1.0, 0.0])
    up = (0.0, 0.0, -1.0)
    fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
    dist = args.closeup_radius / math.sin(math.radians(fov) * 0.5) / 0.85
    allss = np.concatenate([s["ss"] for s in snaps])
    vmax = float(np.percentile(allss, args.vmax_percentile)) or 1.0

    frames = []
    for i, s in enumerate(snaps):
        center = tips[i] if args.track_tip else tips[bottom]
        body = ps.register_surface_mesh("body", s["q"], s["tris"], smooth_shade=False)
        body.set_color(BODY_COLOR)
        body.set_edge_width(0.0)
        body.set_material("clay")
        # A curve network draws a sphere at *every* node it is given, including
        # interior tet vertices that have no surface edge; those show up as stray
        # dots through the translucent capsule. Register surface vertices only.
        used, se = np.unique(s["se"], return_inverse=True)
        se = se.reshape(-1, 2).astype(np.int32)
        ss = s["ss"]
        net = ps.register_curve_network("edge scores", s["q"][used], se)
        net.set_radius(args.edge_radius, relative=False)
        net.add_scalar_quantity("log(1 + score)", ss, defined_on="edges", enabled=True,
                                cmap=args.cmap, vminmax=(0.0, vmax))
        cap = ps.register_surface_mesh("capsule", capsule.vertices, capsule.indices.reshape(-1, 3))
        cap.set_color(TOOL_COLOR)
        cap.set_material("clay")
        cap.set_transparency(args.capsule_alpha)
        cap.set_transform(capsule_transform_matrix(s["xf"]))
        ps.set_view_camera_parameters(ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=fov, aspect=1.0),
            ps.CameraExtrinsics(root=tuple(center - view_dir * dist), look_dir=tuple(view_dir), up_dir=up)))
        buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=False), dtype=np.uint8)[..., :3].copy()
        frames.append(buf)
        ps.remove_all_structures()

    a = np.stack(frames)
    ys, xs = np.where((a < 245).any(-1).any(0))
    pad = 20
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[2]), min(ys.max() + pad, a.shape[1]))
    imgs = [Image.fromarray(f).crop(box) for f in frames]
    w, h = imgs[0].size
    w, h = w - w % 2, h - h % 2
    imgs = [im.crop((0, 0, w, h)) for im in imgs]
    slow = args.sim_fps / args.gif_fps
    # Full-res MP4 (small, smooth) plus a downscaled GIF (GIF frames are huge).
    import subprocess
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mp4 = str(Path(args.out).with_suffix(".mp4"))
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.gif_fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "16", mp4], stdin=subprocess.PIPE)
    for im in imgs:
        p.stdin.write(np.asarray(im).tobytes())
    p.stdin.close(); p.wait()
    gw = args.gif_width
    small = [im.resize((gw, round(h * gw / w)), Image.LANCZOS) for im in imgs]
    small[0].save(args.out, save_all=True, append_images=small[1:],
                  duration=int(round(1000 / args.gif_fps)), loop=0, optimize=True)
    print(f"[slowmo] wrote {mp4} ({w}x{h}) and {args.out} ({gw}px): {len(imgs)} frames at "
          f"{args.gif_fps:g} fps = {slow:.1f}x slow-motion")
    ps.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="octopus")
    ap.add_argument("--sim-fps", type=int, default=120, help="sim frame rate (frames per second of real time)")
    ap.add_argument("--gif-fps", type=float, default=30.0, help="playback rate (GIF delays are 10 ms units)")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--refine-every", type=int, default=1,
                    help="sim frames between refinement passes (higher = slower, more gradual growth)")
    ap.add_argument("--capture-only", action="store_true", help="stop after capturing sim frames")
    ap.add_argument("--from-start", action="store_true",
                    help="begin the clip at sim frame 0 instead of the top of the approach")
    ap.add_argument("--refine-geometric-threshold", type=float, default=None,
                    help="default: the sim's own (benchmark-tuned) value")
    ap.add_argument("--refine-min-edge-length", type=float, default=None)
    ap.add_argument("--closeup-radius", type=float, default=0.03)
    ap.add_argument("--edge-radius", type=float, default=0.0006)
    ap.add_argument("--capsule-alpha", type=float, default=0.45)
    ap.add_argument("--first-poke-search", type=float, default=1.4,
                    help="look for the first poke's bottom within this many seconds")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="tracked frame the sim starts from (default: the episode's rest frame, 19); "
                         "earlier gives a lead-in before contact")
    ap.add_argument("--no-track-tip", dest="track_tip", action="store_false",
                    help="fixed camera on the deepest tip position instead of following the tip")
    ap.add_argument("--pixels", type=int, default=1200)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--vmax-percentile", type=float, default=99.0)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--snapshots", default=None, help="pickle to cache/reuse the captured sim frames")
    ap.add_argument("--out", default="results/media/octopus_slowmo_top.gif")
    apply_config(ap, "octopus_refinement_slowmo", argv)
    args = ap.parse_args(argv)

    if args.snapshots and Path(args.snapshots).exists():
        with open(args.snapshots, "rb") as f:
            sim, capsule, snaps = None, *pickle.load(f)
        import warp as wp  # noqa: F401
        from examples.refinement.octopus_refinement_closeup import build_sim as _b
        sim, _ = _b(["--episode", args.episode, "--quiet"])
        capsule = _CapsuleMesh(sim)
    else:
        sim, capsule, snaps = capture(args)
        if args.snapshots:
            with open(args.snapshots, "wb") as f:
                pickle.dump((None, snaps), f)
    if args.capture_only:
        n = [len(x["q"]) for x in snaps]
        print("[slowmo] verts every 10 frames:", n[::10])
        return
    render(args, sim, capsule, snaps)


if __name__ == "__main__":
    main()
