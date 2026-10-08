"""Error comparison (IEEE figure): coarse mesh, coarse mesh + refinement, fine mesh.

Per-frame correspondence (material-point) RMSE of the octopus benchmark runs, the same
two-panel figure as ``notebooks/refinement_loss_correspondence.ipynb``:

  (a) against the tracked PokeFlex surface (rigid detrend, pairs within 10 mm of the
      poker skipped, because the tracked mesh interpolates straight through the tool);
  (b) against the fine-mesh simulation (``results/reference_runs/fine_default.npz``), which
      sees under the tool and so isolates discretisation error.

Curves are the mean over repeats, the wash the min-max band; the legend carries the
episode RMSE (RMS over frames and repeats). Formatted like ``step_timing.py``:
Times-family serif at 9 pt (8 pt single column), IEEE column width, no in-figure
title, PDF written next to the PNG. Headless.

Per-frame scores are read from the notebook cache when present
(``notebooks/refinement_loss_correspondence_cache.npz``); recordings missing from it are
scored here (slow: ~1 min per recording) and the cache is updated. Usage::

    uv run python -m plots.refinement_loss --out refinement_loss_octopus.png
    uv run python -m plots.refinement_loss --column single --out ...
    uv run python -m plots.refinement_loss --key mse_symmetric ...   # nearest-surface metric

Recordings can be overridden with ``--norefine/--refine`` (globs, repeatable) and
``--fine`` (single recording), e.g. after regenerating them with
``step_timing.py --run DIR`` (``--norefine 'DIR/norefine_r*.npz' --refine
'DIR/refine_r*.npz' --fine DIR/fine.npz``).

``--episodes octopus turtle tp_roll`` draws one row per PokeFlex episode instead (the
figure of ``notebooks/refinement_loss_correspondence_episodes.ipynb``), reading that
notebook's cache; panels are labelled (a)-(f) row by row, left = vs tracked surface,
right = vs fine simulation::

    uv run python -m plots.refinement_loss --episodes octopus turtle tp_roll --out refinement_loss_episodes.png
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np

from plots.step_timing import ROOT, SERIES, COLUMN_WIDTH, ieee_rcparams, save_figure

TRACKED = "PlushOctopus_T1/mesh_trajectories_canonical.npy"
FINE_REFERENCE = "results/reference_runs/fine_default.npz"
EXCLUDE_TOOL_M = 0.01
CACHE = ROOT / "notebooks" / "refinement_loss_correspondence_cache.npz"
DEFAULT_RUNS = {
    "with refinement": ["results/sweep_refine_geometric_work/rec_refine_geometric_threshold=2.0_refine_min_edge_length=0.004.npz",
                        "results/sweep_refine_geometric_work/rec_refine_geometric_threshold=2.0_refine_min_edge_length=0.004_r?.npz"],
    "without refinement": ["results/sweep_refine_legacy_work/rec_norefine*.npz"],
}
EPISODE_CACHE = ROOT / "notebooks" / "refinement_loss_correspondence_episodes_cache.npz"
RUN_DIR = "results/refinement_loss_runs"
EPISODE_RUNS = {          # as in refinement_loss_correspondence_episodes.ipynb
    "octopus": (DEFAULT_RUNS, FINE_REFERENCE),
    **{e: ({"with refinement": [f"{RUN_DIR}/{e}_refine_r*.npz"],
            "without refinement": [f"{RUN_DIR}/{e}_norefine_r*.npz"]}, f"{RUN_DIR}/{e}_fine.npz")
       for e in ("turtle", "tp_roll", "dice")},
}
FINE_LABEL = "fine mesh (no refinement)"
SHORT = {"without refinement": "coarse", "with refinement": "coarse + refinement", FINE_LABEL: "fine"}
REFS = ("vs tracking", "vs fine simulation")
YLABEL = {"corr_mse": "RMSE (mm)", "mse_symmetric": "Symmetric surface RMSE (mm)",
          "mse": "Surface RMSE (mm)"}


def expand(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat if os.path.isabs(pat) else str(ROOT / pat)))
        out += hits if hits else ([pat] if os.path.exists(pat) else [])
    return [os.path.relpath(p, ROOT) if os.path.isabs(p) and p.startswith(str(ROOT)) else p for p in out]


# --------------------------------------------------------------------------
# Scoring (cache first)
# --------------------------------------------------------------------------
def load_cache(path: Path):
    if path.exists():
        with np.load(path, allow_pickle=True) as d:
            return dict(d["curves"].item())
    return {}


def save_cache(path: Path, curves):
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, curves=np.array(curves, dtype=object))


def get_curve(curves, ref, p):
    """Cache row for (reference, recording), whichever path spelling it was stored under."""
    for q in (p, os.path.abspath(p), os.path.relpath(p, ROOT) if os.path.isabs(p) else str(ROOT / p)):
        if (ref, q) in curves:
            return curves[(ref, q)]
    return None


def ensure_scored(curves, runs, fine_ref, cache_path, tracked, episode=None):
    """Score every (reference, recording) pair missing from ``curves``. ``curves`` is keyed
    (ref, path); with ``episode`` the on-disk cache is keyed (episode, ref, path)."""
    to_score = [(ref, p) for ref in REFS for paths in runs.values() for p in paths]
    if fine_ref:
        to_score.append(("vs tracking", fine_ref))
    missing = [(ref, p) for ref, p in to_score if get_curve(curves, ref, p) is None]
    if not missing:
        return curves

    def persist():
        if episode is None:
            save_cache(cache_path, curves)
        else:
            disk = load_cache(cache_path)
            disk.update({(episode, ref, p): c for (ref, p), c in curves.items()})
            save_cache(cache_path, disk)
    os.chdir(ROOT)
    try:  # the tracked-surface loader expects polyscope to be initialised
        import polyscope as ps
        ps.set_allow_headless_backends(True) if hasattr(ps, "set_allow_headless_backends") else None
        ps.init()
    except Exception:
        pass
    from examples.refinement.tracked_surface import TrackedSurfaceOverlay
    from examples.refinement.surface_loss import (
        RecordingOverlay, TrackedSurfaceLoss, TrackedCorrespondenceLoss, score_recording, _load_recording)

    start = _load_recording(missing[0][1])["start_frame"]
    overlays = {"vs tracking": (TrackedSurfaceOverlay(tracked, alpha=0.0, start_frame=start, detrend="rigid"), EXCLUDE_TOOL_M)}
    if fine_ref and os.path.exists(fine_ref):
        overlays["vs fine simulation"] = (RecordingOverlay(fine_ref), None)
    for ref, p in missing:
        if ref not in overlays or not os.path.exists(p):
            print(f"[score] skipping {ref} {p} (missing)", file=sys.stderr)
            continue
        overlay, margin = overlays[ref]
        print(f"[score] {p}  {ref} ...", flush=True)
        surf = TrackedSurfaceLoss(overlay, symmetric=True, exclude_tool_margin=margin)
        # a fresh correspondence per recording: bind() fixes the pairing to that recording's first frame
        curves[(ref, p)] = score_recording(p, surf, corr=TrackedCorrespondenceLoss(overlay, exclude_tool_margin=margin))
        persist()
    return curves


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def episode_rmse_mm(a):
    """RMS over frames and repeats, i.e. sqrt of the sweep's mean-MSE objective."""
    return math.sqrt(np.mean(a)) * 1e3


def emptiest_corner(ax, pts, w=0.5, h=0.34):
    """Corner (axes fraction) whose w x h box holds the fewest plotted points."""
    xs, ys = pts[:, 0], pts[:, 1]
    best, best_n = None, None
    for name, x0, y0 in (("upper right", 1 - w, 1 - h), ("lower right", 1 - w, 0), ("upper left", 0, 1 - h), ("lower left", 0, 0)):
        n = np.count_nonzero((xs >= x0) & (xs <= x0 + w) & (ys >= y0) & (ys <= y0 + h))
        if best_n is None or n < best_n:
            best, best_n = name, n
    return best


def plot(rows, key, out: Path, dpi=600, column="double", title=False):
    """``rows``: list of (episode name or None, curves, runs, fine_ref); one figure row each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = column == "single"
    plt.rcParams.update(ieee_rcparams(column))
    width = COLUMN_WIDTH[column]
    nrows = len(rows)
    row_h = 2.3 if single else (2.5 if nrows == 1 else 2.0)
    if single:
        fig, axes = plt.subplots(2 * nrows, 1, figsize=(width, row_h * 2 * nrows + 0.3), sharex=True, squeeze=False)
        axes = axes.reshape(nrows, 2)
    else:
        fig, axes = plt.subplots(nrows, 2, figsize=(width, row_h * nrows + 0.3), sharex=True, squeeze=False)
    lw = 0.9 if single else 1.1
    panels = [("vs tracking", "vs. tracked surface (tool region excluded)"),
              ("vs fine simulation", "vs. fine-mesh simulation (discretisation error)")]
    summary = []
    letters = iter("abcdefghijkl")
    handles = {}
    for r, (episode, curves, runs, fine_ref) in enumerate(rows):
        row_max = 0.0
        for c_i, (ref, caption) in enumerate(panels):
            ax = axes[r][c_i]
            note, pts = [], []
            for label in ("without refinement", "with refinement"):
                rows_ = [(p, get_curve(curves, ref, p)) for p in runs[label]]
                rows_ = [(p, cv) for p, cv in rows_ if cv is not None]
                if not rows_:
                    continue
                arrs = [np.asarray(cv[key], dtype=float) for _, cv in rows_]
                n = min(len(a) for a in arrs)
                t = np.asarray(rows_[0][1]["time"], dtype=float)[:n]
                a = np.stack([x[:n] for x in arrs])
                rmse = np.sqrt(a) * 1e3
                col = SERIES[label]
                if a.shape[0] > 1:
                    ax.fill_between(t, rmse.min(0), rmse.max(0), color=col, alpha=0.15, linewidth=0)
                mean = np.sqrt(a.mean(0)) * 1e3
                (h,) = ax.plot(t, mean, color=col, linewidth=lw, label=SHORT[label])
                handles.setdefault(label, h)
                row_max = max(row_max, float(rmse.max()))
                pts.append(np.column_stack([t, mean]))
                note.append(f"{SHORT[label]} {episode_rmse_mm(a):.1f}")
                summary.append((episode, ref, label, a.shape[0], episode_rmse_mm(a), a.mean()))
            cf = get_curve(curves, ref, fine_ref) if fine_ref else None
            if cf is not None:                                  # single run, no repeat band
                tf = np.asarray(cf["time"], dtype=float)
                af = np.asarray(cf[key], dtype=float)
                yf = np.sqrt(af) * 1e3
                (h,) = ax.plot(tf, yf, color=SERIES[FINE_LABEL], linewidth=lw, label=SHORT[FINE_LABEL])
                handles.setdefault(FINE_LABEL, h)
                row_max = max(row_max, float(yf.max()))
                pts.append(np.column_stack([tf, yf]))
                note.append(f"{SHORT[FINE_LABEL]} {episode_rmse_mm(af):.1f}")
                summary.append((episode, ref, FINE_LABEL, 1, episode_rmse_mm(af), af.mean()))
            ax.set_xlim(0, None)
            ax._note = (note, np.concatenate(pts) if pts else np.zeros((0, 2)))
            letter = next(letters)
            if title:
                ax.set_title(f"({letter}) {episode + ': ' if episode else ''}{caption}", loc="left")
            last_row = single or r == nrows - 1
            if last_row:
                ax.set_xlabel("Simulation time (s)")
            if not title:   # IEEE: subfigure labels below the panel; the caption explains them
                has_xlabel = last_row and (not single or ax is axes[-1][-1])
                ax.text(0.5, -0.27 if has_xlabel else -0.06, f"({letter})", transform=ax.transAxes,
                        ha="center", va="top")
        ylab = YLABEL.get(key, key)
        if episode and nrows > 1:
            ylab = f"{episode}\n{ylab}"
        for ax in axes[r]:
            ax.set_ylim(0, row_max * 1.06)              # one y-scale per episode row
        axes[r][0].set_ylabel(ylab)
        if single:
            axes[r][1].set_ylabel(ylab)
        else:
            plt.setp(axes[r][1].get_yticklabels(), visible=False)
            axes[r][1].tick_params(axis="y", length=0)
        # episode RMSE per panel: a small boxed note in the emptiest corner (text in ink, not series colour)
        for ax in axes[r]:
            note, pts = ax._note
            if not note:
                continue
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            frac = np.column_stack([(pts[:, 0] - x0) / (x1 - x0), (pts[:, 1] - y0) / (y1 - y0)])
            corner = emptiest_corner(ax, frac)
            va, ha = corner.split()
            xy = (0.97 if ha == "right" else 0.03, 0.95 if va == "upper" else 0.05)
            ax.text(*xy, "episode RMSE (mm)\n" + "\n".join(note), transform=ax.transAxes, ha=ha,
                    va="top" if va == "upper" else "bottom", fontsize=plt.rcParams["legend.fontsize"], linespacing=1.15,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="square,pad=0.25"))
    order = [l for l in ("without refinement", "with refinement", FINE_LABEL) if l in handles]
    fig.legend([handles[l] for l in order], [SHORT[l] for l in order], loc="upper center", ncol=len(order),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=2.0)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.8, rect=(0, 0, 1, 1 - 0.28 / fig.get_figheight()))
    save_figure(fig, out, dpi)

    base = {(e, ref): mse for e, ref, label, _, _, mse in summary if label == "without refinement"}
    print(f"{'episode':<10}{'reference':<20}{'condition':<28}{'runs':>5}{'RMSE mm':>9}{'MSE vs coarse':>15}")
    for e, ref, label, nr, rmse, mse in summary:
        chg = f"{(mse / base[(e, ref)] - 1) * 100:+.0f}%" if (e, ref) in base else ""
        print(f"{str(e):<10}{ref:<20}{label:<28}{nr:>5}{rmse:>9.2f}{chg:>15}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="refinement_loss_octopus.png", help="output PNG path (PDF written alongside)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--column", choices=list(COLUMN_WIDTH), default="double",
                    help="IEEE column width: double (7.16 in, panels side by side, default) or single (3.5 in, stacked)")
    ap.add_argument("--key", default="corr_mse", help="per-frame MSE column to plot (corr_mse | mse_symmetric | mse)")
    ap.add_argument("--title", action="store_true", help="draw panel titles (off for papers: use the caption)")
    ap.add_argument("--norefine", action="append", default=None, metavar="NPZ", help="recording glob, coarse mesh (repeatable)")
    ap.add_argument("--refine", action="append", default=None, metavar="NPZ", help="recording glob, coarse + refinement (repeatable)")
    ap.add_argument("--fine", default=FINE_REFERENCE, help="fine-mesh recording (reference for panel (b), also scored in (a))")
    ap.add_argument("--tracked", default=TRACKED, help="tracked surface .npy for panel (a)")
    ap.add_argument("--cache", default=None, help="per-frame score cache (.npz); defaults to the matching notebook cache")
    ap.add_argument("--episodes", nargs="+", default=None, metavar="EP",
                    help="one figure row per episode (octopus turtle tp_roll [dice]); reads the episodes notebook cache")
    a = ap.parse_args(argv)

    if a.episodes:
        from examples.refinement.pokeflex_episodes import get_episode
        cache_path = Path(a.cache) if a.cache else EPISODE_CACHE
        disk = load_cache(cache_path)
        octo = load_cache(CACHE)                       # the octopus notebook's scoring is reused
        rows = []
        for e in a.episodes:
            run_globs, fine_ref = EPISODE_RUNS[e]
            runs = {k: expand(v) for k, v in run_globs.items()}
            print(f"[runs] {e}: refine {len(runs['with refinement'])}, norefine {len(runs['without refinement'])}, "
                  f"fine {'yes' if os.path.exists(fine_ref) else 'MISSING'}")
            curves = {(ref, p): c for (ep, ref, p), c in disk.items() if ep == e}
            if e == "octopus":
                for k, c in octo.items():
                    curves.setdefault(k, c)
            curves = ensure_scored(curves, runs, fine_ref, cache_path, get_episode(e).tracked_surface_npy(), episode=e)
            rows.append((e, curves, runs, fine_ref))
        plot(rows, a.key, Path(a.out), a.dpi, a.column, a.title)
        return

    runs = {
        "with refinement": expand(a.refine or DEFAULT_RUNS["with refinement"]),
        "without refinement": expand(a.norefine or DEFAULT_RUNS["without refinement"]),
    }
    for k, v in runs.items():
        print(f"[runs] {k}: {len(v)} recording(s) {v}")
    fine_ref = os.path.relpath(a.fine, ROOT) if os.path.isabs(a.fine) and a.fine.startswith(str(ROOT)) else a.fine
    cache_path = Path(a.cache) if a.cache else CACHE
    curves = load_cache(cache_path)
    curves = ensure_scored(curves, runs, fine_ref, cache_path, a.tracked)
    plot([(None, curves, runs, fine_ref)], a.key, Path(a.out), a.dpi, a.column, a.title)


if __name__ == "__main__":
    main()
