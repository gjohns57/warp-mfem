"""Per-step wall-time comparison: coarse mesh, coarse mesh + refinement, fine mesh.

Output is formatted for an IEEE Transactions figure: Times-family serif text at 9 pt
(8 pt single column), 7.16 in double-column width by default (``--column single`` for
3.5 in), no in-figure title, and a PDF (editable text) written next to the PNG.

Parses the ``Step and readback took <ms> ms`` lines that ``octopus_refinement`` prints
(one per simulated frame) and plots them for the three mesh conditions of one
PokeFlex episode. Headless (Agg backend) so it runs over ssh.

Plot from the existing logs (defaults point at the octopus runs used in
``notebooks/step_timing.ipynb``)::

    uv run python -m plots.step_timing --out step_timing_octopus.png

Regenerate the runs on this machine first (3 x coarse, 3 x coarse+refinement,
1 x fine; ~6 min on an RTX 4060 laptop, dominated by the fine run), then plot::

    uv run python -u -m plots.step_timing --run timing_runs/ --out step_timing_octopus.png

``--run`` uses the same arguments as the refinement sweep (``sweep_octopus.BASE_ARGS``:
headless, -g -p -l, neo-Hookean, 150 recorded frames) with the benchmark-best refinement
setting (geometric scoring, threshold 2, min edge 4 mm, every 10 frames) for the refined
runs -- the same runs ``refinement_loss.py`` scores. Custom logs can be passed with ``--norefine/--refine/--fine`` (repeatable).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The octopus benchmark runs: the same recordings refinement_loss.py scores
# (geometric scoring, threshold 2, min edge 4 mm; no-refinement baseline; fine reference).
DEFAULT_LOGS = {
    "without refinement": ["results/sweep_refine_legacy_work/log_norefine*.txt"],
    "with refinement": ["results/sweep_refine_geometric_work/log_refine_geometric_threshold=2.0_refine_min_edge_length=0.004.txt",
                        "results/sweep_refine_geometric_work/log_refine_geometric_threshold=2.0_refine_min_edge_length=0.004_r?.txt"],
    "fine mesh (no refinement)": ["results/reference_runs/fine_default.log"],
}
REFINE_ARGS = ["-r", "--refine-scoring", "geometric", "--refine-geometric-threshold", "2.0",
               "--refine-min-edge-length", "0.004"]
LABELS = list(DEFAULT_LOGS)
SERIES = {
    "without refinement": "#eb6834",
    "with refinement": "#2a78d6",
    "fine mesh (no refinement)": "#1baf7a",
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

STEP_RE = re.compile(r"Step and readback took ([\d.]+) ms")
PHASES = ["Update system", "Assemble system", "Global solve", "Local solve", "Line search"]
PHASE_RE = re.compile(r"^\s*(" + "|".join(PHASES) + r") took ([\d.]+) ms")
CONDITION_COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#8e5bd6"]   # fixed categorical order
VERT_RE = re.compile(r"new tet count (\d+), new tri count \d+, new vert count (\d+)")
WROTE_RE = re.compile(r"wrote (\d+) frames .*positions (\d+)x(\d+)x3")


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------
def parse_log(path):
    """Return (per-step ms, refinement events [(step, new vertex count)], constant vertex count or None)."""
    ms, events, nverts = [], [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                ms.append(float(m.group(1)))
                continue
            m = VERT_RE.search(line)
            if m:
                events.append((len(ms), int(m.group(2))))
                continue
            m = WROTE_RE.search(line)
            if m:
                nverts = int(m.group(3))
    return np.asarray(ms, dtype=float), events, nverts


def parse_phases(path):
    """Per-Newton-iteration solver phase timings (ms) printed by the solver's ScopedTimers.
    Absent for CUDA-graph runs (the captured graph replays without the Python timers)."""
    out = {ph: [] for ph in PHASES}
    with open(path, errors="replace") as f:
        for line in f:
            m = PHASE_RE.match(line)
            if m:
                out[m.group(1)].append(float(m.group(2)))
    return {ph: np.asarray(v, dtype=float) for ph, v in out.items() if v}


def expand(patterns):
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(str(ROOT / pat))) if not os.path.isabs(pat) else sorted(glob.glob(pat))
        out += hits if hits else ([pat] if os.path.exists(pat) else [])
    return [Path(p) for p in out]


# --------------------------------------------------------------------------
# Optional: regenerate the runs
# --------------------------------------------------------------------------
def regenerate(out_dir: Path, episode: str, frames: int, repeats: int, extra):
    from examples.refinement.sweep_octopus import BASE_ARGS, EXCLUDE_TOOL_MARGIN_M

    out_dir.mkdir(parents=True, exist_ok=True)
    base = [a for a in BASE_ARGS if a not in ("-r", "--mesh", "coarse")]
    if EXCLUDE_TOOL_MARGIN_M is not None:
        base += ["--surface-loss-exclude-tool", repr(float(EXCLUDE_TOOL_MARGIN_M))]
    jobs = []
    for r in range(repeats):
        jobs.append((f"norefine_r{r + 1}", ["--mesh", "coarse"]))
        jobs.append((f"refine_r{r + 1}", ["--mesh", "coarse"] + REFINE_ARGS))
    jobs.append(("fine", ["--mesh", "fine"]))

    logs = {"without refinement": [], "with refinement": [], "fine mesh (no refinement)": []}
    for name, args in jobs:
        log = out_dir / f"{name}.log"
        argv = [sys.executable, "-u", "-m", "examples.refinement.octopus_refinement",
                "--episode", episode, "--record-frames", str(frames),
                "--record", str(out_dir / f"{name}.npz")] + base + args + list(extra)
        print(f"[run] {name}: {' '.join(argv[3:])}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        ms, _, _ = parse_log(log)
        print(f"[run] {name}: exit={rc}, {len(ms)} steps, mean {ms[1:].mean() if len(ms) > 1 else float('nan'):.0f} ms/step",
              flush=True)
        key = ("fine mesh (no refinement)" if name == "fine"
               else "with refinement" if name.startswith("refine") else "without refinement")
        logs[key].append(str(log))
    return logs


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
# IEEE Transactions column widths (inches). Fonts: Times-family serif, >= 8 pt.
COLUMN_WIDTH = {"single": 3.5, "double": 7.16}
SERIF_STACK = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]


def ieee_rcparams(column: str = "double"):
    """matplotlib rcParams for an IEEE Transactions figure: Times-family serif, 9 pt
    (8 pt single column), black hairline axes, inward ticks, embedded editable fonts."""
    single = column == "single"
    base_pt = 8 if single else 9
    return {
        "font.family": "serif", "font.serif": SERIF_STACK, "mathtext.fontset": "stix",
        "font.size": base_pt, "axes.labelsize": base_pt, "axes.titlesize": base_pt,
        "xtick.labelsize": base_pt - 1 if single else base_pt, "ytick.labelsize": base_pt - 1 if single else base_pt,
        "legend.fontsize": base_pt - 1 if single else base_pt,
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "text.color": "black",
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.color": "#d0d0d0", "grid.linewidth": 0.4, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # embed editable text
    }


def save_figure(fig, out: Path, dpi: int):
    """PNG at ``dpi`` plus a vector PDF next to it (use the PDF in the manuscript)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    outs = [out] + ([out.with_suffix(".pdf")] if out.suffix.lower() == ".png" else [])
    for o in outs:
        fig.savefig(o, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"[plot] wrote {o}")


def plot(runs, episode: str, out: Path, dpi: int = 600, column: str = "double", title: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = column == "single"
    plt.rcParams.update(ieee_rcparams(column))

    coarse_verts = next((nv for _, _, nv in runs["without refinement"] if nv), None)

    def vert_label(label):
        _, events, nverts = runs[label][0]
        if events:
            return f"{coarse_verts}$\\rightarrow${events[-1][1]} vert." if coarse_verts else f"$\\rightarrow${events[-1][1]} vert."
        return f"{nverts} vert." if nverts else ""

    width = COLUMN_WIDTH[column]
    fig, ax = plt.subplots(figsize=(width, 2.3 if single else 2.9))
    summary = []
    n_max = 0
    lw = 0.9 if single else 1.1
    for label in LABELS:
        rows = [ms for ms, _, _ in runs[label]]
        if not rows:
            print(f"[plot] no logs for '{label}', skipping", file=sys.stderr)
            continue
        n = min(len(r) for r in rows)
        S = np.stack([r[:n] for r in rows])
        n_max = max(n_max, n)
        x = np.arange(1, n + 1)
        if S.shape[0] > 1:
            ax.fill_between(x, S.min(0), S.max(0), color=SERIES[label], alpha=0.2, linewidth=0)
        mean_ms = S[:, 1:].mean()
        short = {"without refinement": "coarse", "with refinement": "coarse + refinement",
                 "fine mesh (no refinement)": "fine"}[label]
        ax.plot(x, S.mean(0), color=SERIES[label], linewidth=lw,
                label=f"{short} ({vert_label(label)}, {mean_ms:.0f} ms/step)")
        summary.append((label, S.shape[0], n, mean_ms, S[:, 1:].sum(1).mean() / 1000.0))
    for step, _ in (runs["with refinement"][0][1] if runs["with refinement"] else []):
        ax.axvline(step, color=SERIES["with refinement"], alpha=0.35, linewidth=0.5, linestyle=(0, (1, 2)))
    ax.set_yscale("log")
    if title:
        ax.set_title(f"{episode}: wall time per simulation step", loc="left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wall time per step (ms)")
    ax.set_xlim(0, n_max)
    ax.margins(y=0.08)
    # The band between the coarse and fine curves is empty on a log axis: put the legend there.
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False,
              handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    save_figure(fig, out, dpi)
    base = summary[0][3] if summary else 1.0
    print(f"{'condition':<28}{'runs':>5}{'steps':>7}{'mean ms':>10}{'x coarse':>10}{'total s':>10}")
    for label, nr, n, mean_ms, total in summary:
        print(f"{label:<28}{nr:>5}{n:>7}{mean_ms:>10.0f}{mean_ms / base:>10.1f}{total:>10.1f}")


def plot_conditions(cond, out: Path, dpi=600, column="double", title=False):
    """``cond``: ordered {label: [log paths]}. Panel (a): wall time per step; panel (b): mean time
    per Newton iteration by solver phase (from the phase timers, when a run prints them)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = column == "single"
    plt.rcParams.update(ieee_rcparams(column))
    width = COLUMN_WIDTH[column]
    parsed = {label: [(parse_log(p)[0], parse_phases(p)) for p in paths] for label, paths in cond.items()}
    parsed = {k: [(ms, ph) for ms, ph in v if len(ms)] for k, v in parsed.items()}
    # Phase timers count only when they were printed for every step: a CUDA-graph run prints them
    # once, during the warm-up frame that is captured, and replays the graph silently afterwards.
    for k, v in parsed.items():
        parsed[k] = [(ms, ph if ph and len(next(iter(ph.values()))) >= len(ms) else {}) for ms, ph in v]
    have_phases = any(ph for v in parsed.values() for _, ph in v)
    if single:
        fig, axes = plt.subplots(2 if have_phases else 1, 1, figsize=(width, 4.6 if have_phases else 2.4), squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = plt.subplots(1, 2 if have_phases else 1, figsize=(width, 2.7), squeeze=False,
                                 gridspec_kw={"width_ratios": [3, 2]} if have_phases else None)
        axes = axes[0]
    lw = 0.9 if single else 1.1
    colors = {label: CONDITION_COLORS[i % len(CONDITION_COLORS)] for i, label in enumerate(cond)}
    summary = []

    ax = axes[0]
    for label, runs_ in parsed.items():
        if not runs_:
            print(f"[plot] no steps for '{label}'", file=sys.stderr)
            continue
        n = min(len(ms) for ms, _ in runs_)
        S = np.stack([ms[:n] for ms, _ in runs_])
        x = np.arange(1, n + 1)
        if S.shape[0] > 1:
            ax.fill_between(x, S.min(0), S.max(0), color=colors[label], alpha=0.2, linewidth=0)
        mean_ms = S[:, 1:].mean() if n > 1 else S.mean()
        ax.plot(x, S.mean(0), color=colors[label], linewidth=lw, marker="o" if n <= 12 else None, ms=2.5,
                label=f"{label} ({mean_ms:,.0f} ms/step)")
        summary.append((label, S.shape[0], n, mean_ms))
    ax.set_yscale("log")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wall time per step (ms)")
    ax.set_xlim(0, None)
    ax.legend(loc="center right", frameon=False, handlelength=1.6, borderaxespad=0.3, labelspacing=0.3)
    if title:
        ax.set_title("(a) wall time per simulation step", loc="left")
    else:
        ax.text(0.5, -0.27, "(a)", transform=ax.transAxes, ha="center", va="top")

    if have_phases:
        ax = axes[1]
        y = np.arange(len(PHASES))
        labels_with = [l for l, v in parsed.items() if any(ph for _, ph in v)]
        k = len(labels_with)
        h = 0.8 / max(k, 1)
        for i, label in enumerate(labels_with):
            means = []
            for ph in PHASES:
                vals = np.concatenate([p[ph] for _, p in parsed[label] if ph in p]) if any(ph in p for _, p in parsed[label]) else np.array([np.nan])
                means.append(np.nanmedian(vals))          # median: the first iteration carries JIT / warm-up
            ax.barh(y - 0.4 + h * (i + 0.5), means, height=h * 0.9, color=colors[label], label=label)
            for yy, m in zip(y - 0.4 + h * (i + 0.5), means):
                if np.isfinite(m):
                    ax.text(m * 1.15, yy, f"{m:,.1f}" if m < 100 else f"{m:,.0f}", va="center", fontsize=plt.rcParams["legend.fontsize"])
        ax.set_yticks(y)
        ax.set_yticklabels(PHASES)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(right=ax.get_xlim()[1] * 4)
        ax.set_xlabel("Median time per Newton iteration (ms)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        missing = [l for l in parsed if l not in labels_with]
        if missing:
            ax.text(0.98, 0.98, "not timed per phase:\n" + ", ".join(missing), transform=ax.transAxes, ha="right",
                    va="top", fontsize=plt.rcParams["legend.fontsize"] - 1, color="#555555", linespacing=1.15)
        if title:
            ax.set_title("(b) solver phases", loc="left")
        else:
            ax.text(0.5, -0.27, "(b)", transform=ax.transAxes, ha="center", va="top")
    fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=0.8)
    save_figure(fig, out, dpi)
    base = summary[0][3] if summary else 1.0
    print(f"{'condition':<32}{'runs':>5}{'steps':>7}{'mean ms':>12}{'x first':>10}")
    for label, nr, n, mean_ms in summary:
        print(f"{label:<32}{nr:>5}{n:>7}{mean_ms:>12.0f}{mean_ms / base:>10.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="octopus", help="episode name for the title / --run (default octopus)")
    ap.add_argument("--out", default="step_timing_octopus.png", help="output PNG path")
    ap.add_argument("--dpi", type=int, default=600, help="raster dpi for the PNG (a PDF is written alongside)")
    ap.add_argument("--column", choices=list(COLUMN_WIDTH), default="double",
                    help="IEEE Transactions column width: single (3.5 in) or double (7.16 in, default)")
    ap.add_argument("--title", action="store_true", help="draw a title on the axes (off for papers: use the caption)")
    ap.add_argument("--norefine", action="append", default=None, metavar="LOG",
                    help="log/glob for the coarse mesh without refinement (repeatable)")
    ap.add_argument("--refine", action="append", default=None, metavar="LOG",
                    help="log/glob for the coarse mesh with refinement (repeatable)")
    ap.add_argument("--fine", action="append", default=None, metavar="LOG",
                    help="log/glob for the fine mesh (repeatable)")
    ap.add_argument("--run", metavar="DIR", default=None,
                    help="regenerate the runs with octopus_refinement into DIR (logs + recordings), then plot them")
    ap.add_argument("--frames", type=int, default=150, help="--record-frames for --run (default 150)")
    ap.add_argument("--repeats", type=int, default=3, help="coarse repeats for --run (default 3)")
    ap.add_argument("--sim-arg", action="append", default=[],
                    help="extra literal argument forwarded to octopus_refinement under --run (repeatable)")
    ap.add_argument("--series", action="append", default=None, metavar="LABEL=GLOB",
                    help="generic comparison: one condition per option, e.g. "
                         "--series 'GPU, CUDA graph=results/timing_runs_device/gpu_graph.log' (repeatable, order kept). "
                         "Draws wall time per step plus the per-phase solver breakdown where the log has phase timers.")
    a = ap.parse_args(argv)

    if a.series:
        cond = {}
        for item in a.series:
            label, _, pat = item.partition("=")
            cond[label.strip()] = expand([pat.strip()])
            print(f"[logs] {label.strip()}: {[str(p) for p in cond[label.strip()]]}")
        plot_conditions(cond, Path(a.out), a.dpi, a.column, a.title)
        return

    if a.run:
        logs = regenerate(Path(a.run), a.episode, a.frames, a.repeats, a.sim_arg)
    else:
        logs = {
            "without refinement": a.norefine or DEFAULT_LOGS["without refinement"],
            "with refinement": a.refine or DEFAULT_LOGS["with refinement"],
            "fine mesh (no refinement)": a.fine or DEFAULT_LOGS["fine mesh (no refinement)"],
        }
    runs = {}
    for label, pats in logs.items():
        paths = expand(pats)
        runs[label] = [parse_log(p) for p in paths]
        runs[label] = [r for r in runs[label] if len(r[0])]
        print(f"[logs] {label}: {len(runs[label])} run(s) from {[str(p) for p in paths]}")
    if not any(runs.values()):
        raise SystemExit("no timing lines found in any log")
    plot(runs, a.episode, Path(a.out), a.dpi, a.column, a.title)


if __name__ == "__main__":
    main()
