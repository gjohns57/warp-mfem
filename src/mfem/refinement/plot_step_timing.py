"""Per-step wall-time comparison: coarse mesh, coarse mesh + refinement, fine mesh.

Parses the ``Step and readback took <ms> ms`` lines that ``sim_octopus`` prints
(one per simulated frame) and plots them for the three mesh conditions of one
PokeFlex episode. Headless (Agg backend) so it runs over ssh.

Plot from the existing logs (defaults point at the octopus runs used in
``notebooks/step_timing.ipynb``)::

    uv run python -m mfem.refinement.plot_step_timing --out step_timing_octopus.png

Regenerate the runs on this machine first (3 x coarse, 3 x coarse+refinement,
1 x fine; ~6 min on an RTX 4060 laptop, dominated by the fine run), then plot::

    uv run python -u -m mfem.refinement.plot_step_timing --run timing_runs/ --out step_timing_octopus.png

``--run`` uses the same arguments as the refinement sweep (``sweep_octopus.BASE_ARGS``:
headless, -g -p -l, neo-Hookean, 150 recorded frames) with ``--refine-scoring geometric
--refine-every 10`` for the refined runs, i.e. the setting that won the correspondence
sweep. Custom logs can be passed with ``--norefine/--refine/--fine`` (repeatable).
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

ROOT = Path(__file__).resolve().parents[3]

# The runs behind notebooks/step_timing.ipynb (octopus).
DEFAULT_LOGS = {
    "without refinement": ["sweep_refine_corr_fine_geometric_work/log_norefine*.txt"],
    "with refinement": ["sweep_refine_corr_fine_geometric_work/log_refine_every=10.txt",
                        "sweep_refine_corr_fine_geometric_work/log_refine_every=10_r?.txt"],
    "fine mesh (no refinement)": ["reference_runs/fine_default.log"],
}
LABELS = list(DEFAULT_LOGS)
SERIES = {
    "without refinement": "#eb6834",
    "with refinement": "#2a78d6",
    "fine mesh (no refinement)": "#1baf7a",
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

STEP_RE = re.compile(r"Step and readback took ([\d.]+) ms")
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
    from mfem.refinement.sweep_octopus import BASE_ARGS, EXCLUDE_TOOL_MARGIN_M

    out_dir.mkdir(parents=True, exist_ok=True)
    base = [a for a in BASE_ARGS if a not in ("-r", "--mesh", "coarse")]
    if EXCLUDE_TOOL_MARGIN_M is not None:
        base += ["--surface-loss-exclude-tool", repr(float(EXCLUDE_TOOL_MARGIN_M))]
    jobs = []
    for r in range(repeats):
        jobs.append((f"norefine_r{r + 1}", ["--mesh", "coarse"]))
        jobs.append((f"refine_r{r + 1}", ["--mesh", "coarse", "-r",
                                           "--refine-scoring", "geometric", "--refine-every", "10"]))
    jobs.append(("fine", ["--mesh", "fine"]))

    logs = {"without refinement": [], "with refinement": [], "fine mesh (no refinement)": []}
    for name, args in jobs:
        log = out_dir / f"{name}.log"
        argv = [sys.executable, "-u", "-m", "mfem.refinement.sim_octopus",
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
def plot(runs, episode: str, out: Path, dpi: int = 130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })

    coarse_verts = next((nv for _, _, nv in runs["without refinement"] if nv), None)

    def vert_label(label):
        _, events, nverts = runs[label][0]
        if events:
            return f"{coarse_verts} → {events[-1][1]} vertices" if coarse_verts else f"→ {events[-1][1]} vertices"
        return f"{nverts} vertices" if nverts else ""

    fig, ax = plt.subplots(figsize=(13, 5.2))
    summary = []
    n_max = 0
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
            ax.fill_between(x, S.min(0), S.max(0), color=SERIES[label], alpha=0.18, linewidth=0)
        mean_ms = S[:, 1:].mean()
        ax.plot(x, S.mean(0), color=SERIES[label], linewidth=2,
                label=f"{label}  ·  {vert_label(label)}  ·  {S.shape[0]} run{'s' if S.shape[0] > 1 else ''}"
                      f"  ·  mean {mean_ms:.0f} ms")
        ax.text(x[-1] + 1.5, S.mean(0)[-10:].mean(), label.split(" (")[0], color=SERIES[label],
                fontsize=9, va="center")
        summary.append((label, S.shape[0], n, mean_ms, S[:, 1:].sum(1).mean() / 1000.0))
    for step, _ in (runs["with refinement"][0][1] if runs["with refinement"] else []):
        ax.axvline(step, color=SERIES["with refinement"], alpha=0.3, linewidth=1, linestyle=":")
    ax.set_yscale("log")
    ax.set_title(f"{episode} — wall time per simulation step  (mean over runs, band = min–max; "
                 "dotted = refinement pass)", loc="left", fontsize=11, pad=8)
    ax.set_xlabel("frame")
    ax.set_ylabel("ms / step (log)")
    ax.set_xlim(0, n_max * 1.1)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, fontsize=9, ncol=1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    base = summary[0][3] if summary else 1.0
    print(f"{'condition':<28}{'runs':>5}{'steps':>7}{'mean ms':>10}{'x coarse':>10}{'total s':>10}")
    for label, nr, n, mean_ms, total in summary:
        print(f"{label:<28}{nr:>5}{n:>7}{mean_ms:>10.0f}{mean_ms / base:>10.1f}{total:>10.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="octopus", help="episode name for the title / --run (default octopus)")
    ap.add_argument("--out", default="step_timing_octopus.png", help="output PNG path")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--norefine", action="append", default=None, metavar="LOG",
                    help="log/glob for the coarse mesh without refinement (repeatable)")
    ap.add_argument("--refine", action="append", default=None, metavar="LOG",
                    help="log/glob for the coarse mesh with refinement (repeatable)")
    ap.add_argument("--fine", action="append", default=None, metavar="LOG",
                    help="log/glob for the fine mesh (repeatable)")
    ap.add_argument("--run", metavar="DIR", default=None,
                    help="regenerate the runs with sim_octopus into DIR (logs + recordings), then plot them")
    ap.add_argument("--frames", type=int, default=150, help="--record-frames for --run (default 150)")
    ap.add_argument("--repeats", type=int, default=3, help="coarse repeats for --run (default 3)")
    ap.add_argument("--sim-arg", action="append", default=[],
                    help="extra literal argument forwarded to sim_octopus under --run (repeatable)")
    a = ap.parse_args(argv)

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
    plot(runs, a.episode, Path(a.out), a.dpi)


if __name__ == "__main__":
    main()
