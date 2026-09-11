"""Compare per-step solver timings between the new and old solver.

Reads timings.json (new solver) and timings_old.json (old solver) — both
written by Solver.write_timings() via warp.ScopedTimer, in milliseconds —
and produces timings_comparison.png: a summary bar chart of median step
cost plus per-iteration trend lines for each step.

Usage:
    python plot_timings.py [--new timings.json] [--old timings_old.json] [--out timings_comparison.png]
"""

import argparse
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np

# Colors: fixed categorical order (blue = new, orange = old), validated
# for colorblind-safe adjacent contrast. See dataviz skill palette.
COLOR_NEW = "#2a78d6"
COLOR_OLD = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e5e4e0"
SURFACE = "#fcfcfb"

# Steps are recorded under slightly different casing between solvers.
STEP_ORDER = ["Update system", "Assemble system", "Global solve", "Local solve"]

# First iteration includes one-time JIT/warmup cost and skews summary
# stats heavily; exclude it from medians (still shown in the trend lines).
WARMUP_ITERS = 1


def load_timings(path: str) -> dict[str, list[float]]:
    with open(path) as f:
        raw = json.load(f)
    return {k.strip().lower(): v for k, v in raw.items()}


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    out = np.empty(len(values) - window + 1)
    for i in range(len(out)):
        out[i] = np.median(values[i : i + window])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", default="timings.json")
    parser.add_argument("--old", default="timings_old.json")
    parser.add_argument("--out", default="timings_comparison.png")
    args = parser.parse_args()

    new = load_timings(args.new)
    old = load_timings(args.old)

    steps = [s for s in STEP_ORDER if s.lower() in new and s.lower() in old]
    if not steps:
        raise SystemExit("No matching step names found between the two timing files.")

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "text.color": TEXT_PRIMARY,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, len(steps), height_ratios=[1.1, 1])

    # --- Top: grouped bar chart of median cost per step ---
    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(steps))
    width = 0.32

    new_medians = [statistics.median(new[s.lower()][WARMUP_ITERS:]) for s in steps]
    old_medians = [statistics.median(old[s.lower()][WARMUP_ITERS:]) for s in steps]

    bars_old = ax_bar.bar(x - width / 2, old_medians, width, label="Old solver", color=COLOR_OLD)
    bars_new = ax_bar.bar(x + width / 2, new_medians, width, label="New solver", color=COLOR_NEW)

    for bars in (bars_old, bars_new):
        for b in bars:
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_SECONDARY,
            )

    for i, (om, nm) in enumerate(zip(old_medians, new_medians)):
        speedup = om / nm if nm else float("inf")
        ax_bar.text(
            i,
            max(om, nm) * 1.12,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
            fontweight="bold",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(steps)
    ax_bar.set_ylabel("Median time per step (ms)")
    ax_bar.set_ylim(top=max(max(old_medians), max(new_medians)) * 1.35)
    ax_bar.set_title(
        "Median step cost: old vs. new solver (warmup iteration excluded)", color=TEXT_PRIMARY, pad=14
    )
    ax_bar.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.legend(frameon=False)

    # --- Bottom: per-iteration trend, one small multiple per step ---
    smooth_window = 25
    for i, step in enumerate(steps):
        ax = fig.add_subplot(gs[1, i])
        for values, color, label in (
            (old[step.lower()], COLOR_OLD, "Old"),
            (new[step.lower()], COLOR_NEW, "New"),
        ):
            arr = np.asarray(values)
            ax.plot(arr, color=color, alpha=0.18, linewidth=1)
            smoothed = rolling_median(arr, smooth_window)
            offset = smooth_window // 2
            ax.plot(
                np.arange(offset, offset + len(smoothed)),
                smoothed,
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                label=label,
            )
        ax.set_yscale("log")
        ax.set_title(step, fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("Iteration")
        if i == 0:
            ax.set_ylabel("Time (ms, log scale)")
        ax.grid(color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == len(steps) - 1:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Solver step timings: new vs. old", fontsize=14, color=TEXT_PRIMARY, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    # --- Console summary ---
    print(f"\n{'Step':<18}{'Old (median ms)':>18}{'New (median ms)':>18}{'Speedup':>10}")
    for s, om, nm in zip(steps, old_medians, new_medians):
        print(f"{s:<18}{om:>18.3f}{nm:>18.3f}{om / nm:>9.2f}x")


if __name__ == "__main__":
    main()
