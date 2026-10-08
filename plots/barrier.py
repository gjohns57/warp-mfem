"""Contact barrier figure (IEEE): plain log barrier vs its quadratic extrapolation.

Draws the IPC-style barrier used in :mod:`mfem.refinement.contact`::

    b(d) = -(d - d1)^2 log(d / d1)      for d0 < d < d1,   0 for d >= d1

and, for d <= d0, the two continuations: the plain log barrier (which diverges as
d -> 0+ and is undefined for penetrating d < 0) versus the C2 quadratic Taylor
extrapolation about d0 that the solver actually uses (finite, defined for every d,
so a Newton step that overshoots into penetration still has a well-defined energy).

Formatted like ``step_timing.py``: Times-family serif, IEEE single-column width
(3.5 in) by default, PDF written next to the PNG. Headless. Usage::

    uv run python -m plots.barrier --out notebooks/barrier_ieee.png
    uv run python -m plots.barrier --d0 0.5 --d1 2.0 --column double --out ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plots.step_timing import COLUMN_WIDTH, ieee_rcparams, save_figure

LOG_COLOR, QUAD_COLOR = "#eb6834", "#2a78d6"    # fixed categorical order (as SERIES)


def log_barrier(d, d1):
    """b, b', b'' of the plain log barrier on (0, d1); zero for d >= d1 (nan for d <= 0)."""
    d = np.asarray(d, dtype=float)
    b = np.zeros_like(d)
    db = np.zeros_like(d)
    d2b = np.zeros_like(d)
    m = (d > 0) & (d < d1)
    dd, L = d[m], np.log(d[m] / d1)
    b[m] = -(dd - d1) ** 2 * L
    db[m] = -(2.0 * (dd - d1) * L + (dd - d1) ** 2 / dd)
    d2b[m] = -(2.0 * L + 4.0 * (dd - d1) / dd - (dd - d1) ** 2 / dd**2)
    for a in (b, db, d2b):
        a[d <= 0] = np.nan
    return b, db, d2b


def extrapolated_barrier(d, d0, d1):
    """The solver's barrier: log barrier on (d0, d1), quadratic Taylor extrapolation for d <= d0."""
    d = np.asarray(d, dtype=float)
    b, db, _ = log_barrier(d, d1)
    b0, db0, d2b0 = (float(x) for x in log_barrier(d0, d1))
    m = d <= d0
    b[m] = b0 + db0 * (d[m] - d0) + 0.5 * d2b0 * (d[m] - d0) ** 2
    db[m] = db0 + d2b0 * (d[m] - d0)
    return b, db


def plot(d0: float, d1: float, out: Path, dpi: int = 600, column: str = "single", gradient: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(ieee_rcparams(column))
    width = COLUMN_WIDTH[column]
    lw = 1.1

    d = np.linspace(-0.5 * d1, 1.15 * d1, 2000)
    b_log, db_log, _ = log_barrier(d, d1)
    b_q, db_q = extrapolated_barrier(d, d0, d1)
    b0 = float(log_barrier(d0, d1)[0])
    ymax = 1.08 * float(extrapolated_barrier(d.min(), d0, d1)[0])   # show the whole extrapolation

    panels = [("$b(d)$", b_log, b_q)] + ([("$b'(d)$", db_log, db_q)] if gradient else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, (1.9 if gradient else 2.2) * len(panels)), sharex=True, squeeze=False)

    for ax, (ylabel, y_log, y_q) in zip(axes[:, 0], panels):
        # Penetration half-plane and the active band (d0, d1).
        ax.axvspan(d.min(), 0.0, color="#000000", alpha=0.05, lw=0)
        for x in (0.0, d0, d1):
            ax.axvline(x, color="#898781", lw=0.5, ls=(0, (3, 2)), zorder=1)

        shared = d >= d0
        ax.plot(d[shared], y_q[shared], color="#0b0b0b", lw=lw, zorder=3)
        ax.plot(d[~shared], y_log[~shared], color=LOG_COLOR, lw=lw, zorder=3, label="log barrier")
        ax.plot(d[~shared], y_q[~shared], color=QUAD_COLOR, lw=lw, ls=(0, (4, 1.5)), zorder=4,
                label="quadratic extrapolation")
        ax.plot([d0], [y_q[np.argmin(np.abs(d - d0))]], "o", ms=3.2, color="#0b0b0b", mfc="white", mew=0.8, zorder=5)

        ax.set_ylabel(ylabel)
        ax.set_xlim(d.min(), d.max())
        ax.grid(False)
        ax.set_yticks([0.0])
        ax.set_yticklabels(["0"])
    ax = axes[0, 0]
    ax.set_ylim(-0.08 * ymax, ymax)
    if gradient:
        g_lo = float(np.nanmin(np.concatenate([db_q, db_log[d >= d0]])))
        axes[1, 0].set_ylim(1.35 * g_lo, -0.08 * g_lo)

    # Axis: distances in units of d1 with d0, d1 named.
    axes[-1, 0].set_xticks([0.0, d0, d1])
    axes[-1, 0].set_xticklabels(["0", r"$d_0$", r"$d_1$"])
    axes[-1, 0].set_xlabel("signed distance $d$")

    # Direct labels (text in ink, identity carried by the adjacent stroke).
    fs = plt.rcParams["legend.fontsize"]
    ax.text(0.5 * (d0 + d1), 0.12 * ymax, "shared log barrier\n$-(d-d_1)^2\\log(d/d_1)$", ha="center", va="bottom",
            fontsize=fs, color="#52514e")
    ax.text(d1 + 0.02 * d1, 0.04 * ymax, "$b=0$", ha="left", va="bottom", fontsize=fs, color="#52514e")
    ax.text(-0.03 * d1, 0.97 * ymax, "penetration\n$(d<0)$", ha="right", va="top", fontsize=fs, color="#52514e")
    i_log = np.nanargmin(np.abs(b_log - 0.6 * ymax))
    ax.annotate("log barrier\n$\\to\\infty$ as $d\\to0^+$,\nundefined for $d<0$", xy=(d[i_log], b_log[i_log]),
                xytext=(0.3 * d1, 0.75 * ymax), ha="left", va="center", fontsize=fs, color=LOG_COLOR,
                arrowprops=dict(arrowstyle="-", color=LOG_COLOR, lw=0.5, shrinkA=0, shrinkB=1))
    i_q = np.argmin(np.abs(d + 0.3 * d1))
    ax.annotate("quadratic\nextrapolation\n($C^2$ at $d_0$)", xy=(d[i_q], b_q[i_q]), xytext=(-0.46 * d1, 0.3 * ymax),
                ha="left", va="center", fontsize=fs, color=QUAD_COLOR,
                arrowprops=dict(arrowstyle="-", color=QUAD_COLOR, lw=0.5, shrinkA=0, shrinkB=1))

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.4)
    save_figure(fig, out, dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d0", type=float, default=0.5, help="inner barrier distance (default 0.5, i.e. mm for the octopus defaults)")
    ap.add_argument("--d1", type=float, default=2.0, help="outer barrier distance (default 2.0)")
    ap.add_argument("--column", choices=list(COLUMN_WIDTH), default="single")
    ap.add_argument("--gradient", action="store_true", help="add a second panel with the derivative b'(d)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="notebooks/barrier_ieee.png")
    a = ap.parse_args()
    plot(a.d0, a.d1, Path(a.out), a.dpi, a.column, a.gradient)


if __name__ == "__main__":
    main()
