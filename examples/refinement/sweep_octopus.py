"""Parameter sweep for ``sim_octopus.py``, scored by the surface-tracking loss.

Each trial runs a short headless ``sim_octopus`` record (``--record-frames``)
with ``--surface-loss`` and a set of CLI overrides, then reads the per-frame loss
curve back from ``--surface-loss-out`` and reduces it to one scalar objective
(default: the frame-mean MSE, in m^2, matching the sim's own summary line).

A **tool-submersion penalty** is folded into the objective: the sim now records
a per-frame ``capsule_gap`` (min signed distance from a soft-body particle to the
poker capsule; negative => the tool has sunk into the mesh), and any depth past
``--tool-pen-tol`` is charged with weight ``--tool-pen-weight`` (default 2.0; 0
disables) so that runs where the poker punches into the body sort worse even when
their bare surface RMSE looks fine. The raw ``tool_pen_{rms,max}_mm`` /
``tool_pen_frac`` / ``tool_worst_gap_mm`` are kept unpenalised in the results.

Three search strategies:

* ``coordinate`` (default) -- start from the baseline (no overrides), then walk
  one parameter at a time over its listed values, keeping the best; repeat for
  ``--passes`` sweeps. Cheap (~sum of the value-list lengths per pass) and the
  trace shows which knob actually moved the loss.
* ``random`` -- draw ``--trials`` independent combinations. List values are
  sampled uniformly; a ``[lo, hi]`` / ``[lo, hi, "log"]`` triple is sampled
  continuously.
* ``grid`` -- full Cartesian product of the list values (guarded by
  ``--max-trials``).

The search space is a dict of ``name -> spec``. A spec is either

* ``{"flag": "--cli-flag", "values": [...]}`` -- a discrete set (``random``
  picks one uniformly; ``grid`` / ``coordinate`` walk them all), or
* ``{"flag": "--cli-flag", "range": [lo, hi], "scale": "log"|"lin",
  "steps": N}`` -- a continuous interval. ``random`` samples it; ``grid`` /
  ``coordinate`` discretise it into ``steps`` (default 5) points.

A bare list is shorthand for ``{"flag": "--"+name.replace("_","-"),
"values": [...]}``. A ``store_true`` switch is ``{"flag": "--refine",
"type": "flag", "values": [false, true]}`` -- the flag is emitted when the value
is truthy and omitted otherwise. Override the whole space with ``--space
space.json``.

The built-in default is built around the hand-run baseline

    sim_octopus --iterations 12 --substeps 3 -gplr --record
                --energy neohookean --mesh coarse

``BASE_ARGS`` pins ``-g -p -l -r``, neo-Hookean and the coarse mesh on every
trial, and ``DEFAULT_SPACE`` varies the solver-convergence (``--iterations``,
``--substeps``), contact-barrier (``--contact-d0/d1/stiffness``), refinement
(``--refine-every``, ``--max-new-vertices``, the two ``--refine-*-score-weight``,
``--refine-conflict-iterations``) and line-search (``--line-search-*``) knobs;
their midpoint values reproduce the baseline. ``CONSTRAINTS`` (currently
``contact_d0 < contact_d1``) filters invalid random / grid combos.

Results stream to ``--out`` as JSON lines (one per trial, written as it
finishes, so the sweep resumes / is inspectable mid-run) and a sorted CSV is
written next to it at the end.

    python -m mfem.refinement.sweep_octopus --frames 45 --passes 2
    python -m mfem.refinement.sweep_octopus --strategy random --trials 60 --jobs 1
    python -m mfem.refinement.sweep_octopus --space my_space.json --strategy grid
    python -m mfem.refinement.sweep_octopus --dry-run          # just print the plan
"""

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

from mfem.refinement.pokeflex_episodes import EPISODES


# --------------------------------------------------------------------------
# Default search space -- edit here or pass --space space.json
# --------------------------------------------------------------------------
# Baseline this sweep is built around (the invocation the user runs by hand):
#   sim_octopus --iterations 12 --substeps 3 -gplr --record
#               --energy neohookean --mesh coarse
# so BASE_ARGS pins -g/-p/-l/-r + neohookean + coarse, and the space varies the
# solver-convergence, contact-barrier, refinement and line-search knobs (their
# midpoint values below reproduce the baseline). Material / gravity / tool
# placement are left at their identified defaults.
DEFAULT_SPACE = {
    # ---- Solver convergence -----------------------------------------------
    "iterations": {"flag": "--iterations", "values": [6, 9, 12, 16, 24]},
    "substeps": {"flag": "--substeps", "values": [1, 2, 3, 4, 6]},

    # ---- Contact barrier (metres; keep contact_d0 < contact_d1) ----------
    "contact_d0": {"flag": "--contact-d0", "values": [1.0e-3, 2.0e-3, 4.0e-3]},
    "contact_d1": {"flag": "--contact-d1", "values": [4.0e-3, 8.0e-3, 1.2e-2]},
    "contact_stiffness": {
        "flag": "--contact-stiffness", "values": [1.0e4, 1.0e5, 1.0e6],
    },

    # ---- Adaptive refinement (-r is on in BASE_ARGS) --------------------
    "refine_every": {"flag": "--refine-every", "values": [5, 10, 20]},
    # (--max-new-vertices is accepted by the CLI but the solver ignores it.)
    "refine_tet_score_weight": {
        "flag": "--refine-tet-score-weight", "values": [0.0, 100.0, 400.0],
    },
    "refine_vertex_score_weight": {
        "flag": "--refine-vertex-score-weight", "values": [0.0, 200.0, 800.0],
    },
    "refine_conflict_iterations": {
        "flag": "--refine-conflict-iterations", "values": [5, 10, 20],
    },
    # Geometric scoring (only matters with --refine-scoring geometric in extra).
    "refine_min_edge_length": {
        "flag": "--refine-min-edge-length", "values": [4.0e-3, 8.0e-3, 1.2e-2],
    },
    "refine_elastic_weight": {
        "flag": "--refine-elastic-weight", "values": [0.0, 1.0, 4.0],
    },
    "refine_tri_contact_weight": {
        "flag": "--refine-tri-contact-weight", "values": [0.5, 1.0, 2.0],
    },
    "refine_geometric_threshold": {
        "flag": "--refine-geometric-threshold", "values": [0.5, 1.0, 2.0],
    },

    # ---- Backtracking line search (-l is on in BASE_ARGS) --------------
    "line_search_max_iterations": {
        "flag": "--line-search-max-iterations", "values": [10, 30, 60],
    },
    "line_search_alpha0": {"flag": "--line-search-alpha0", "values": [0.5, 1.0]},
    "line_search_tau": {"flag": "--line-search-tau", "values": [0.3, 0.5, 0.7]},
    "line_search_c": {"flag": "--line-search-c", "values": [1.0e-4, 1.0e-2, 1.0e-1]},
}

# Ordering / validity constraints on a full parameter set (name -> value).
# Combos that fail are skipped by the random / grid strategies; coordinate
# descent walks one knob at a time off a valid baseline so it never trips these.
# --space refine: only the adaptive-refinement knobs, walked from the sim's
# refinement defaults (legacy scoring). Coordinate descent tries the geometric
# scoring first, then the growth rate, then the geometric weights (which are
# no-ops under legacy scoring, and vice versa for the legacy weights). The
# no-refinement reference is always run first so the result says whether any
# of it beats simply not remeshing.
# Geometric-scoring knobs; --space refine-geometric fixes --refine-scoring
# geometric for every trial (reference and baseline included) and walks these.
REFINE_GEOMETRIC_SPACE = {
    "refine_every": {"flag": "--refine-every", "values": [1, 3, 10, 30]},
    "refine_min_edge_length": {
        "flag": "--refine-min-edge-length", "values": [4.0e-3, 8.0e-3, 1.2e-2],
    },
    "refine_geometric_threshold": {
        "flag": "--refine-geometric-threshold", "values": [0.5, 1.0, 2.0, 4.0],
    },
    "refine_elastic_weight": {
        "flag": "--refine-elastic-weight", "values": [0.0, 1.0, 4.0],
    },
    "refine_tri_contact_weight": {
        "flag": "--refine-tri-contact-weight", "values": [0.0, 1.0, 2.0],
    },
    "refine_contact_weight": {
        "flag": "--refine-contact-weight", "values": [0.0, 1.0, 2.0],
    },
}
# Legacy-scoring knobs (--space refine-legacy).
REFINE_LEGACY_SPACE = {
    "refine_every": {"flag": "--refine-every", "values": [1, 3, 10, 30]},
    "refine_tet_score_weight": {
        "flag": "--refine-tet-score-weight", "values": [0.0, 100.0, 400.0],
    },
    "refine_vertex_score_weight": {
        "flag": "--refine-vertex-score-weight", "values": [0.0, 200.0, 800.0],
    },
}
BUILTIN_SPACES = {
    "default": (DEFAULT_SPACE, []),
    "refine-geometric": (REFINE_GEOMETRIC_SPACE, ["--refine-scoring", "geometric"]),
    "refine-legacy": (REFINE_LEGACY_SPACE, ["--refine-scoring", "legacy"]),
}

CONSTRAINTS = [
    (lambda p: p.get("contact_d0", 2.0e-3) < p.get("contact_d1", 8.0e-3),
     "contact_d0 < contact_d1"),
]

# Which PokeFlex episode every trial simulates (sim_octopus --episode). Set from
# --episode in main().
EPISODE = "octopus"

# Args every trial gets (before the per-trial overrides, which win).
# Mirrors the hand-run baseline: -g graph capture, -p preconditioner,
# -l line search, -r refinement, neo-Hookean energy, coarse mesh.
BASE_ARGS = [
    "--quiet",
    "--headless",                       # polyscope mock backend: no window pops up
    "-g", "-p", "-l", "-r",
    "--energy", "neohookean",
    "--mesh", "coarse",
    "--surface-loss",
    "--surface-loss-every", "100000",   # accumulate every frame, print never
    "--tracked-surface-detrend", "rigid",
    "--tracked-surface-alpha", "0.0",
    "--surface-loss-symmetric",         # also tracked -> sim (density independent)
    "--surface-loss-correspondence",    # material-point loss (corr_* columns)
    "--bail-unstable",                  # abort divergent / punch-through trials fast
]
# Tracked vertices closer than this (m, negative = inside) to the poker are
# treated as untracked and sim vertices that close are left out of the
# sim -> tracked mean: the tracked mesh interpolates straight through the tool
# (8 mm inside on average), so it cannot judge the dimple. Overridable with
# --exclude-tool-margin; appended to BASE_ARGS in build_argv.
EXCLUDE_TOOL_MARGIN_M = 0.01


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
# Scalar objectives derived from the per-frame loss curve. The two *symmetric*
# / *tracked_to_sim* ones need --surface-loss-symmetric (in BASE_ARGS): the
# tracked -> sim direction counts every tracked vertex once, so it cannot be
# gamed by how many sim vertices refinement puts under the tool, which is
# exactly where the tracked mesh is blind (see --surface-loss-exclude-tool).
# *corr_mse* needs --surface-loss-correspondence (also in BASE_ARGS): the
# material-point loss, scored on the initial surface vertices only, so it is
# likewise independent of how many vertices refinement adds and, unlike the
# nearest-surface distances, also sees sliding / tangential slip.
OBJECTIVES = ("symmetric_mse", "tracked_to_sim_mse", "corr_mse", "mean_mse", "mean_rmse_mm",
              "median_rmse_mm", "max_rmse_mm", "last_rmse_mm")
# Tool-submersion penalty. sim_octopus writes per-frame `capsule_gap` (min
# signed distance from a soft-body particle to the poker capsule surface, m;
# negative => the tool has sunk into the mesh). Depth past PEN_TOL_M is charged
# to the objective with weight PEN_WEIGHT, in units matching the objective:
#   mean_mse   += PEN_WEIGHT * mean(pen_m ** 2)
#   *_mm       += 1e3 * sqrt(PEN_WEIGHT * mean(pen_m ** 2))
# so sqrt(mean_mse) * 1e3 stays consistent with the mm objectives.
PEN_WEIGHT = 2.0
PEN_TOL_M = 0.002   # ~= the sim's default --contact-d0: a gap shallower than
                   # this is still the contact barrier's stiff zone, not the
                   # tool truly submerging.


def tool_pen_depths(curve):
    """Per-frame tool-into-mesh penetration depth past PEN_TOL_M, metres."""
    gap = curve.get("capsule_gap")
    if gap is None:
        return np.zeros(0)
    gap = np.asarray(gap, dtype=np.float64)
    gap = gap[np.isfinite(gap)]
    return np.clip(-gap - PEN_TOL_M, 0.0, None)


def tool_penetration_stats(curve):
    """Unpenalised tool-submersion summary for logging / the results table."""
    gap = curve.get("capsule_gap")
    pen = tool_pen_depths(curve)
    n = int(pen.size)
    gap_arr = (np.asarray(gap, dtype=np.float64) if gap is not None
               else np.zeros(0))
    gap_arr = gap_arr[np.isfinite(gap_arr)]
    return {
        "tool_pen_rms_mm": float(np.sqrt(np.mean(pen ** 2)) * 1e3) if n else 0.0,
        "tool_pen_max_mm": float(pen.max() * 1e3) if n else 0.0,
        "tool_pen_frac": float((pen > 0).mean()) if n else 0.0,
        "tool_worst_gap_mm": float(gap_arr.min() * 1e3) if gap_arr.size else 0.0,
    }


def reduce_curve(curve, objective):
    """Scalar objective from a loss curve dict (arrays keyed by mse/rmse_mm/...),
    with the tool-submersion penalty folded in."""
    mse = np.asarray(curve["mse"], dtype=np.float64)
    rmse_mm = np.asarray(curve["rmse_mm"], dtype=np.float64)
    if mse.size == 0 or not np.all(np.isfinite(mse)):
        return math.inf

    pen = tool_pen_depths(curve)
    pen_mse = PEN_WEIGHT * float(np.mean(pen ** 2)) if pen.size else 0.0
    pen_mm = 1.0e3 * math.sqrt(pen_mse)

    if objective in ("symmetric_mse", "tracked_to_sim_mse", "corr_mse"):
        key = {"symmetric_mse": "mse_symmetric", "tracked_to_sim_mse": "mse_tracked_to_sim",
               "corr_mse": "corr_mse"}[objective]
        if key not in curve:
            return math.inf   # run without --surface-loss-symmetric / -correspondence
        arr = np.asarray(curve[key], dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return math.inf
        return float(arr.mean()) + pen_mse

    base = {
        "mean_mse": float(mse.mean()),
        "mean_rmse_mm": float(rmse_mm.mean()),
        "median_rmse_mm": float(np.median(rmse_mm)),
        "max_rmse_mm": float(rmse_mm.max()),
        "last_rmse_mm": float(rmse_mm[-1]),
    }[objective]
    return base + (pen_mse if objective == "mean_mse" else pen_mm)


SUMMARY_RE = None  # (compiled lazily) fallback parse of the stdout summary line


def _parse_summary_line(text):
    """Recover (mean_rmse_mm, mean_mse) from the sim's own summary print if the
    --surface-loss-out npz is missing (e.g. the run crashed after printing)."""
    global SUMMARY_RE
    if SUMMARY_RE is None:
        import re
        SUMMARY_RE = re.compile(
            r"RMSE mm\s+mean\s+([-\d.eE+]+).*?frame-mean MSE\s+([-\d.eE+]+)",
            re.S,
        )
    m = SUMMARY_RE.search(text or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


# --------------------------------------------------------------------------
# One trial
# --------------------------------------------------------------------------
def build_argv(overrides, frames, extra, *, no_refine=False):
    """``no_refine=True`` drops the -r switch: the no-remeshing reference every
    refinement setting has to beat."""
    argv = [sys.executable, "-m", "mfem.refinement.sim_octopus",
            "--episode", EPISODE,
            "--record-frames", str(int(frames))]
    argv += [a for a in BASE_ARGS if not (no_refine and a == "-r")]
    if EXCLUDE_TOOL_MARGIN_M is not None:
        argv += ["--surface-loss-exclude-tool", repr(float(EXCLUDE_TOOL_MARGIN_M))]
    argv += list(extra)
    for name, spec in overrides.items():
        flag = spec["flag"]
        val = spec["value"]
        if spec.get("type") == "flag":
            # store_true switch: present when truthy, omitted otherwise.
            if val is True or val in ("true", "True", "1", 1):
                argv += [flag]
        else:
            argv += [flag, _fmt(val)]
    return argv


def _fmt(v):
    if isinstance(v, float):
        return repr(v)
    return str(v)


REPEATS = 1   # runs per trial, averaged (see --repeats)


def run_trial(overrides, frames, extra, workdir, timeout, *, no_refine=False):
    """Run one trial = REPEATS sims, return (objective_dict, meta). Candidate
    selection inside refine() resolves ties through atomic races, so identical
    settings differ by a couple of percent between runs; averaging repeats is
    what makes small parameter effects distinguishable from that noise. The
    per-objective spread is reported in meta as ``<objective>_std``."""
    runs = [_run_trial_once(overrides, frames, extra, workdir, timeout,
                            no_refine=no_refine, repeat=k)
            for k in range(max(int(REPEATS), 1))]
    objs = {}
    meta = dict(runs[0][1])
    for o in OBJECTIVES:
        vals = np.array([r[0][o] for r in runs], dtype=np.float64)
        objs[o] = float(vals.mean()) if np.all(np.isfinite(vals)) else math.inf
        meta[f"{o}_std"] = float(vals.std()) if np.all(np.isfinite(vals)) else math.inf
    meta["seconds"] = round(sum(r[1]["seconds"] for r in runs), 1)
    meta["n_frames"] = min(r[1]["n_frames"] for r in runs)
    statuses = [r[1]["status"] for r in runs]
    meta["status"] = "ok" if all(st == "ok" for st in statuses) else "/".join(sorted(set(statuses)))
    meta["repeats"] = len(runs)
    for k in ("tool_pen_rms_mm", "tool_pen_max_mm", "tool_pen_frac", "tool_worst_gap_mm"):
        if k in runs[0][1]:
            meta[k] = float(np.mean([r[1][k] for r in runs]))
    return objs, meta


def _run_trial_once(overrides, frames, extra, workdir, timeout, *, no_refine=False, repeat=0):
    """Run one sim, return (objective_dict, meta)."""
    full = "_".join(f"{k}={overrides[k]['value']}" for k in sorted(overrides)) or "baseline"
    if no_refine:
        full = "norefine" + ("_" + full if overrides else "")
    if repeat:
        full += f"_r{repeat}"
    full = full.replace("/", "-").replace(" ", "")
    # Keep the tag readable but collision-proof (params can exceed a filename):
    # a short hash of the full spec disambiguates truncated names.
    import hashlib  # noqa: PLC0415
    h = hashlib.sha1(full.encode()).hexdigest()[:8]
    tag = (full[:100] + "_" + h) if len(full) > 100 else full
    loss_npz = os.path.join(workdir, f"loss_{tag}.npz")
    rec_npz = os.path.join(workdir, f"rec_{tag}.npz")
    log_path = os.path.join(workdir, f"log_{tag}.txt")
    for p in (loss_npz, rec_npz):
        try:
            os.remove(p)
        except OSError:
            pass

    argv = build_argv(overrides, frames, extra, no_refine=no_refine) + [
        "--surface-loss-out", loss_npz, "--record", rec_npz,
    ]
    t0 = time.time()
    status = "ok"
    # Own process group + SIGKILL on timeout: a sim that has printed its
    # summary but never exits (seen: hung for 16 h at teardown) must not stall
    # the sweep, and subprocess.run's own timeout could not reap it.
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=os.getcwd(), start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        stdout = out + "\n" + err
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        import signal  # noqa: PLC0415
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        stdout = (out or "") + "\n" + (err or "") + "\n[timeout]"
        rc = -9
        # The loss npz is written before the sim's teardown, so a run that
        # completed its frames still scores; only a truly cut-short run is
        # marked below via n_frames.
        status = "timeout"
    dt = time.time() - t0

    with open(log_path, "w") as fh:
        fh.write(" ".join(argv) + "\n\n" + stdout)

    curve = None
    if os.path.exists(loss_npz):
        try:
            with np.load(loss_npz) as d:
                curve = {k: d[k] for k in d.files}
        except Exception:
            curve = None

    n_frames = int(len(curve["mse"])) if curve is not None else 0
    if curve is not None and n_frames > 0:
        objs = {o: reduce_curve(curve, o) for o in OBJECTIVES}
    else:
        parsed = _parse_summary_line(stdout)
        if parsed is not None:
            mean_rmse, mean_mse = parsed
            objs = {"mean_mse": mean_mse, "mean_rmse_mm": mean_rmse,
                    "median_rmse_mm": mean_rmse, "max_rmse_mm": mean_rmse,
                    "last_rmse_mm": mean_rmse, "symmetric_mse": math.inf,
                    "tracked_to_sim_mse": math.inf, "corr_mse": math.inf}
            status = status if status != "ok" else "npz-missing-parsed-stdout"
        else:
            objs = {o: math.inf for o in OBJECTIVES}
            status = f"no-loss(rc={rc})"

    # Penalise a run that stopped well short of the requested frames (blew up).
    if 0 < n_frames < 0.8 * frames:
        for k in objs:
            objs[k] = objs[k] * (frames / max(n_frames, 1))
        status = f"short({n_frames}/{frames})"

    pen = (tool_penetration_stats(curve) if curve is not None
           else {"tool_pen_rms_mm": 0.0, "tool_pen_max_mm": 0.0,
                 "tool_pen_frac": 0.0, "tool_worst_gap_mm": 0.0})
    if pen["tool_pen_max_mm"] > 0.0 and status == "ok":
        status = f"toolpen{pen['tool_pen_max_mm']:.0f}mm"

    meta = {
        "tag": tag, "argv": argv, "seconds": round(dt, 1),
        "n_frames": n_frames, "status": status, "log": log_path,
        "params": {**({"refine": False} if no_refine else {}),
                   **{k: overrides[k]["value"] for k in overrides}},
        **pen,
    }
    return objs, meta


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
def _as_spec(name, spec):
    if isinstance(spec, dict):
        out = dict(spec)
        out.setdefault("flag", "--" + name.replace("_", "-"))
    else:
        out = {"flag": "--" + name.replace("_", "-"), "values": list(spec)}
    if "range" not in out and "values" not in out:
        raise ValueError(f"space entry {name!r} needs 'values' or 'range'")
    return out


def load_space(path):
    """(space, extra sim args) for a built-in name or a JSON file."""
    if path is None:
        path = "default"
    if path in BUILTIN_SPACES:
        raw, extra = BUILTIN_SPACES[path]
        return {k: _as_spec(k, v) for k, v in raw.items()}, list(extra)
    with open(path) as fh:
        raw = json.load(fh)
    return {k: _as_spec(k, v) for k, v in raw.items()}, []


def spec_values(spec):
    """Concrete value list for grid / coordinate walks."""
    if "values" in spec:
        return list(spec["values"])
    lo, hi = float(spec["range"][0]), float(spec["range"][1])
    steps = int(spec.get("steps", 5))
    if spec.get("scale") == "log":
        pts = np.exp(np.linspace(math.log(lo), math.log(hi), steps))
    else:
        pts = np.linspace(lo, hi, steps)
    return [float(x) for x in pts]


def spec_sample(spec):
    """One value for the random strategy."""
    if "values" in spec:
        return random.choice(spec["values"])
    lo, hi = float(spec["range"][0]), float(spec["range"][1])
    if spec.get("scale") == "log":
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    return random.uniform(lo, hi)


def _override(spec, value):
    """One per-trial override entry, carrying the spec's flag + kind."""
    return {"flag": spec["flag"], "value": value, "type": spec.get("type")}


def combo_ok(overrides):
    """True if a full override set satisfies every CONSTRAINTS predicate."""
    params = {k: overrides[k]["value"] for k in overrides}
    for pred, _desc in CONSTRAINTS:
        try:
            if not pred(params):
                return False
        except Exception:
            pass
    return True


def gen_grid(space, max_trials):
    names = list(space)
    combos = itertools.product(*(spec_values(space[n]) for n in names))
    yielded = skipped = 0
    for combo in combos:
        ov = {n: _override(space[n], v) for n, v in zip(names, combo)}
        if not combo_ok(ov):
            skipped += 1
            continue
        if yielded >= max_trials:
            print(f"[sweep] grid truncated at --max-trials {max_trials} "
                  f"({skipped} invalid combos skipped so far)")
            return
        yielded += 1
        yield ov


def gen_random(space, trials):
    names = list(space)
    got = 0
    attempts = 0
    while got < trials and attempts < trials * 50:
        attempts += 1
        ov = {n: _override(space[n], spec_sample(space[n])) for n in names}
        if not combo_ok(ov):
            continue
        got += 1
        yield ov


def coordinate_search(space, passes, frames, extra, workdir, timeout, objective,
                      sink):
    """Greedy coordinate descent from the baseline. Returns (best_params, best_obj)."""
    best_overrides = {}
    ref_objs, ref_meta = run_trial({}, frames, extra, workdir, timeout, no_refine=True)
    sink({}, ref_objs, ref_meta, objective)
    ref_obj = ref_objs[objective]
    print(f"[sweep] reference, no refinement: {objective} = {ref_obj:.6g}"
          f" +-{ref_meta.get(f'{objective}_std', 0.0):.2g}")
    baseline_objs, meta = run_trial({}, frames, extra, workdir, timeout)
    sink({}, baseline_objs, meta, objective)
    best_obj = baseline_objs[objective]
    best_full = {}
    print(f"[sweep] baseline {objective} = {best_obj:.6g}"
          + ("  <-- worse than no refinement" if best_obj > ref_obj else ""))

    for p in range(passes):
        improved = False
        for name in space:
            spec = space[name]
            for val in spec_values(spec):
                if best_full.get(name, None) == val:
                    continue
                trial = dict(best_overrides)
                trial[name] = _override(spec, val)
                objs, meta = run_trial(trial, frames, extra, workdir, timeout)
                sink(trial, objs, meta, objective)
                o = objs[objective]
                flag = ""
                if o < best_obj - 1e-12:
                    best_obj, best_overrides, best_full = o, dict(trial), {
                        **best_full, name: val}
                    improved = True
                    flag = "  <-- best"
                pen = meta.get("tool_pen_rms_mm", 0.0)
                pen_txt = f" toolpen {pen:.1f}mm" if pen > 0 else ""
                std = meta.get(f"{objective}_std", 0.0)
                std_txt = f" +-{std:.2g}" if std else ""
                print(f"[sweep] p{p} {name}={val!r:>28}  {objective}={o:.6g}{std_txt}"
                      f"{pen_txt}  ({meta['status']}, {meta['seconds']}s){flag}")
        if not improved:
            print(f"[sweep] pass {p}: no improvement, stopping")
            break
    verdict = ("beats" if best_obj < ref_obj else "does NOT beat")
    print(f"[sweep] best refined {objective} = {best_obj:.6g} {verdict} the "
          f"no-refinement reference {ref_obj:.6g}")
    return best_full, best_obj


# --------------------------------------------------------------------------
# Result sink / reporting
# --------------------------------------------------------------------------
class ResultLog:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.seen = set()
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    self.rows.append(r)
                    self.seen.add(r["key"])
            print(f"[sweep] resumed {len(self.rows)} trials from {path}")
        self._fh = open(path, "a")

    @staticmethod
    def key(params):
        return json.dumps(params, sort_keys=True, default=str)

    def record(self, params, objs, meta, objective):
        k = self.key(params)
        row = {"key": k, "params": params, "objective_name": objective,
               "objective": objs[objective], **{f"obj_{o}": v for o, v in objs.items()},
               "status": meta["status"], "n_frames": meta["n_frames"],
               "seconds": meta["seconds"], "log": meta["log"],
               "objective_std": meta.get(f"{objective}_std", 0.0),
               "repeats": meta.get("repeats", 1),
               **{p: meta[p] for p in
                  ("tool_pen_rms_mm", "tool_pen_max_mm", "tool_pen_frac",
                   "tool_worst_gap_mm") if p in meta}}
        self.rows.append(row)
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()

    def already(self, params):
        return self.key(params) in self.seen

    def close(self):
        self._fh.close()

    def write_sorted_csv(self, objective):
        import csv
        csv_path = os.path.splitext(self.path)[0] + "_sorted.csv"
        rows = sorted(self.rows, key=lambda r: (not math.isfinite(r["objective"]),
                                                r["objective"]))
        if not rows:
            return None
        param_names = sorted({p for r in rows for p in r["params"]})
        pen_cols = ["tool_pen_rms_mm", "tool_pen_max_mm", "tool_pen_frac",
                    "tool_worst_gap_mm"]
        cols = (["objective"] + [f"obj_{o}" for o in OBJECTIVES]
                + ["status", "n_frames", "seconds"]
                + pen_cols + param_names + ["log"])
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow(
                    [f"{r['objective']:.6g}"]
                    + [f"{r.get('obj_' + o, ''):.6g}" if isinstance(r.get('obj_' + o), float) else ""
                       for o in OBJECTIVES]
                    + [r["status"], r["n_frames"], r["seconds"]]
                    + [f"{r[c]:.3g}" if isinstance(r.get(c), (int, float)) else ""
                       for c in pen_cols]
                    + [r["params"].get(p, "") for p in param_names]
                    + [r["log"]]
                )
        return csv_path


# --------------------------------------------------------------------------
def main():
    global PEN_WEIGHT, PEN_TOL_M, EXCLUDE_TOOL_MARGIN_M, EPISODE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", choices=tuple(EPISODES), default="octopus",
                    help="which PokeFlex episode every trial simulates "
                         "(sim_octopus --episode; default octopus).")
    ap.add_argument("--strategy", choices=("coordinate", "random", "grid"),
                    default="coordinate")
    ap.add_argument("--frames", type=int, default=45,
                    help="--record-frames per trial (shorter = faster, noisier)")
    ap.add_argument("--passes", type=int, default=2,
                    help="coordinate strategy: number of sweeps over the space")
    ap.add_argument("--trials", type=int, default=40,
                    help="random strategy: number of sampled combinations")
    ap.add_argument("--max-trials", type=int, default=400,
                    help="grid strategy: hard cap on the product size")
    ap.add_argument("--objective", default="symmetric_mse", choices=OBJECTIVES,
                    help="symmetric_mse (default): mean of sim->tracked and "
                         "tracked->sim MSE with the tool region excluded; "
                         "tracked_to_sim_mse: the density-independent direction "
                         "only; corr_mse: material-point (first-frame "
                         "correspondence) MSE, also density independent and "
                         "sensitive to sliding; the rest are the old "
                         "sim->tracked reductions")
    ap.add_argument("--repeats", type=int, default=1,
                    help="sims per trial, objective averaged (identical settings "
                         "differ ~2%% run to run; use 3+ for small effects)")
    ap.add_argument("--space", default=None,
                    help="JSON search-space override, or a built-in name: "
                         + " / ".join(f"'{k}'" for k in BUILTIN_SPACES))
    ap.add_argument("--reference-recording", default=None, metavar="REC.npz",
                    help="score every trial against this recording (a converged "
                         "fine-mesh run of the same episode) instead of the tracked "
                         "surface, and drop the tool exclusion: measures "
                         "discretisation error only")
    ap.add_argument("--exclude-tool-margin", default=EXCLUDE_TOOL_MARGIN_M,
                    help="metres; tracked vertices closer than this to the poker "
                         "(negative = inside) are ignored by the loss, and so are "
                         "sim vertices that close in the sim->tracked direction. "
                         "'none' disables.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel trials (GPU-bound; 1 is usually right)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-trial wall-clock limit, seconds")
    ap.add_argument("--out", default="sweep_results.jsonl",
                    help="JSON-lines result log (appended; resumes)")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for per-trial npz/logs "
                         "(default: <out>_work/)")
    ap.add_argument("--sim-arg", action="append", default=[],
                    help="extra literal arg forwarded to sim_octopus, appended "
                         "after BASE_ARGS and before the swept overrides "
                         "(repeatable), e.g. --sim-arg --gravity --sim-arg 0")
    ap.add_argument("--tool-pen-weight", type=float, default=PEN_WEIGHT,
                    help="penalty weight on the tool sinking into the mesh "
                         "(0 disables). mean_mse += w*mean(pen_m^2); the *_mm "
                         "objectives += 1e3*sqrt(w*mean(pen_m^2)).")
    ap.add_argument("--tool-pen-tol", type=float, default=PEN_TOL_M,
                    help="capsule->particle gap (m) below which the tool counts "
                         "as submerged; only depth past this is penalised.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned trials and exit")
    args = ap.parse_args()

    EPISODE = args.episode
    PEN_WEIGHT = float(args.tool_pen_weight)
    PEN_TOL_M = float(args.tool_pen_tol)
    EXCLUDE_TOOL_MARGIN_M = (None if str(args.exclude_tool_margin).lower() == "none"
                             else float(args.exclude_tool_margin))

    global REPEATS
    REPEATS = max(int(args.repeats), 1)
    space, space_extra = load_space(args.space)
    extra = space_extra + list(args.sim_arg)
    if args.reference_recording:
        extra += ["--surface-loss-reference", args.reference_recording]
        EXCLUDE_TOOL_MARGIN_M = None
    workdir = args.workdir or (os.path.splitext(args.out)[0] + "_work")
    os.makedirs(workdir, exist_ok=True)

    print(f"[sweep] strategy={args.strategy} objective={args.objective} "
          f"frames={args.frames} repeats={REPEATS} jobs={args.jobs}"
          + (f" space-args={' '.join(space_extra)}" if space_extra else ""))
    print(f"[sweep] tool-submersion penalty: weight {PEN_WEIGHT:g}, "
          f"tolerance {PEN_TOL_M * 1e3:g} mm"
          + ("  (disabled)" if PEN_WEIGHT == 0.0 else ""))
    print(f"[sweep] space: " + ", ".join(
        f"{k}{spec_values(space[k])}" for k in space))
    print(f"[sweep] work dir: {workdir}")

    if args.dry_run:
        if args.strategy == "grid":
            plan = list(gen_grid(space, args.max_trials))
        elif args.strategy == "random":
            plan = list(gen_random(space, args.trials))
        else:
            plan = [{}]  # coordinate walks adaptively
            print("[sweep] coordinate strategy is adaptive; baseline first, then "
                  "one param at a time")
        for ov in plan:
            print("  " + " ".join(build_argv(ov, args.frames, extra)))
        print(f"[sweep] {len(plan)} trial(s) planned"
              + ("" if args.strategy != "coordinate" else " (+ adaptive)"))
        return

    log = ResultLog(args.out)

    def sink(params_over, objs, meta, objective):
        log.record(meta["params"], objs, meta, objective)

    t_start = time.time()
    if args.strategy == "coordinate":
        best, best_obj = coordinate_search(
            space, args.passes, args.frames, extra, workdir, args.timeout,
            args.objective, sink)
        print(f"\n[sweep] best {args.objective} = {best_obj:.6g}  params: {best}")
    else:
        gen = (gen_grid(space, args.max_trials) if args.strategy == "grid"
               else gen_random(space, args.trials))
        combos = [c for c in gen]
        # dedup + resume
        todo = []
        for ov in combos:
            params = {k: ov[k]["value"] for k in ov}
            if log.already(params):
                continue
            todo.append(ov)
        print(f"[sweep] {len(todo)} trial(s) to run "
              f"({len(combos) - len(todo)} already in {args.out})")

        def _job(ov):
            objs, meta = run_trial(ov, args.frames, extra, workdir, args.timeout)
            return ov, objs, meta

        if args.jobs <= 1:
            for i, ov in enumerate(todo):
                _, objs, meta = _job(ov)
                sink(ov, objs, meta, args.objective)
                pen = meta.get("tool_pen_rms_mm", 0.0)
                pen_txt = f" toolpen {pen:.1f}mm" if pen > 0 else ""
                print(f"[sweep] {i+1}/{len(todo)}  {meta['tag']}  "
                      f"{args.objective}={objs[args.objective]:.6g}{pen_txt}  "
                      f"({meta['status']}, {meta['seconds']}s)")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(_job, ov) for ov in todo]
                for i, fut in enumerate(concurrent.futures.as_completed(futs)):
                    ov, objs, meta = fut.result()
                    sink(ov, objs, meta, args.objective)
                    print(f"[sweep] {i+1}/{len(todo)}  {meta['tag']}  "
                          f"{args.objective}={objs[args.objective]:.6g}  "
                          f"({meta['status']}, {meta['seconds']}s)")

    log.close()
    csv_path = log.write_sorted_csv(args.objective)

    finite = [r for r in log.rows if math.isfinite(r["objective"])]
    finite.sort(key=lambda r: r["objective"])
    print(f"\n[sweep] {len(log.rows)} trials in {time.time() - t_start:.0f}s"
          f"  ({len(finite)} finite)")
    print(f"[sweep] top 10 by {args.objective} (penalised):")
    for r in finite[:10]:
        pen = r.get("tool_pen_rms_mm", 0.0) or 0.0
        pen_txt = (f" toolpen {pen:.1f}/{r.get('tool_pen_max_mm', 0.0):.0f}mm"
                   f" {100 * (r.get('tool_pen_frac', 0.0) or 0.0):.0f}%f"
                   if pen > 0 else "")
        print(f"  {r['objective']:.6g}  "
              f"rmse_mm mean {r.get('obj_mean_rmse_mm', float('nan')):.2f} "
              f"max {r.get('obj_max_rmse_mm', float('nan')):.2f}{pen_txt}  "
              f"[{r['status']}]  {r['params']}")
    if csv_path:
        print(f"[sweep] sorted table: {csv_path}")
        print(f"[sweep] raw log:      {args.out}")


if __name__ == "__main__":
    main()
