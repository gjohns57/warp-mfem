"""Per-episode configuration for the PokeFlex tracked poke scenarios.

``sim_octopus.py`` and its helpers (``make_octopus_tetmesh``,
``make_octopus_rest``, ``replay_octopus``, ``sweep_octopus``) were originally
wired to a single tracked episode, ``PlushOctopus_T1``. The PokeFlex dataset at
``~/Documents/PokeFlex_Tracked-mesh-episodes-and-gifs/PokeFlex_Tracked/`` ships
three more episodes in the identical file layout -- ``FoamDice_T1``,
``PlushTurtle_T1``, ``ToiletPaperRoll_T1`` -- differing only in a handful of
identified / geometric constants. This module is the single source of truth for
those constants; the sim modules take an ``--episode`` flag (default
``octopus``) and read everything episode-specific from here.

Only the standard library + numpy are imported, so the lightweight
``replay_octopus`` / ``surface_loss`` tools can import this without pulling in
Warp / Newton.

Values are copied from each episode's ``manifest.json`` (``T``),
``physics/physics_params.json`` (``E``, ``nu``, ``rho``, ``tip_r``, ``tip_len``,
``dyn_fit.eta``, ``lattice_dx``) and ``physics/setup_cache.npz`` (``table_y``).
The octopus entry reproduces the old ``OCTO_*`` module constants in
``sim_octopus.py`` verbatim, so ``--episode octopus`` (the default) is
byte-identical to the pre-refactor behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# ``src/mfem/refinement/pokeflex_episodes.py`` -> repo root is 3 up.
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

# The 4-object tool-trajectory drop unzips with a doubled directory level.
_TOOL_TRAJ_DIR = (
    WORKSPACE_ROOT
    / "models"
    / "PokeFlex_Tracked-tool-trajectories-4objects"
    / "PokeFlex_Tracked-tool-trajectories-4objects"
)

# Where the raw tracked episodes live when there is no workspace symlink.
_DATASET_FALLBACK = (
    Path.home()
    / "Documents"
    / "PokeFlex_Tracked-mesh-episodes-and-gifs"
    / "PokeFlex_Tracked"
)

# The fitted poking-tool tip geometry is identical across all four episodes
# (same physical indenter): tip_r / tip_len from physics/physics_params.json.
TIP_R = 0.014842664490570312
TIP_LEN = 0.017289941100636492

# Contact / table friction default. NOT identified -- the fitted tool/body and
# body/table coefficients are both ~0.3, but at that level the free body skates
# on the table under the laterally-sweeping tool, so the shared sim default is
# cranked up to keep it planted (see sim_octopus --friction-mu).
SIM_FRICTION_MU = 0.9

# Membrane-shell fabric thickness (m). A plush-fabric guess, not identified --
# the PokeFlex identification never fit a shell (all shell params zeroed).
SHELL_THICKNESS = 0.002

_STIFFNESS_KINDS_DEFAULT = {
    "field": "stiffness_field.npy",
    "dyn": "stiffness_field_dyn.npy",
    "sota": "stiffness_field_sota.npy",
}
# Turtle / TP-roll: the plain quasi-static and SOTA stiffness fields were not
# saved; only the dyn fit and a high-fidelity field. Map field + sota onto the
# high-fidelity one.
_STIFFNESS_KINDS_HIFI = {
    "field": "stiffness_field_high_fidelity.npy",
    "dyn": "stiffness_field_dyn.npy",
    "sota": "stiffness_field_high_fidelity.npy",
}


def _lame_from_youngs(E: float, nu: float) -> tuple[float, float]:
    """(E, nu) -> (mu, lambda), the isotropic-linear-elastic Lame parameters."""
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lmbda


@dataclass(frozen=True)
class PokeflexEpisode:
    """Everything ``sim_octopus.py`` needs that varies between tracked episodes.

    Lengths in metres, PokeFlex world frame, +Y up. ``mesh_paths`` /
    ``rest_npz`` / ``stiffness_field_npz`` are workspace-relative strings so they
    read the same as the old hard-coded ``models/octopus_*`` literals.
    """

    key: str
    dataset_dirname: str
    frame_count: int
    initial_frame: int
    table_y: float
    youngs_pa: float
    poisson: float
    density: float = 80.0
    damp_eta: float = 0.0
    lattice_dx: float = 0.02
    shell_thickness: float = SHELL_THICKNESS
    tip_r: float = TIP_R
    tip_len: float = TIP_LEN
    # Tool origin (X, Z) at trajectory frame 0. Only used for the manual-poker
    # GUI default pose (--tool-trajectory none); the recorded path overrides it.
    poke_xz_frame0: tuple[float, float] = (0.0, 0.0)
    sim_friction_mu: float = SIM_FRICTION_MU
    capture_fps: int = 30
    stiffness_field_kinds: dict = field(default_factory=lambda: dict(_STIFFNESS_KINDS_DEFAULT))

    # -- derived material -------------------------------------------------
    @property
    def mu_pa(self) -> float:
        return _lame_from_youngs(self.youngs_pa, self.poisson)[0]

    @property
    def lambda_pa(self) -> float:
        return _lame_from_youngs(self.youngs_pa, self.poisson)[1]

    # -- generated-artifact locations (workspace-relative strings) --------
    @property
    def mesh_paths(self) -> dict:
        """``--mesh`` resolution -> initial tet-mesh .npz. ``fine`` is the
        ``full`` fTetWild tag, matching the octopus naming."""
        return {
            "coarse": f"models/{self.key}_initial_coarse.npz",
            "medium": f"models/{self.key}_initial_medium.npz",
            "fine": f"models/{self.key}_initial_full.npz",
        }

    @property
    def mesh_default(self) -> str:
        return "coarse"

    @property
    def stiffness_field_npz(self) -> str:
        return f"models/{self.key}_stiffness_field.npz"

    def rest_npz(self, mesh_key: str) -> str:
        """Un-sagged rest-shape .npz built by ``make_octopus_rest`` for one
        ``--mesh`` resolution."""
        return f"models/{self.key}_rest_{mesh_key}.npz"

    # -- raw dataset locations ------------------------------------------
    def dataset_root(self) -> Path:
        """Directory holding this episode's tracked data. Prefers a workspace
        symlink (``<repo>/<dataset_dirname>``, as the octopus has), else the
        ``~/Documents`` dataset drop."""
        local = WORKSPACE_ROOT / self.dataset_dirname
        if local.exists():
            return local
        return _DATASET_FALLBACK / self.dataset_dirname

    @property
    def tool_trajectory_npz(self) -> str:
        return str(_TOOL_TRAJ_DIR / self.dataset_dirname / "tool_trajectory.npz")

    def tracked_surface_npy(self) -> str:
        """Fused surface-tracking trajectory for the ``--tracked-surface``
        overlay (``template_canonical.obj`` / ``valid_mask_canonical.npy`` sit
        next to it)."""
        return str(self.dataset_root() / "mesh_trajectories_canonical.npy")

    def with_initial_frame(self, frame: int) -> "PokeflexEpisode":
        return replace(self, initial_frame=int(frame))


# ---------------------------------------------------------------------------
# Registry. octopus reproduces the old sim_octopus OCTO_* constants exactly.
# The three new episodes' initial_frame values are placeholders pending a
# hand-picked "sprung back, just after the first poke" frame per episode; update
# them here and rebuild that episode's meshes (make_octopus_tetmesh --episode X
# --frame N ; make_octopus_rest --episode X --mesh coarse).
# ---------------------------------------------------------------------------
EPISODES: dict[str, PokeflexEpisode] = {
    "octopus": PokeflexEpisode(
        key="octopus",
        dataset_dirname="PlushOctopus_T1",
        frame_count=155,
        initial_frame=19,
        table_y=0.278365,
        youngs_pa=2832.182628857769,
        poisson=0.42625,
        density=80.0,
        damp_eta=26.8368007779163,
        lattice_dx=0.02,
        poke_xz_frame0=(0.49361, 0.11301),
    ),
    "dice": PokeflexEpisode(
        key="dice",
        dataset_dirname="FoamDice_T1",
        frame_count=150,
        initial_frame=33,  # PLACEHOLDER -- pending hand-picked frame
        table_y=0.278134,
        youngs_pa=8610.466292144307,
        poisson=0.23848312501795593,
        density=80.0,
        damp_eta=14.728766875109606,
        lattice_dx=0.02,
        poke_xz_frame0=(0.47847, 0.10732),
    ),
    "turtle": PokeflexEpisode(
        key="turtle",
        dataset_dirname="PlushTurtle_T1",
        frame_count=155,
        initial_frame=23,  # PLACEHOLDER -- pending hand-picked frame
        table_y=0.279295,
        youngs_pa=8485.623424650852,
        poisson=0.35562499999999997,
        density=80.0,
        damp_eta=76.48364016300238,
        lattice_dx=0.02,
        poke_xz_frame0=(0.47622, 0.13426),
        stiffness_field_kinds=dict(_STIFFNESS_KINDS_HIFI),
    ),
    "tp_roll": PokeflexEpisode(
        key="tp_roll",
        dataset_dirname="ToiletPaperRoll_T1",
        frame_count=150,
        initial_frame=27,  # PLACEHOLDER -- pending hand-picked frame
        table_y=0.280848,
        youngs_pa=9486.832980505133,
        poisson=0.31500000000000006,
        density=80.0,
        damp_eta=452.97362829795105,
        lattice_dx=0.02,
        poke_xz_frame0=(0.48109, 0.10735),
        stiffness_field_kinds=dict(_STIFFNESS_KINDS_HIFI),
    ),
}

DEFAULT_EPISODE = "octopus"


def add_episode_arg(parser) -> None:
    """Add ``--episode`` to an argparse parser (choices = the registry keys)."""
    parser.add_argument(
        "--episode",
        choices=tuple(EPISODES),
        default=DEFAULT_EPISODE,
        help="Which PokeFlex tracked episode to simulate (default "
             f"{DEFAULT_EPISODE}). Selects the identified material (E, nu, "
             "damping), table height, tracked-surface overlay, recorded tool "
             "path and the models/<episode>_* tet meshes / rest shapes.",
    )


def get_episode(spec) -> PokeflexEpisode:
    """Resolve ``spec`` to a :class:`PokeflexEpisode`.

    ``spec`` may be a registry key string, an argparse ``Namespace`` with an
    ``.episode`` attribute (falls back to :data:`DEFAULT_EPISODE` when absent or
    ``None``), or an already-resolved :class:`PokeflexEpisode`.
    """
    if isinstance(spec, PokeflexEpisode):
        return spec
    if spec is None:
        key = DEFAULT_EPISODE
    elif isinstance(spec, str):
        key = spec
    else:  # argparse Namespace or similar
        key = getattr(spec, "episode", None) or DEFAULT_EPISODE
    try:
        return EPISODES[key]
    except KeyError:
        raise KeyError(
            f"unknown PokeFlex episode {key!r}; choices are {sorted(EPISODES)}"
        ) from None


def _selftest() -> None:
    """Sanity-check the registry against the identified values (see the plan
    table). Run with ``python -m mfem.refinement.pokeflex_episodes``."""
    expect = {
        "octopus": (2832.182628857769, 0.42625, 992.88, 5738.49),
        "dice": (8610.466292144307, 0.23848312501795593, 3476.21, 3170.04),
        "turtle": (8485.623424650852, 0.35562499999999997, 3129.78, 7709.29),
        "tp_roll": (9486.832980505133, 0.315, 3607.16, 6141.92),
    }
    for key, ep in EPISODES.items():
        E, nu, mu_approx, lam_approx = expect[key]
        assert abs(ep.youngs_pa - E) < 1e-6, key
        assert abs(ep.poisson - nu) < 1e-9, key
        assert abs(ep.mu_pa - mu_approx) < 1.0, (key, ep.mu_pa)
        assert abs(ep.lambda_pa - lam_approx) < 1.0, (key, ep.lambda_pa)
        print(
            f"{key:8s} E={ep.youngs_pa:9.2f} nu={ep.poisson:.5f} "
            f"mu={ep.mu_pa:8.2f} lambda={ep.lambda_pa:9.2f}  "
            f"table_y={ep.table_y:.6f}  frame0={ep.initial_frame}  "
            f"dataset={'ok' if ep.dataset_root().exists() else 'MISSING'}"
        )
    print("pokeflex_episodes selftest OK")


if __name__ == "__main__":
    _selftest()
