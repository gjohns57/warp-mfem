"""Bundle a PokeFlex episode's identified per-tet stiffness field into the
``models/<episode>_stiffness_field.npz`` that ``sim_octopus.py
--per-tet-material`` transfers onto the sim mesh.

The identification fits a spatially-varying Young's modulus -- one E per tet of
its own lattice-derived tet mesh. That mesh (``physics/setup_cache.npz``
``nodes`` / ``tets``) plus the per-tet moduli
(``physics/stiffness_field*.npy``) and the identified ``nu`` /
viscoelastic ``eta`` (``physics/physics_params.json``) are packed here into one
small ``.npz`` with the schema ``_per_tet_lame_from_field`` expects:

    nodes    (N, 3) float32     identification-mesh vertices (PokeFlex world m)
    tets     (M, 4) int32       identification-mesh tetrahedra
    E_field  (M,)   float32     quasi-static per-tet Young's modulus, Pa
    E_dyn    (M,)   float32     viscoelastic dyn-fit per-tet E, Pa
    E_sota   (M,)   float32     SOTA-fit per-tet E, Pa
    nu       scalar float64     identified Poisson ratio
    eta_dyn  scalar float64     dyn-fit Kelvin-Voigt eta, Ns/m
    source   str                provenance string

octopus / dice saved ``stiffness_field.npy`` (quasi-static) and
``stiffness_field_sota.npy``; turtle / tp_roll saved only the dyn fit and a
``stiffness_field_high_fidelity.npy`` -- for those, ``E_field`` and ``E_sota``
both map onto the high-fidelity field (see
``PokeflexEpisode.stiffness_field_kinds``).

Examples
--------
    python -m mfem.refinement.models.make_pokeflex_stiffness_field --episode dice
    python -m mfem.refinement.models.make_pokeflex_stiffness_field --episode octopus  # regenerate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mfem.refinement.pokeflex_episodes import EPISODES, get_episode

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = WORKSPACE_ROOT / "models"


def build(episode_key: str, out: Path | None = None) -> Path:
    ep = get_episode(episode_key)
    physics = ep.dataset_root() / "physics"
    setup_cache = physics / "setup_cache.npz"
    params_json = physics / "physics_params.json"
    for p in (setup_cache, params_json):
        if not p.exists():
            raise SystemExit(f"{p} not found (is the {ep.key} dataset present?)")

    sc = np.load(setup_cache, allow_pickle=True)
    nodes = np.asarray(sc["nodes"], dtype=np.float32)
    tets = np.asarray(sc["tets"], dtype=np.int32)
    n_tets = tets.shape[0]

    params = json.loads(params_json.read_text())
    nu = float(params["nu"])
    eta_dyn = float(params.get("dyn_fit", {}).get("eta", 0.0))

    fields: dict[str, np.ndarray] = {}
    for out_key, npy_name in {
        "E_field": ep.stiffness_field_kinds["field"],
        "E_dyn": ep.stiffness_field_kinds["dyn"],
        "E_sota": ep.stiffness_field_kinds["sota"],
    }.items():
        npy = physics / npy_name
        if not npy.exists():
            raise SystemExit(f"{npy} not found")
        arr = np.asarray(np.load(npy), dtype=np.float32).reshape(-1)
        if arr.shape[0] != n_tets:
            raise SystemExit(
                f"{npy_name} has {arr.shape[0]} entries but setup_cache has "
                f"{n_tets} tets"
            )
        fields[out_key] = arr

    out = Path(out) if out is not None else MODELS_DIR / f"{ep.key}_stiffness_field.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = (
        f"PokeFlex_Tracked/{ep.dataset_dirname} physics/{{setup_cache.npz,"
        f"{ep.stiffness_field_kinds['field']},"
        f"{ep.stiffness_field_kinds['dyn']},"
        f"{ep.stiffness_field_kinds['sota']},physics_params.json}}"
    )
    np.savez_compressed(
        out,
        nodes=nodes,
        tets=tets,
        E_field=fields["E_field"],
        E_dyn=fields["E_dyn"],
        E_sota=fields["E_sota"],
        nu=np.float64(nu),
        eta_dyn=np.float64(eta_dyn),
        source=str(src),
    )

    def _rng(a: np.ndarray) -> str:
        return f"{a.min():.0f}..{np.median(a):.0f}..{a.max():.0f}"

    print(
        f"{ep.key}: {nodes.shape[0]} id nodes / {n_tets} id tets  ->  {out}\n"
        f"  E_field (Pa) {_rng(fields['E_field'])}   "
        f"E_dyn {_rng(fields['E_dyn'])}   E_sota {_rng(fields['E_sota'])}\n"
        f"  nu {nu:.5f}   eta_dyn {eta_dyn:.4f} Ns/m"
        + ("   [field & sota = high-fidelity field]"
           if ep.stiffness_field_kinds["field"] == ep.stiffness_field_kinds["sota"]
           else "")
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", choices=tuple(EPISODES), default="octopus")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .npz (default models/<episode>_stiffness_field.npz).")
    ap.add_argument("--all", action="store_true",
                    help="build the bundle for every episode.")
    args = ap.parse_args()

    if args.all:
        for key in EPISODES:
            build(key)
    else:
        build(args.episode, args.out)


if __name__ == "__main__":
    main()
