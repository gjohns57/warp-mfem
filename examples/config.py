"""Shared defaults for the example scripts, read from ``examples/config.yaml``.

Every example calls :func:`apply_config` on its argparse parser right after
building it. The YAML values become the parser's *defaults*, so the precedence
is: command line > ``<script>`` section > ``common`` section > the default
written in the script's ``add_argument``.

``common`` applies to every example that has an option of that name (keys are
the option names, ``-`` or ``_``); a section named after the script
(``octopus_refinement``, ``sweep_octopus``, ...) overrides it for that script.
Options absent from the YAML keep the script's built-in default, so an option
whose default is a ``None`` sentinel resolved later (e.g. the per-episode
values in ``octopus_refinement``) is left alone unless set explicitly.

Pick another file with ``--config PATH`` or ``$MFEM_EXAMPLES_CONFIG``.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
ENV_VAR = "MFEM_EXAMPLES_CONFIG"


def config_path(argv=None) -> Path:
    """``--config`` from ``argv`` (default ``sys.argv``), else ``$MFEM_EXAMPLES_CONFIG``, else the default file."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = known.config or os.environ.get(ENV_VAR) or DEFAULT_CONFIG
    return Path(path)


def load_config(path=None) -> dict:
    """Parse the YAML file into ``{section: {option_dest: value}}`` (empty if missing)."""
    path = Path(path) if path is not None else config_path()
    if not path.is_file():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {sec: {str(k).replace("-", "_"): v for k, v in (vals or {}).items()}
            for sec, vals in raw.items()}


def apply_config(parser: argparse.ArgumentParser, script: str, argv=None) -> argparse.ArgumentParser:
    """Install the YAML values for ``script`` as defaults on ``parser`` and add ``--config``."""
    if not any("--config" in a.option_strings for a in parser._actions):
        parser.add_argument("--config", default=None, metavar="YAML",
                            help=f"YAML file of shared example defaults (default: {DEFAULT_CONFIG.name}, "
                                 f"or ${ENV_VAR})")
    path = config_path(argv)
    os.environ[ENV_VAR] = str(path)  # nested parsers (closeup/slowmo -> octopus sim) read the same file
    cfg = load_config(path)
    dests = {a.dest for a in parser._actions}
    own = cfg.get(script, {})
    for key in own:
        if key not in dests:
            print(f"[config] {path.name}: section '{script}' has unknown option '{key}'", file=sys.stderr)
    values = {k: v for k, v in {**cfg.get("common", {}), **own}.items() if k in dests}
    parser.set_defaults(**values)
    return parser
