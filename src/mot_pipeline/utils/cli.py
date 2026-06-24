"""Shared argument parsing for the thin CLI wrappers under ``scripts/``.

Every script follows the same shape: parse args -> load+validate config ->
call a library function. This module owns the first two steps so each
script only does the third.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mot_pipeline.config import Config, load_config


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with the standard config-loading flags.

    Args:
        description: One-line description shown in ``--help``.

    Returns:
        A parser accepting ``--config``, ``--override``, and repeatable
        ``--set key=value`` flags.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="Path to the base YAML config.")
    parser.add_argument(
        "--override", type=Path, default=None, help="Optional YAML file merged on top of --config."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.subkey=value",
        help="Ad hoc config override, repeatable.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Build a validated :class:`Config` from parsed CLI arguments.

    Args:
        args: Namespace produced by a parser built with :func:`build_arg_parser`.

    Returns:
        The validated, fully merged :class:`Config`.
    """
    return load_config(args.config, override_path=args.override, cli_overrides=args.overrides)
