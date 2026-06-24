#!/usr/bin/env python
"""Thin CLI: load config, run detect+track over the eval sequence with each tracker."""

from __future__ import annotations

from pathlib import Path

from mot_pipeline.tracking.track import track_eval_sequence
from mot_pipeline.utils.cli import build_arg_parser, resolve_config


def main() -> None:
    """Parse args, load config, and run tracking for every configured tracker."""
    parser = build_arg_parser("Detect+track the eval sequence with each configured tracker.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Detector checkpoint to track with. Defaults to <weights_dir>/best.pt.",
    )
    args = parser.parse_args()
    config = resolve_config(args)
    weights_path = args.weights if args.weights else config.paths.weights_dir / "best.pt"

    results = track_eval_sequence(config, weights_path)

    for result in results:
        print(
            f"{result.tracker:10s} detections={result.n_detections:6d} tracks={result.n_tracks:4d} "
            f"-> {result.results_path}, {result.video_path}"
        )


if __name__ == "__main__":
    main()
